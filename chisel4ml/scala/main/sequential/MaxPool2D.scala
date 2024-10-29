/*
 * Copyright 2022 Computer Systems Department, Jozef Stefan Insitute
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *    https://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
package chisel4ml.sequential

import chisel3._
import chisel3.util._
import chisel4ml.combinational.MaxPoolOperation
import chisel4ml.compute.OrderCompute
import chisel4ml.implicits._
import chisel4ml.logging.HasParameterLogging
import chisel4ml.util.risingEdge
import chisel4ml.{HasAXIStream, HasAXIStreamParameters, HasLayerWrapSeq}
import interfaces.amba.axis._
import lbir.{LayerWrap, MaxPool2DConfig}
import org.chipsalliance.cde.config.Parameters
import services.Accelerator

trait HasMaxPoolParameters extends HasAXIStreamParameters with HasLayerWrapSeq {
  val p: Parameters
  val cfg: MaxPool2DConfig =
    LayerWrap.LayerWrapTypeMapper.toCustom(_cfg.head.asMessage).get.asInstanceOf[MaxPool2DConfig]
  val maxPoolSize = cfg.input.width / cfg.output.width
  val shiftRegsSize = cfg.input.width * maxPoolSize - (cfg.input.width - maxPoolSize)
  require(_cfg.length == 1)
  require(cfg.output.width * maxPoolSize == cfg.input.width)
  require(cfg.output.height * maxPoolSize == cfg.input.height, f"${cfg.input} / ${cfg.output} == $maxPoolSize")
}

/* MaxPool2D
 *
 * A buffer holds a line maxPoolSize rows of a single channel of the input tensor.
 * A series of combinational max selectors then selects the max element at appropriate
 * moments.
 *
 */
class MaxPool2D(val oc: OrderCompute)(implicit val p: Parameters)
    extends Module
    with HasAXIStream
    with HasLayerWrapSeq
    with HasAXIStreamParameters
    with HasMaxPoolParameters
    with HasParameterLogging {
  logParameters
  val inStream = IO(Flipped(AXIStream(oc.genT, numBeatsIn)))
  val outStream = IO(AXIStream(oc.genT, numBeatsOut))

  object InputBufferState extends ChiselEnum {
    val sEMPTY = Value(0.U)
    val sREAD_WORD = Value(1.U)
    val sSTALL = Value(2.U)
  }
  val state = RegInit(InputBufferState.sEMPTY)

  val inputsBuffer = RegEnable(inStream.bits, inStream.fire)
  val outputsBuffer = Reg(Vec(numBeatsIn, oc.genT))

  val (_, channelElementsCounterWrap) =
    Counter(state === InputBufferState.sREAD_WORD, cfg.input.numActiveParams(depthwise = true))
  val (widthCounter, widthCounterWrap) =
    Counter(0 until cfg.input.width, state === InputBufferState.sREAD_WORD, channelElementsCounterWrap)
  val (heightCounter, heightCounterWrap) =
    Counter(0 until cfg.input.height, widthCounterWrap, channelElementsCounterWrap)
  val (totalInputElements, totalInputElementsWrap) =
    Counter(state === InputBufferState.sREAD_WORD, cfg.input.numParams)
  val (transactionsCounter, transactionsCounterWrap) =
    Counter(0 until cfg.input.numTransactions(numBeatsIn), inStream.fire)
  val (inputBufferCounter, inputBufferCounterWrap) =
    Counter(0 until numBeatsIn, state === InputBufferState.sREAD_WORD, totalInputElementsWrap)
  dontTouch(totalInputElements)
  dontTouch(transactionsCounter)

  when(state === InputBufferState.sEMPTY && inStream.fire) {
    state := InputBufferState.sREAD_WORD
  }.elsewhen(state === InputBufferState.sREAD_WORD && (inputBufferCounterWrap || totalInputElementsWrap)) {
    when(!outStream.ready) {
      state := InputBufferState.sSTALL
    }.otherwise {
      state := InputBufferState.sEMPTY
    }
  }.elsewhen(state === InputBufferState.sSTALL && outStream.ready) {
    state := InputBufferState.sEMPTY
  }

  inStream.ready := state === InputBufferState.sEMPTY
  val startOfMaxPoolWindow = ((heightCounter % maxPoolSize.U === 0.U) &&
    (widthCounter % maxPoolSize.U === 0.U) &&
    state === InputBufferState.sREAD_WORD)

  val inputAndValid = Wire(Valid(oc.genT))
  inputAndValid.bits := inputsBuffer(inputBufferCounter)
  inputAndValid.valid := startOfMaxPoolWindow
  val shiftRegs = ShiftRegisters(inputAndValid, shiftRegsSize, state =/= InputBufferState.sEMPTY && outStream.ready)

  val partialMaximums = VecInit((0 until maxPoolSize).map((i: Int) => {
    MaxPoolOperation.max(oc)(
      shiftRegs.map(_.bits).reverse(0 + (i * cfg.input.width)),
      shiftRegs.map(_.bits).reverse(1 + (i * cfg.input.width))
    )
  }))

  val shiftRegWrite = risingEdge(shiftRegs.last.valid)
  val (_, totalOutputCounterWrap) = Counter(shiftRegWrite, cfg.output.numParams)
  val (outputBufferCounter, outputBufferCounterWrap) =
    Counter(0 until numBeatsIn, shiftRegWrite, totalOutputCounterWrap)
  when(shiftRegWrite) {
    outputsBuffer(outputBufferCounter) := partialMaximums.reduceTree((in0, in1) => MaxPoolOperation.max(oc)(in0, in1))
  }
  outStream.bits := outputsBuffer.asTypeOf(outStream.bits)
  outStream.valid := RegNext(outputBufferCounterWrap || totalOutputCounterWrap)
  outStream.last := RegNext(totalOutputCounterWrap)

  // VERIFICATION
  when(inStream.fire && !totalInputElementsWrap) {
    assert(inputBufferCounter === 0.U)
  }
  when(transactionsCounterWrap) {
    assert(inStream.last)
  }
}

object MaxPool2D {
  def apply(accel: Accelerator)(implicit p: Parameters): Module with HasAXIStream = {
    require(accel.name == "MaxPool2D")
    require(accel.layers.length == 1)
    accel.layers.head.get match {
      case l: MaxPool2DConfig => new MaxPool2D(OrderCompute(l))
      case _ => throw new RuntimeException
    }
  }
}

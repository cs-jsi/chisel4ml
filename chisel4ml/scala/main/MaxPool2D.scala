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

import lbir.MaxPool2DConfig
import org.slf4j.LoggerFactory
import services.LayerOptions
import chisel3._
import chisel3.util._
import chisel4ml.LBIRStream
import chisel4ml.implicits._
import interfaces.amba.axis._
import chisel4ml.util.risingEdge

/* MaxPool2D
 *
 * A buffer holds a line maxPoolSize rows of a single channel of the input tensor.
 * A series of combinational max selectors then selects the max element at appropriate
 * moments.
 *
 */
class MaxPool2D[T <: Bits with Num[T]](layer: MaxPool2DConfig, options: LayerOptions) extends Module with LBIRStream {
  val logger = LoggerFactory.getLogger(this.getClass())
  val inStream = IO(Flipped(AXIStream(UInt(options.busWidthIn.W))))
  val outStream = IO(AXIStream(UInt(options.busWidthOut.W)))

  val maxPoolSize: Int = layer.input.width / layer.output.width
  require(layer.output.width * maxPoolSize == layer.input.width)
  require(layer.output.height * maxPoolSize == layer.input.height, f"${layer.input} / ${layer.output} == $maxPoolSize")

  val paramsPerTransaction: Int = options.busWidthIn / layer.input.dtype.bitwidth
  val shiftRegsSize:        Int = layer.input.width * maxPoolSize - (layer.input.width - maxPoolSize)

  object InputBufferState extends ChiselEnum {
    val sEMPTY = Value(0.U)
    val sREAD_WORD = Value(1.U)
    val sSTALL = Value(2.U)
  }
  val state = RegInit(InputBufferState.sEMPTY)

  logger.info(s""" MaxPool2D parameters are: maxPoolSize -> $maxPoolSize, paramsPerTransaction -> $paramsPerTransaction
                 | shiftRegsSize -> $shiftRegsSize, input width -> ${layer.input.width}, input height ->
                 | ${layer.input.height}, output width -> ${layer.output.width}, output height ->
                 | ${layer.output.height}.""".stripMargin.replaceAll("\n", ""))

  val inputs = Wire(Vec(paramsPerTransaction, layer.input.getType[T]))
  val validBits = paramsPerTransaction * layer.input.dtype.bitwidth
  inputs := inStream.bits(validBits - 1, 0).asTypeOf(inputs)
  val inputsBuffer = RegEnable(inputs, inStream.fire)
  val outputsBuffer = Reg(Vec(paramsPerTransaction, layer.input.getType[T]))

  val (_, channelElementsCounterWrap) =
    Counter(state === InputBufferState.sREAD_WORD, layer.input.numActiveParams(depthwise = true))
  val (widthCounter, widthCounterWrap) =
    Counter(0 until layer.input.width, state === InputBufferState.sREAD_WORD, channelElementsCounterWrap)
  val (heightCounter, heightCounterWrap) =
    Counter(0 until layer.input.height, widthCounterWrap, channelElementsCounterWrap)
  val (totalInputElements, totalInputElementsWrap) =
    Counter(state === InputBufferState.sREAD_WORD, layer.input.numParams)
  val (transactionsCounter, transactionsCounterWrap) =
    Counter(0 until layer.input.numTransactions(options.busWidthIn), inStream.fire)
  val (inputBufferCounter, inputBufferCounterWrap) =
    Counter(0 until paramsPerTransaction, state === InputBufferState.sREAD_WORD, totalInputElementsWrap)
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

  val inputAndValid = Wire(Valid(layer.input.getType[T]))
  inputAndValid.bits := inputsBuffer(inputBufferCounter)
  inputAndValid.valid := startOfMaxPoolWindow
  val shiftRegs = ShiftRegisters(inputAndValid, shiftRegsSize, state =/= InputBufferState.sEMPTY && outStream.ready)

  val partialMaximums = VecInit((0 until maxPoolSize).map((i: Int) => {
    MaxSelect(
      shiftRegs.map(_.bits).reverse(0 + (i * layer.input.width)),
      shiftRegs.map(_.bits).reverse(1 + (i * layer.input.width)),
      layer.input.getType[T]
    )
  }))

  val shiftRegWrite = risingEdge(shiftRegs.last.valid)
  val (_, totalOutputCounterWrap) = Counter(shiftRegWrite, layer.output.numParams)
  val (outputBufferCounter, outputBufferCounterWrap) = Counter(shiftRegWrite, paramsPerTransaction)
  when(shiftRegWrite) {
    outputsBuffer(outputBufferCounter) := partialMaximums.reduceTree((in0: T, in1: T) =>
      MaxSelect(in0, in1, layer.input.getType[T])
    )
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

class MaxSelect[T <: Bits with Num[T]](genT: T) extends Module {
  val in0 = IO(Input(genT))
  val in1 = IO(Input(genT))
  val out = IO(Output(genT))

  when(in0 > in1) {
    out := in0
  }.otherwise {
    out := in1
  }
}

object MaxSelect {
  def apply[T <: Bits with Num[T]](in0: T, in1: T, genT: T): T = {
    val m = Module(new MaxSelect[T](genT))
    m.in0 := in0
    m.in1 := in1
    m.out
  }
}

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

import interfaces.amba.axis._
import chisel4ml.LBIRStream
import chisel4ml.implicits._
import _root_.lbir.MaxPool2DConfig
import _root_.services.LayerOptions
import chisel3._
import chisel3.util._
import chisel4ml.LBIRStream

import _root_.org.slf4j.Logger
import _root_.org.slf4j.LoggerFactory

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
  require(layer.input.width % maxPoolSize == 0)
  require(layer.input.height % maxPoolSize == 0)
  val paramsPerWord: Int = options.busWidthIn / layer.input.dtype.bitwidth
  val shiftRegsSize: Int = layer.input.width * maxPoolSize
  val genT:          T = UInt(layer.input.dtype.bitwidth.W).asInstanceOf[T]

  logger.info(s""" MaxPool2D parameters are: maxPoolSize -> $maxPoolSize, paramsPerWord -> $paramsPerWord
                 | shiftRegsSize -> $shiftRegsSize, input width -> ${layer.input.width}, input height ->
                 | ${layer.input.height}, output width -> ${layer.output.width}, output height ->
                 | ${layer.output.height}.""".stripMargin.replaceAll("\n", ""))

  val inputs = Wire(Vec(paramsPerWord, genT))
  inputs := inStream.bits.asTypeOf(inputs)
  val inputsBuffer = RegEnable(inputs, inStream.fire)
  val outputsBuffer = Reg(Vec(paramsPerWord, genT))

  object mpState extends ChiselEnum {
    val sWAIT = Value(0.U)
    val sREAD_WORD = Value(1.U)
    val sEMPTY_SHIFT_REGS = Value(2.U)
  }
  val state = RegInit(mpState.sWAIT)
  val (totalElemCntValue, totalElemCntWrap) = Counter(state === mpState.sREAD_WORD, layer.input.numParams)
  val (widthCntValue, widthCntWrap) = Counter(state === mpState.sREAD_WORD, layer.input.width)
  // totalElemCntWraps the counter in cases where widthCntValue doesn't reach full words. I.e. input not perfectly
  // divisible by word size.
  val (heightCntValue, heightCntWrap) = Counter(widthCntWrap || totalElemCntWrap, layer.input.height)
  val isFirstOfWindow = ((heightCntValue % maxPoolSize.U === 0.U) &&
    (widthCntValue % maxPoolSize.U === 0.U) &&
    state === mpState.sREAD_WORD)
  val (inputBufferCntValue, inputBufferCntWrap) = Counter(state === mpState.sREAD_WORD, paramsPerWord)

  val shiftRegs = ShiftRegisters(inputsBuffer(inputBufferCntValue), shiftRegsSize, state =/= mpState.sWAIT)
  val shiftValidRegs = ShiftRegisters(isFirstOfWindow, shiftRegsSize, state =/= mpState.sWAIT)

  inStream.ready := state === mpState.sWAIT // outStream.ready
  val partialMaximums = VecInit((0 until maxPoolSize).map((i: Int) => {
    MaxSelect(shiftRegs.reverse(0 + (i * layer.input.width)), shiftRegs.reverse(1 + (i * layer.input.width)), genT)
  }))

  val (_, totalOutCntWrap) = Counter(shiftValidRegs.last && state =/= mpState.sWAIT, layer.output.numParams)
  val (outputCntValue, outputCntWrap) = Counter(shiftValidRegs.last && state =/= mpState.sWAIT, paramsPerWord)
  outputsBuffer(outputCntValue) := partialMaximums.reduceTree((in0: T, in1: T) => MaxSelect(in0, in1, genT))

  when(state === mpState.sWAIT && inStream.fire) {
    state := mpState.sREAD_WORD
  }.elsewhen(state === mpState.sREAD_WORD) {
    when(totalElemCntWrap) {
      state := mpState.sEMPTY_SHIFT_REGS
    }.elsewhen(inputBufferCntWrap) {
      state := mpState.sWAIT
    }
  }.elsewhen(state === mpState.sEMPTY_SHIFT_REGS) {
    when(totalOutCntWrap) {
      state := mpState.sWAIT
    }
  }

  outStream.bits := outputsBuffer.asTypeOf(outStream.bits)
  outStream.valid := outputCntWrap || totalOutCntWrap
  outStream.last := totalOutCntWrap
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

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

import chisel4ml.{ProcessingElementSequential, ProcessingElementSequentialConfigMaxPool}
import _root_.lbir.Layer
import _root_.services.LayerOptions
import chisel3._
import chisel3.util._


/* MaxPool2D
 *
 * A buffer holds a line maxPoolSize rows of a single channel of the input tensor.
 * A series of combinational max selectors then selects the max element at appropriate
 * moments.
 *
 */
class MaxPool2D[T <: Bits with Num[T]](layer: Layer, options: LayerOptions)
extends ProcessingElementSequential(layer, options) {
    val cfg = ProcessingElementSequentialConfigMaxPool(layer)
    val maxPoolSize: Int = cfg.input.width / cfg.result.width
    require(cfg.input.width % maxPoolSize == 0)
    require(cfg.input.height % maxPoolSize == 0)
    val paramsPerWord: Int = options.busWidthIn / cfg.input.paramBitwidth
    val shiftRegsSize: Int = cfg.input.width * maxPoolSize
    val genT: T = UInt(layer.input.get.dtype.get.bitwidth.W).asInstanceOf[T]

    logger.info(s""" MaxPool2D parameters are: maxPoolSize -> $maxPoolSize, paramsPerWord -> $paramsPerWord
                   | shiftRegsSize -> $shiftRegsSize, input width -> ${cfg.input.width}, input height ->
                   | ${cfg.input.height}, output width -> ${cfg.result.width}, output height ->
                   | ${cfg.result.height}.""".stripMargin.replaceAll("\n", ""))

    val inputs = Wire(Vec(paramsPerWord, genT))
    inputs := inStream.bits.asTypeOf(inputs)
    val inputsBuffer = RegEnable(inputs, inStream.fire)
    val outputsBuffer = Reg(Vec(paramsPerWord, genT))

    object mpState extends ChiselEnum {
        val sWAIT = Value(0.U)
        val sREAD_WORD = Value(1.U)
    }
    val state = RegInit(mpState.sWAIT)
    val (widthCntValue, widthCntWrap) = Counter(state === mpState.sREAD_WORD, cfg.input.width)
    val (heightCntValue, heightCntWrap) = Counter(widthCntWrap, cfg.input.height)
    val (channelCntValue, channelCntWrap) = Counter(heightCntWrap, cfg.input.numChannels)
    val isFirstOfWindow = (heightCntValue % maxPoolSize.U === 0.U) && (widthCntValue % maxPoolSize.U === 0.U)
    val (inputBufferCntValue, inputBufferCntWrap) = Counter(state === mpState.sREAD_WORD, paramsPerWord)
    when (state === mpState.sWAIT && inStream.fire) {
        state := mpState.sREAD_WORD
    }.elsewhen(state === mpState.sREAD_WORD && inputBufferCntWrap) {
        state := mpState.sWAIT
    }


    val shiftRegs = ShiftRegisters(inputsBuffer(inputBufferCntValue), shiftRegsSize, state === mpState.sREAD_WORD)
    val shiftValidRegs = ShiftRegisters(isFirstOfWindow, shiftRegsSize, state === mpState.sREAD_WORD)

    inStream.ready := state === mpState.sWAIT  // outStream.ready
    val partialMaximums = VecInit((0 until maxPoolSize).map((i: Int) => {
        MaxSelect(shiftRegs.reverse(0 + (i * cfg.input.width)),
                           shiftRegs.reverse(1 + (i * cfg.input.width)),
                           genT)
    }))


    val (outputCntValue, outputCntWrap) = Counter(shiftValidRegs.last && state === mpState.sREAD_WORD, paramsPerWord)
    outputsBuffer(outputCntValue) := partialMaximums.reduceTree((in0:T, in1:T) => MaxSelect(in0, in1, genT))


    outStream.bits := outputsBuffer.asTypeOf(outStream.bits)
    outStream.valid := outputCntWrap || outStream.last
    outStream.last := RegNext(RegNext(RegNext(RegNext(channelCntWrap)))) // TODO
}

class MaxSelect[T <: Bits with Num[T]](genT: T) extends Module {
    val in0 = IO(Input(genT))
    val in1 = IO(Input(genT))
    val out = IO(Output(genT))

    when (in0 > in1) {
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

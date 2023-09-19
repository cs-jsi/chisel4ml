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
import chisel4ml.MemWordSize
import chisel4ml.implicits._

/** Result Memory Buffer
  */
class ResultMemoryBuffer[O <: Bits](genOut: O, output: lbir.QTensor) extends Module {

  val resultsPerKernel = output.width * output.height
  val resultsPerWord: Int = MemWordSize.bits / genOut.getWidth
  val numKernels =
    output.numChannels // Number of channels in output tensor is the same as number of kernels in the weights

  val io = IO(new Bundle {
    // interface to the dynamic neuron (alu)
    val result = Input(genOut)
    val resultValid = Input(Bool())

    // result memory interface
    val resRamEn = Output(Bool())
    val resRamAddr = Output(UInt(log2Up(output.memDepth).W))
    val resRamData = Output(UInt(MemWordSize.bits.W))

    // control inerface
    val start = Input(Bool())
  })

  object rmbState extends ChiselEnum {
    val sWAIT = Value(0.U)
    val sCOMP = Value(1.U)
  }

  val resPerWordCnt = RegInit(0.U(log2Up(resultsPerWord + 1).W))
  val resPerKernelCnt = RegInit(0.U(log2Up(resultsPerKernel + 1).W))
  val kernelCnt = RegInit(0.U(log2Up(numKernels).W))

  val dataBuf = RegInit(VecInit(Seq.fill(resultsPerWord)(0.U(genOut.getWidth.W))))
  val ramAddr = RegInit(0.U(log2Up(output.memDepth).W))

  val state = RegInit(rmbState.sWAIT)

  state := state
  when(io.start) {
    state := rmbState.sCOMP
  }.elsewhen(state === rmbState.sCOMP) {
    when(
      resPerKernelCnt === resultsPerKernel.U &&
        kernelCnt === (numKernels - 1).U
    ) {
      state := rmbState.sWAIT
    }
  }

  when(io.start) {
    resPerWordCnt := 0.U
    resPerKernelCnt := 0.U
    kernelCnt := 0.U
  }.otherwise {
    when(io.resultValid) {
      resPerKernelCnt := resPerKernelCnt + 1.U
      when(resPerKernelCnt === resultsPerKernel.U) {
        kernelCnt := kernelCnt + 1.U
        resPerKernelCnt := 0.U
        resPerWordCnt := 0.U
      }.elsewhen(resPerWordCnt === (resultsPerWord - 1).U) {
        resPerWordCnt := 0.U
      }.otherwise {
        resPerWordCnt := resPerWordCnt + 1.U
      }
    }
  }

  ramAddr := ramAddr
  when(io.start) {
    ramAddr := 0.U
  }.elsewhen(io.resRamEn) {
    ramAddr := ramAddr + 1.U
  }

  when(io.start) {
    dataBuf := VecInit(Seq.fill(resultsPerWord)(0.U(genOut.getWidth.W)))
  }.elsewhen(io.resultValid) {
    dataBuf(resPerWordCnt) := io.result.asUInt
  }

  io.resRamEn := RegNext(
    (RegNext(state === rmbState.sCOMP) &&
      ((resPerWordCnt === (resultsPerWord - 1).U) ||
        resPerKernelCnt === (resultsPerKernel - 1).U) &&
      io.resultValid)
  )
  io.resRamAddr := ramAddr
  io.resRamData := dataBuf.asUInt
}

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
import lbir.Conv2DConfig

/** PeSeqConvController
  */
class PeSeqConvController(layer: Conv2DConfig) extends Module {
  val io = IO(new Bundle {
    // SWU interface
    val swuEnd = Input(Bool())
    val swuStart = Output(Bool())

    // interface to the Kernel RF/loader
    val krfReady = Input(Bool())
    val krfKernelNum = Output(UInt(log2Up(layer.kernel.numKernels).W))
    val krfLoadKernel = Output(Bool())

    val rmbStart = Output(Bool())

    // input stream
    val inStreamReady = Output(Bool())
    val inStreamValid = Input(Bool())
    val inStreamLast = Input(Bool())
    val actMemAddr = Output(UInt(log2Up(layer.input.memDepth).W))

    // output stream
    val outStreamReady = Input(Bool())
    val outStreamValid = Output(Bool())
    val outStreamLast = Output(Bool())
    val resMemAddr = Output(UInt(log2Up(layer.output.memDepth).W))
    val resMemEna = Output(Bool())
  })

  object ctrlState extends ChiselEnum {
    val sWAITFORDATA = Value(0.U)
    val sLOADINPACT = Value(1.U)
    val sLOADKERNEL = Value(2.U)
    val sCOMP = Value(3.U)
    val sWAITWRITE = Value(4.U)
    val sWAITWRITE2 = Value(5.U)
    val sSENDDATA = Value(6.U)
  }

  val kernelCnt = RegInit(0.U(log2Up(layer.kernel.numKernels + 1).W))
  val actMemCnt = RegInit(0.U(log2Up(layer.input.numTransactions(MemWordSize.bits) + 1).W))
  val resMemCnt = RegInit(0.U(log2Up(layer.output.memDepth + 1).W))

  val state = RegInit(ctrlState.sWAITFORDATA)
  val nstate = WireInit(ctrlState.sCOMP)

  nstate := state
  when(state === ctrlState.sWAITFORDATA && io.inStreamValid) {
    nstate := ctrlState.sLOADINPACT
  }.elsewhen(state === ctrlState.sLOADINPACT && actMemCnt === layer.input.numTransactions(MemWordSize.bits).U) {
    nstate := ctrlState.sLOADKERNEL
  }.elsewhen(state === ctrlState.sLOADKERNEL && io.krfReady) {
    nstate := ctrlState.sCOMP
  }.elsewhen(state === ctrlState.sCOMP && io.swuEnd) {
    when(kernelCnt === layer.kernel.numKernels.U) {
      nstate := ctrlState.sWAITWRITE
    }.otherwise {
      nstate := ctrlState.sLOADKERNEL
    }
  }.elsewhen(state === ctrlState.sWAITWRITE) { // we wait two cycle for results to be written to resMem
    nstate := ctrlState.sWAITWRITE2
  }.elsewhen(state === ctrlState.sWAITWRITE2) {
    nstate := ctrlState.sSENDDATA
  }.elsewhen(state === ctrlState.sSENDDATA && resMemCnt === layer.output.memDepth.U) {
    nstate := ctrlState.sWAITFORDATA
  }
  state := nstate

  when(state === ctrlState.sWAITFORDATA) {
    kernelCnt := 0.U
  }.elsewhen(io.krfReady) {
    kernelCnt := kernelCnt + 1.U
  }

  actMemCnt := actMemCnt
  when(state === ctrlState.sWAITWRITE) {
    actMemCnt := 0.U
  }.elsewhen(io.inStreamReady && io.inStreamValid) {
    actMemCnt := actMemCnt + 1.U
  }

  resMemCnt := resMemCnt
  when(state === ctrlState.sLOADINPACT) {
    resMemCnt := 0.U
  }.elsewhen(
    (state === ctrlState.sSENDDATA ||
      nstate === ctrlState.sSENDDATA) && io.outStreamReady && io.outStreamValid
  ) {
    resMemCnt := resMemCnt + 1.U
  }

  io.swuStart := (((state === ctrlState.sLOADKERNEL) && io.krfReady) ||
    (state === ctrlState.sCOMP) && RegNext(RegNext(io.swuEnd)))

  io.krfKernelNum := kernelCnt
  io.krfLoadKernel := RegNext(
    (RegNext(io.swuEnd) ||
      ((state === ctrlState.sLOADKERNEL) &&
        RegNext(state === ctrlState.sLOADINPACT)))
  )

  io.inStreamReady := state === ctrlState.sLOADINPACT || state === ctrlState.sWAITFORDATA
  io.actMemAddr := actMemCnt

  io.outStreamValid := (state === ctrlState.sSENDDATA)
  io.outStreamLast := ((resMemCnt === layer.output.memDepth.U) && (state === ctrlState.sSENDDATA))
  io.resMemAddr := resMemCnt
  io.resMemEna := (state === ctrlState.sSENDDATA)

  io.rmbStart := (state === ctrlState.sWAITFORDATA) && (nstate === ctrlState.sLOADINPACT)
}

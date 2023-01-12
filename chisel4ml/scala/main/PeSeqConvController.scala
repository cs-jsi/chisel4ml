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

import _root_.chisel4ml.util.reqWidth
import chisel3._
import chisel3.experimental.ChiselEnum
import chisel3.util._

/** PeSeqConvController
  *
  */
class PeSeqConvController(numKernels: Int,
                          resMemDepth: Int,
                          actMemDepth: Int) extends Module {

  val io = IO(new Bundle {
    // SWU interface
    val swuEnd        = Input(Bool())
    val swuStart      = Output(Bool())

    // interface to the Kernel RF/loader
    val krfReady      = Input(Bool())
    val krfKernelNum  = Output(UInt(reqWidth(numKernels).W))
    val krfLoadKernel = Output(Bool())

    // input stream
    val inStreamReady = Output(Bool())
    val inStreamValid = Input(Bool())
    val inStreamLast  = Input(Bool())
    val actMemAddr    = Output(UInt(actMemDepth.W))

    // output stream
    val outStreamReady = Input(Bool())
    val outStreamValid = Output(Bool())
    val outStreamLast  = Output(Bool())
    val resMemAddr     = Output(UInt(reqWidth(resMemDepth).W))
    val resMemEna      = Output(Bool())
  })

  object ctrlState extends ChiselEnum {
    val sWAITFORDATA = Value(0.U)
    val sLOADKERNEL  = Value(1.U)
    val sCOMP        = Value(2.U)
    val sSENDDATA    = Value(3.U)
  }

  val kernelCnt = RegInit(0.U(reqWidth(numKernels).W))
  val actMemCnt = RegInit(0.U(reqWidth(actMemDepth).W))
  val resMemCnt = RegInit(0.U(reqWidth(resMemDepth).W))

  val state  = RegInit(ctrlState.sWAITFORDATA)
  val nstate = WireInit(ctrlState.sCOMP)


  nstate := state
  when (state === ctrlState.sWAITFORDATA && io.inStreamValid) {
    nstate := ctrlState.sLOADKERNEL
  }.elsewhen (state === ctrlState.sLOADKERNEL && actMemCnt === (actMemDepth - 1).U) {
    nstate := ctrlState.sCOMP
  }.elsewhen (state === ctrlState.sCOMP && kernelCnt === (numKernels - 1).U && io.swuEnd) {
    nstate := ctrlState.sSENDDATA
  }.elsewhen (state === ctrlState.sSENDDATA && resMemCnt === (resMemDepth - 1).U) {
    nstate := ctrlState.sWAITFORDATA
  }
  state := nstate


  when (state === ctrlState.sLOADKERNEL) {
    kernelCnt := 0.U
  }.elsewhen (io.swuEnd) {
    kernelCnt := kernelCnt + 1.U
  }

  actMemCnt := actMemCnt
  when (state === ctrlState.sCOMP && kernelCnt === (numKernels - 1).U && io.swuEnd) {
    actMemCnt := 0.U
  }.elsewhen (io.inStreamReady && io.inStreamValid) {
    actMemCnt := actMemCnt + 1.U
  }

  resMemCnt := resMemCnt
  when (state === ctrlState.sCOMP) {
    resMemCnt := 0.U
  }.elsewhen (io.outStreamReady && io.outStreamValid) {
    resMemCnt := resMemCnt + 1.U
  }

  io.swuStart := (state === ctrlState.sCOMP) && RegNext(state === ctrlState.sLOADKERNEL)

  io.krfKernelNum  := kernelCnt
  io.krfLoadKernel := (state === ctrlState.sLOADKERNEL)

  io.inStreamReady := actMemCnt =/= (actMemDepth - 1).U
  io.actMemAddr    := actMemCnt

  io.outStreamValid := (resMemCnt =/= (resMemDepth - 1).U) && (state === ctrlState.sSENDDATA)
  io.outStreamLast  := (resMemCnt === (resMemDepth - 1).U) && (state === ctrlState.sSENDDATA)
  io.resMemAddr     := resMemCnt
  io.resMemEna      := (state === ctrlState.sSENDDATA)
}

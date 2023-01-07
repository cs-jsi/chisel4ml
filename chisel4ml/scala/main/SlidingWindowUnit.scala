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

/** Sliding Window Unit
  *
  * kernelSize - Size of one side of a square 2d kernel (i.e. kernelSize=3 for a 3x3 kernel). kernelDepth - The depth of
  * a kernel, i.e. for a RGB image, the kernel depth is 3. (i.e. num of channels) actParamSize - The bitwidth of each
  * activation parameter.
  */
class SlidingWindowUnit(
    kernelSize:   Int,
    kernelDepth:  Int,
    actWidth:     Int,
    actHeight:    Int,
    actParamSize: Int,
  ) extends Module {

  val totalNumOfKernelElements: Int = kernelSize * kernelSize * kernelDepth
  val wrDataWidth:              Int = kernelSize * actParamSize
  val chAddrWidth:              Int = reqWidth(kernelDepth.toFloat)
  val rowAddrWidth:             Int = reqWidth(kernelSize.toFloat)

  val memWordWidth: Int = 32

  val actParamsPerWord: Int = memWordWidth / actParamSize
  val actMemValidBits:  Int = actParamsPerWord * actParamSize
  val leftoverBits:     Int = memWordWidth - actMemValidBits

  val actMemChSize:  Int = kernelSize * kernelSize * actParamSize
  val actMemRowSize: Int = kernelSize * actParamSize
  val actMemColSize: Int = kernelSize

  val actMemWordsForKernel: Int = math.ceil(totalNumOfKernelElements.toFloat / actParamsPerWord.toFloat).toInt
  val actMemDepthBits:      Int = actWidth * actHeight * kernelDepth * actParamSize
  val actMemDepthWords:     Int = (actMemDepthBits / actMemValidBits) + 1

  val colAddConstant:   Int = actParamSize
  val rowAddConstant:   Int = ((actWidth - kernelSize) + (kernelSize - 1)) * actParamSize
  val chAddConstant:    Int = ((actWidth - kernelSize) + ((actHeight - kernelSize) * actWidth) + 1) * actParamSize
  val constantWireSize: Int = if (reqWidth(chAddConstant) > 5) reqWidth(chAddConstant) else 5

  val io = IO(new Bundle {
    // interface to the RollingRegisterFile module.
    val shiftRegs    = Output(Bool())
    val rowWriteMode = Output(Bool())
    val rowAddr      = Output(UInt(rowAddrWidth.W))
    val chAddr       = Output(UInt(chAddrWidth.W))
    val data         = Output(UInt(wrDataWidth.W))
    val valid        = Output(Bool())

    // interface to the activation memory
    val actRdEn   = Output(Bool())
    val actRdAddr = Output(UInt(reqWidth(actMemDepthWords).W))
    val actRdData = Input(UInt(memWordWidth.W))

    // control interface
    val start = Input(Bool())
  })

  val bitAddr        = RegInit(0.U(reqWidth(actMemDepthBits).W)) // bit addressing, because params can be any width
  val bitAddrNext    = Wire(UInt(reqWidth(actMemDepthBits).W))
  val bitAddrNextMod = Wire(UInt(reqWidth(actMemDepthBits).W))
  val addConstant    = WireInit(0.U(constantWireSize.W))
  val correction     = Wire(UInt())
  val subwordAddr    = Wire(UInt(reqWidth(memWordWidth).W))      // Which part of the memory word are we looking at?
  val ramDataAsVec   = Wire(Vec(actParamsPerWord, UInt(actParamSize.W)))
  val dataBuf        = RegInit(VecInit(Seq.fill(kernelSize)(0.U(actParamSize.W))))
  val dataIndex      = Wire(UInt())

  object swuState extends ChiselEnum {
    // val sWAIT, sADDCOL, sADDROW, sADDCH, sEND = Value
    val sWAITSTART = Value(0.U)
    val sADDCOL    = Value(1.U)
    val sADDROW    = Value(2.U)
    val sADDCH     = Value(3.U)
    val sSTALL     = Value(4.U)
    val sEND       = Value(5.U)
    val sERROR     = Value(6.U)
  }

  val nstate     = WireInit(swuState.sERROR)
  val nstateTemp = WireInit(swuState.sERROR)
  val state      = RegNext(next = nstate, init = swuState.sWAITSTART)
  val stallState = RegInit(swuState.sERROR)

  val colCnt = RegInit(0.U(reqWidth(kernelSize).W))
  val rowCnt = RegInit(0.U(reqWidth(kernelSize).W))
  val chCnt  = RegInit(0.U(reqWidth(kernelDepth).W))

  // //// NEXT STATE LOGIC //////
  when(state === swuState.sWAITSTART) {
    when(io.start) {
      nstate     := swuState.sSTALL
      stallState := swuState.sADDCOL
    }.otherwise {
      nstate := swuState.sWAITSTART
    }
  }.elsewhen(state === swuState.sSTALL) {
    nstate := stallState
  }.elsewhen(
    state === swuState.sADDCOL ||
      state === swuState.sADDROW ||
      state === swuState.sADDCH,
  ) {
    when(
      colCnt === (kernelSize - 1).U &&
        rowCnt === (kernelSize - 1).U &&
        chCnt === (kernelDepth - 1).U,
    ) {
      nstateTemp := swuState.sEND
    }.elsewhen(
      colCnt === (kernelSize - 2).U &&
        rowCnt === (kernelSize - 1).U,
    ) {
      nstateTemp := swuState.sADDCH
    }.elsewhen(colCnt === (kernelSize - 2).U) {
      nstateTemp := swuState.sADDROW
    }.otherwise {
      nstateTemp := swuState.sADDCOL
    }
    when(correction =/= 0.U) {
      nstate     := swuState.sSTALL
      stallState := nstateTemp
    }.otherwise {
      nstate     := nstateTemp
      stallState := swuState.sERROR
    }
  }.elsewhen(state === swuState.sEND) {
    nstate := swuState.sWAITSTART
  }.elsewhen(state === swuState.sERROR) {
    nstate := swuState.sERROR
  }.otherwise {
    nstate := swuState.sERROR
  }

  // //// CONSTANTS //////
  addConstant := 0.U
  switch(state) {
    is(swuState.sADDCOL)(addConstant := colAddConstant.U)
    is(swuState.sADDROW)(addConstant := rowAddConstant.U)
    is(swuState.sADDCH)(addConstant  := chAddConstant.U)
  }

  // //// COUNTERS //////
  switch(state) {
    is(swuState.sWAITSTART) { colCnt := 0.U; rowCnt := 0.U; chCnt := 0.U }
    is(swuState.sADDCOL)(colCnt := colCnt + 1.U)
    is(swuState.sADDROW) { rowCnt := rowCnt + 1.U; colCnt := 0.U }
    is(swuState.sADDCH) { chCnt := chCnt + 1.U; rowCnt := 0.U; colCnt := 0.U }
  }

  // //// ADDRESS CALCULATIONS //////
  bitAddrNext := bitAddr + addConstant
  // The correction factor corrects for the stuffing that happens (if 32 / bw is not whole) at the end of each word
  correction     := (addConstant +& bitAddr(4, 0) + leftoverBits.U) >> 5
  bitAddrNextMod := bitAddrNext + correction * leftoverBits.U
  when(
    state === swuState.sADDCOL ||
      state === swuState.sADDROW ||
      state === swuState.sADDCH,
  ) {
    bitAddr := bitAddrNextMod
  }.elsewhen(state === swuState.sWAITSTART) {}

  // //// ACTIVATION MEMORY INTERFACE //////
  io.actRdEn := (state === swuState.sSTALL ||
    state === swuState.sADDCOL ||
    state === swuState.sADDROW ||
    state === swuState.sADDCH)
  io.actRdAddr := (bitAddr >> 5)

  // Subword (bit) address is translated into indexes via a lookup table
  subwordAddr := bitAddr(4, 0)
  dataIndex := MuxLookup(
    subwordAddr,
    0.U,
    Seq.tabulate(actParamsPerWord)(_ * actParamSize).zipWithIndex.map(x => (x._1.U -> x._2.U)),
  )
  ramDataAsVec := io.actRdData(actMemValidBits - 1, 0).asTypeOf(ramDataAsVec)
  when(
    state === swuState.sADDCOL ||
      state === swuState.sADDROW ||
      state === swuState.sADDCH,
  ) {
    dataBuf(colCnt) := ramDataAsVec(dataIndex)
  }

  // //// ROLLING REGISTER FILE INTERFACE //////
  io.shiftRegs    := false.B
  io.rowWriteMode := true.B
  io.rowAddr      := RegNext(rowCnt)
  io.chAddr       := RegNext(chCnt)
  io.data         := dataBuf.asUInt
  when(io.rowWriteMode) {
    io.valid := RegNext(colCnt === (kernelSize - 1).U)
  }.otherwise {
    io.valid := false.B
  }
}

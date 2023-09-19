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

/** KernelRFLoader
  */
class KernelRFLoader(kernel: lbir.QTensor) extends Module {
  val wordsPerKernel:     Int = math.ceil(kernel.numKernelParams.toFloat / kernel.paramsPerWord.toFloat).toInt
  val kernelMemValidBits: Int = kernel.paramsPerWord * kernel.dtype.bitwidth

  val io = IO(new Bundle {
    // interface to the kernel register file
    val chAddr = Output(UInt(log2Up(kernel.numChannels).W))
    val rowAddr = Output(UInt(log2Up(kernel.width).W))
    val colAddr = Output(UInt(log2Up(kernel.width).W))
    val data = Output(UInt(kernel.dtype.bitwidth.W))
    val valid = Output(Bool())

    // interface to the kernel ROM
    val romRdEna = Output(Bool())
    val romRdAddr = Output(UInt(log2Up(kernel.memDepth + 1).W))
    val romRdData = Input(UInt(MemWordSize.bits.W))

    // control interface
    val kernelReady = Output(Bool())
    val loadKernel = Input(Bool())
    val kernelNum = Input(UInt(log2Up(kernel.numKernels).W))
  })

  object krfState extends ChiselEnum {
    val sWAIT = Value(0.U)
    val sFILLRF = Value(1.U)
    val sEND = Value(2.U)
  }

  val nstate = WireInit(krfState.sEND)
  val state = RegInit(krfState.sWAIT)
  val stall = RegInit(false.B)

  val dataBuf = RegInit(0.U(kernel.dtype.bitwidth.W))

  val kernelOffset = RegInit(0.U(log2Up(kernel.memDepth).W))
  val colCnt = RegInit(0.U(log2Up(kernel.height).W))
  val rowCnt = RegInit(0.U(log2Up(kernel.width).W))
  val chCnt = RegInit(0.U(log2Up(kernel.numChannels).W))

  val elemBitOffset = WireInit(0.U(log2Up(MemWordSize.bits).W))
  val ramAddr = RegInit(0.U(log2Up(kernel.memDepth + 1).W))
  val nramAddr = WireInit(0.U(log2Up(kernel.memDepth + 1).W))
  val wordElemCnt = RegInit(0.U(log2Up(kernel.paramsPerWord).W))
  val nwordElemCnt = WireInit(0.U(log2Up(kernel.paramsPerWord).W))
  val totalElemCnt = RegInit(0.U(log2Up(kernel.numKernelParams).W))
  val ntotalElemCnt = WireInit(0.U(log2Up(kernel.numKernelParams).W))
  val romDataAsVec = Wire(Vec(kernel.paramsPerWord, UInt(kernel.dtype.bitwidth.W)))

  ///////////////////////
  // NEXT STATE LOGIC  //
  ///////////////////////
  nstate := state
  when(state === krfState.sWAIT && io.loadKernel) {
    nstate := krfState.sFILLRF
  }.elsewhen(state === krfState.sFILLRF && totalElemCnt === (kernel.numKernelParams - 1).U) {
    nstate := krfState.sEND
  }.elsewhen(state === krfState.sEND) {
    nstate := krfState.sWAIT
  }
  when(!stall) {
    state := nstate
  }

  ////////////////////////
  // RAM ADDRESS LOGIC  //
  ////////////////////////
  nramAddr := ramAddr
  nwordElemCnt := wordElemCnt
  ntotalElemCnt := totalElemCnt
  when(state === krfState.sWAIT && io.loadKernel) {
    // we map the index to the offset with a static lookup table
    nramAddr := MuxLookup(
      io.kernelNum,
      0.U,
      Seq.tabulate(kernel.numKernels)(_ * wordsPerKernel).zipWithIndex.map(x => (x._2.U -> x._1.U))
    )
    nwordElemCnt := 0.U
    ntotalElemCnt := 0.U
  }.elsewhen(state === krfState.sFILLRF) {
    when(wordElemCnt === (kernel.paramsPerWord - 1).U) {
      nramAddr := ramAddr + 1.U
      nwordElemCnt := 0.U
      ntotalElemCnt := totalElemCnt + 1.U
    }.otherwise {
      nwordElemCnt := wordElemCnt + 1.U
      ntotalElemCnt := totalElemCnt + 1.U
    }
  }
  when(!stall) {
    ramAddr := nramAddr
    wordElemCnt := nwordElemCnt
    totalElemCnt := ntotalElemCnt
  }

  //////////////////////////////
  // HANDLE KERNEL ROM INPUT  //
  //////////////////////////////
  romDataAsVec := io.romRdData(kernelMemValidBits - 1, 0).asTypeOf(romDataAsVec)
  when(state === krfState.sFILLRF && !stall) {
    dataBuf := romDataAsVec(wordElemCnt)
  }

  /////////////////////
  // COUNTERS LOGIC  //
  /////////////////////
  when(state === krfState.sWAIT && io.loadKernel) {
    colCnt := 0.U
    rowCnt := 0.U
    chCnt := 0.U
  }.elsewhen(!stall) {
    when(
      colCnt === (kernel.height - 1).U &&
        rowCnt === (kernel.width - 1).U
    ) {
      chCnt := chCnt + 1.U
      rowCnt := 0.U
      colCnt := 0.U
    }.elsewhen(colCnt === (kernel.height - 1).U) {
      rowCnt := rowCnt + 1.U
      colCnt := 0.U
    }.otherwise {
      colCnt := colCnt + 1.U
    }
  }

  ///// STALL LOGIC ////
  stall := (ramAddr =/= nramAddr) && !stall

  ///////////////////////
  // MODULE INTERFACES //
  ///////////////////////

  // kernel RF interface
  io.chAddr := RegNext(chCnt)
  io.rowAddr := RegNext(rowCnt)
  io.colAddr := RegNext(colCnt)
  io.data := dataBuf
  io.valid := RegNext((state === krfState.sFILLRF) && !stall)

  // kernel ROM interface
  io.romRdEna := (state === krfState.sFILLRF)
  io.romRdAddr := ramAddr

  // control interface
  io.kernelReady := (state === krfState.sEND) && !stall
}

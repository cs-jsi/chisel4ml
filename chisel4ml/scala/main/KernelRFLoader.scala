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

/** KernelRFLoader
  *
  * kernelSize - Size of one side of a square 2d kernel (i.e. kernelSize=3 for a 3x3 kernel).
  * kernelDepth - The depth of a kernel/actMap, i.e. for a RGB image, the kernel depth is 3. (i.e. num of channels)
  * kernelParamSize - size of each kernel parameter
  * numKernels - The number of different kernels
  */
class KernelRFLoader(
    kernelSize: Int,
    kernelDepth: Int,
    kernelParamSize: Int,
    numKernels: Int
  ) extends Module {

  val memWordWidth:                Int = 32
  val kernelParamsPerWord:         Int = memWordWidth / kernelParamSize
  val totalNumOfElementsPerKernel: Int = kernelSize * kernelSize * kernelDepth
  val wordsPerKernel:              Int = math.ceil(totalNumOfElementsPerKernel.toFloat / kernelParamsPerWord.toFloat).toInt
  val kernelMemDepthWords:         Int = wordsPerKernel * numKernels
  val kernelMemValidBits:          Int = kernelParamsPerWord * kernelParamSize
  val kernelNumOfElements:         Int = kernelSize * kernelSize * kernelDepth

  val io = IO(new Bundle {
    // interface to the kernel register file
    val chAddr  = Output(UInt(reqWidth(kernelDepth).W))
    val rowAddr = Output(UInt(reqWidth(kernelSize).W))
    val colAddr = Output(UInt(reqWidth(kernelSize).W))
    val data    = Output(UInt(kernelParamSize.W))
    val valid   = Output(Bool())

    // interface to the kernel ROM
    val romRdEna  = Output(Bool())
    val romRdAddr = Output(UInt(reqWidth(kernelMemDepthWords+1).W))
    val romRdData = Input(UInt(memWordWidth.W))

    // control interface
    val kernelReady = Output(Bool())
    val loadKernel  = Input(Bool())
    val kernelNum   = Input(UInt(reqWidth(numKernels).W))
  })

  object krfState extends ChiselEnum {
    val sWAIT   = Value(0.U)
    val sFILLRF = Value(1.U)
    val sEND    = Value(2.U)
  }

  val nstate = WireInit(krfState.sEND)
  val state  = RegInit(krfState.sWAIT)
  val stall  = RegInit(false.B)

  val dataBuf = RegInit(0.U(kernelParamSize.W))

  val kernelOffset = RegInit(0.U(reqWidth(kernelMemDepthWords).W))
  val colCnt = RegInit(0.U(reqWidth(kernelSize).W))
  val rowCnt = RegInit(0.U(reqWidth(kernelSize).W))
  val chCnt  = RegInit(0.U(reqWidth(kernelDepth).W))

  val elemBitOffset = WireInit(0.U(reqWidth(memWordWidth).W))
  val ramAddr       = RegInit(0.U(reqWidth(kernelMemDepthWords+1).W))
  val nramAddr      = WireInit(0.U(reqWidth(kernelMemDepthWords+1).W))
  val wordElemCnt   = RegInit(0.U(reqWidth(kernelParamsPerWord).W))
  val nwordElemCnt  = WireInit(0.U(reqWidth(kernelParamsPerWord).W))
  val totalElemCnt  = RegInit(0.U(reqWidth(kernelNumOfElements).W))
  val ntotalElemCnt = WireInit(0.U(reqWidth(kernelNumOfElements).W))
  val romDataAsVec  = Wire(Vec(kernelParamsPerWord, UInt(kernelParamSize.W)))

  ///////////////////////
  // NEXT STATE LOGIC  //
  ///////////////////////
  nstate := state
  when (state === krfState.sWAIT && io.loadKernel) {
    nstate := krfState.sFILLRF
  }.elsewhen(state === krfState.sFILLRF && totalElemCnt === (kernelNumOfElements - 1).U) {
    nstate := krfState.sEND
  }.elsewhen(state === krfState.sEND) {
    nstate := krfState.sWAIT
  }
  when (!stall) {
    state := nstate
  }


  ////////////////////////
  // RAM ADDRESS LOGIC  //
  ////////////////////////
  nramAddr := ramAddr
  nwordElemCnt := wordElemCnt
  ntotalElemCnt := totalElemCnt
  when (state === krfState.sWAIT && io.loadKernel) {
    // we map the index to the offset with a static lookup table
    nramAddr := MuxLookup(io.kernelNum,
                          0.U,
                          Seq.tabulate(numKernels)(_ * wordsPerKernel).zipWithIndex.map(x => (x._2.U -> x._1.U)))
    nwordElemCnt := 0.U
    ntotalElemCnt := 0.U
  }.elsewhen (state === krfState.sFILLRF) {
    when(wordElemCnt === (kernelParamsPerWord - 1).U) {
      nramAddr := ramAddr + 1.U
      nwordElemCnt := 0.U
      ntotalElemCnt := totalElemCnt + 1.U
    }.otherwise {
      nwordElemCnt := wordElemCnt + 1.U
      ntotalElemCnt := totalElemCnt + 1.U
    }
  }
  when (!stall) {
    ramAddr := nramAddr
    wordElemCnt := nwordElemCnt
    totalElemCnt := ntotalElemCnt
  }


  //////////////////////////////
  // HANDLE KERNEL ROM INPUT  //
  //////////////////////////////
  romDataAsVec := io.romRdData(kernelMemValidBits - 1, 0).asTypeOf(romDataAsVec)
  when (state === krfState.sFILLRF && ! stall) {
    dataBuf := romDataAsVec(wordElemCnt)
  }

  /////////////////////
  // COUNTERS LOGIC  //
  /////////////////////
  when (state === krfState.sWAIT && io.loadKernel) {
    colCnt := 0.U
    rowCnt := 0.U
    chCnt := 0.U
  }.elsewhen(!stall) {
    when (colCnt === (kernelSize - 1).U &&
          rowCnt === (kernelSize - 1).U) {
      chCnt  := chCnt + 1.U
      rowCnt := 0.U
      colCnt := 0.U
    }.elsewhen(colCnt === (kernelSize - 1).U) {
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
  io.chAddr  := RegNext(chCnt)
  io.rowAddr := RegNext(rowCnt)
  io.colAddr := RegNext(colCnt)
  io.data    := dataBuf
  io.valid   := RegNext((state === krfState.sFILLRF) && !stall)

  // kernel ROM interface
  io.romRdEna  := (state === krfState.sFILLRF)
  io.romRdAddr := ramAddr

  // control interface
  io.kernelReady := (state === krfState.sEND) && !stall
}

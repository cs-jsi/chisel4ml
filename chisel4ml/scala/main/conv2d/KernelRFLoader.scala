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
package chisel4ml.conv2d

import chisel3._
import chisel3.util._
import chisel4ml.MemWordSize
import chisel4ml.implicits._
import memories.SRAMRead

/** KernelRFLoader
  */
class KernelRFLoader(kernel: lbir.QTensor) extends Module {
  val wordsPerKernel:     Int = math.ceil(kernel.numKernelParams.toFloat / kernel.paramsPerWord.toFloat).toInt
  val kernelMemValidBits: Int = kernel.paramsPerWord * kernel.dtype.bitwidth

  val io = IO(new Bundle {
    val krf = Valid(new KernelRegisterFileInput(kernel))
    val kernelLoaded = Output(Bool())
    val rom = Flipped(new SRAMRead(depth = kernel.memDepth, width = MemWordSize.bits))
    val loadKernel = Flipped(Valid(UInt(log2Up(kernel.numKernels).W)))
  })

  object krlState extends ChiselEnum {
    val sWAIT = Value(0.U)
    val sFILLRF = Value(1.U)
    val sEND = Value(2.U)
  }

  val nstate = WireInit(krlState.sEND)
  val state = RegInit(krlState.sWAIT)
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
  when(state === krlState.sWAIT && io.loadKernel.valid) {
    nstate := krlState.sFILLRF
  }.elsewhen(state === krlState.sFILLRF && totalElemCnt === (kernel.numKernelParams - 1).U) {
    nstate := krlState.sEND
  }.elsewhen(state === krlState.sEND) {
    nstate := krlState.sWAIT
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
  when(state === krlState.sWAIT && io.loadKernel.valid) {
    // we map the index to the offset with a static lookup table
    nramAddr := MuxLookup(
      io.loadKernel.bits,
      0.U,
      Seq.tabulate(kernel.numKernels)(_ * wordsPerKernel).zipWithIndex.map(x => (x._2.U -> x._1.U))
    )
    nwordElemCnt := 0.U
    ntotalElemCnt := 0.U
  }.elsewhen(state === krlState.sFILLRF) {
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
  romDataAsVec := io.rom.data(kernelMemValidBits - 1, 0).asTypeOf(romDataAsVec)
  when(state === krlState.sFILLRF && !stall) {
    dataBuf := romDataAsVec(wordElemCnt)
  }

  /////////////////////
  // COUNTERS LOGIC  //
  /////////////////////
  when(state === krlState.sWAIT && io.loadKernel.valid) {
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
  io.kernelLoaded := state === krlState.sEND

  // kernel RF interface
  io.krf.bits.channelAddress := RegNext(chCnt)
  io.krf.bits.rowAddress := RegNext(rowCnt)
  io.krf.bits.columnAddress := RegNext(colCnt)
  io.krf.bits.data := dataBuf
  io.krf.valid := RegNext((state === krlState.sFILLRF) && !stall)

  // kernel ROM interface
  io.rom.enable := (state === krlState.sFILLRF)
  io.rom.address := ramAddr
}

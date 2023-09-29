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
package chisel4ml.tests

import _root_.chisel4ml.implicits._
import _root_.chisel4ml.sequential._
import _root_.lbir.QTensor
import chisel3._
import chisel3.util._
import memories.MemoryGenerator
import chisel4ml.MemWordSize

/** Sliding Window Unit - test bed
  */

class SlidingWindowUnitTestBed(kernel: QTensor, input: QTensor) extends Module {
  val outDataSize: Int = kernel.width * kernel.height * kernel.numChannels * input.dtype.bitwidth

  val io = IO(new Bundle {
    val start = Input(Bool())
    val rrfInValid = Output(Bool())
    val rrfImageValid = Output(Bool())
    val rrfInData = Output(UInt((kernel.width * input.dtype.bitwidth).W))
    val rrfRowAddr = Output(UInt(log2Up(kernel.width).W))
    val rrfChAddr = Output(UInt(log2Up(kernel.numChannels).W))
    val rrfRowWrMode = Output(Bool())
    val rrfOutData = Output(UInt(outDataSize.W))
    val rrfEnd = Output(Bool())
  })

  val swu = Module(new SlidingWindowUnit(input = input, kernel = kernel))
  val rrf = Module(new RollingRegisterFile(input = input, kernel = kernel))

  // For testing purposes we use a prewritten ROM
  val actMem = Module(MemoryGenerator.SRAMInitFromString(hexStr = input.toHexStr, width = MemWordSize.bits))

  rrf.io.shiftRegs := swu.io.shiftRegs
  rrf.io.rowWriteMode := swu.io.rowWriteMode
  rrf.io.rowAddr := swu.io.rowAddr
  rrf.io.chAddr := swu.io.chAddr
  rrf.io.inData := swu.io.data
  rrf.io.inValid := swu.io.valid
  io.rrfOutData := rrf.io.outData

  io.rrfInData := swu.io.data
  io.rrfInValid := swu.io.valid
  io.rrfImageValid := swu.io.imageValid
  io.rrfRowAddr := swu.io.rowAddr
  io.rrfChAddr := swu.io.chAddr
  io.rrfRowWrMode := swu.io.rowWriteMode
  io.rrfEnd := swu.io.end
  /*
  actMem.io.read.enable := swu.io.actRdEna
  actMem.io.read.address := swu.io.actRdAddr
  swu.io.actRdData := actMem.io.read.data*/
  actMem.io.read <> swu.io.actMem
  actMem.io.write.address := 0.U
  actMem.io.write.enable := false.B
  actMem.io.write.data := 0.U

  swu.io.start := io.start

}

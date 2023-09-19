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

import memories.MemoryGenerator
import _root_.chisel4ml.sequential._
import _root_.chisel4ml.util._
import _root_.chisel4ml.implicits._
import _root_.lbir.QTensor
import java.nio.file.Paths
import chisel3._
import chisel3.util._

/** Sliding Window Unit - test bed
  */

class SlidingWindowUnitTestBed(
    kernelSize:   Int,
    kernelDepth:  Int,
    actWidth:     Int,
    actHeight:    Int,
    actParamSize: Int,
    parameters:   QTensor,
  ) extends Module {

  val memWordWidth:     Int = 32
  val actParamsPerWord: Int = memWordWidth / actParamSize
  val actMemValidBits:  Int = actParamsPerWord * actParamSize
  val actMemDepthBits:  Int = actWidth * actHeight * kernelDepth * actParamSize
  val actMemDepthWords: Int = (actMemDepthBits / actMemValidBits) + 1

  val outDataSize: Int = kernelSize * kernelSize * kernelDepth * actParamSize

  val io = IO(new Bundle {
    val start         = Input(Bool())
    val rrfInValid    = Output(Bool())
    val rrfImageValid = Output(Bool())
    val rrfInData     = Output(UInt((kernelSize * actParamSize).W))
    val rrfRowAddr    = Output(UInt(log2Up(kernelSize).W))
    val rrfChAddr     = Output(UInt(log2Up(kernelDepth).W))
    val rrfRowWrMode  = Output(Bool())
    val rrfOutData    = Output(UInt(outDataSize.W))
    val rrfEnd        = Output(Bool())
  })

  val swu = Module(
    new SlidingWindowUnit(
      kernelSize = kernelSize,
      kernelDepth = kernelDepth,
      actWidth = actWidth,
      actHeight = actHeight,
      actParamSize = actParamSize,
    ),
  )

  val rrf = Module(
    new RollingRegisterFile(kernelSize = kernelSize, kernelDepth = kernelDepth, paramSize = actParamSize),
  )

  // For testing purposes we use a prewritten ROM
  val actMem = Module(MemoryGenerator.SRAMInitFromString(hexStr=parameters.toHexStr, width=memWordWidth))

  rrf.io.shiftRegs    := swu.io.shiftRegs
  rrf.io.rowWriteMode := swu.io.rowWriteMode
  rrf.io.rowAddr      := swu.io.rowAddr
  rrf.io.chAddr       := swu.io.chAddr
  rrf.io.inData       := swu.io.data
  rrf.io.inValid      := swu.io.valid
  io.rrfOutData       := rrf.io.outData

  io.rrfInData        := swu.io.data
  io.rrfInValid       := swu.io.valid
  io.rrfImageValid    := swu.io.imageValid
  io.rrfRowAddr       := swu.io.rowAddr
  io.rrfChAddr        := swu.io.chAddr
  io.rrfRowWrMode     := swu.io.rowWriteMode
  io.rrfEnd           := swu.io.end

  actMem.io.rdEna  := swu.io.actRdEna
  actMem.io.rdAddr := swu.io.actRdAddr
  swu.io.actRdData := actMem.io.rdData
  actMem.io.wrAddr := 0.U
  actMem.io.wrEna  := false.B
  actMem.io.wrData := 0.U

  swu.io.start := io.start

}

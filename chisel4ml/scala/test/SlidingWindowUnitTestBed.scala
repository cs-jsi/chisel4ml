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

import chisel3._
import chisel3.util._

import _root_.chisel4ml.memory.ROM
import _root_.chisel4ml._
import _root_.chisel4ml.sequential._
import _root_.chisel4ml.util._
import _root_.lbir.QTensor

/** Sliding Window Unit - test bed
  */

class SlidingWindowUnitTestBed(
    kernelSize:   Int,
    kernelDepth:  Int,
    actWidth:     Int,
    actHeight:    Int,
    actParamSize: Int,
    parameters:   QTensor)
    extends Module {

    val memWordWidth:     Int = 32
    val actParamsPerWord: Int = memWordWidth / actParamSize
    val actMemValidBits:  Int = actParamsPerWord * actParamSize
    val actMemDepthBits:  Int = (actWidth * actHeight * kernelDepth * actParamSize)
    val actMemDepthWords: Int = (actMemDepthBits / actMemValidBits) + 1

    val io = IO(new Bundle {
        val start     = Input(Bool())
        val actRdData = Output(UInt(32.W))
    })

    val swu = Module(
      new SlidingWindowUnit(
        kernelSize = kernelSize,
        kernelDepth = kernelDepth,
        actWidth = actWidth,
        actHeight = actHeight,
        actParamSize = actParamSize
      )
    )

    val rrf = Module(
      new RollingRegisterFile(kernelSize = kernelSize, kernelDepth = kernelDepth, paramSize = actParamSize)
    )

    // For testing purposes we use a prewritten ROM
    val actMem = Module(
      new ROM(depth = actMemDepthWords, width = memWordWidth, memFile = genHexMemoryFile(parameters, layout = "CDHW"))
    )

    rrf.io.shiftRegs    := swu.io.shiftRegs
    rrf.io.rowWriteMode := swu.io.rowWriteMode
    rrf.io.rowAddr      := swu.io.rowAddr
    rrf.io.chAddr       := swu.io.chAddr
    rrf.io.inData       := swu.io.data
    rrf.io.inValid      := swu.io.valid

    actMem.io.rdEna  := swu.io.actRdEn
    actMem.io.rdAddr := swu.io.actRdAddr
    swu.io.actRdData := actMem.io.rdData

    swu.io.start := io.start
    io.actRdData := actMem.io.rdData
}

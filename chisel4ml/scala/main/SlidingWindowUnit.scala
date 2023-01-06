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

import _root_.chisel4ml.util.reqWidth
import _root_.chisel4ml.implicits._
import chisel3.experimental.ChiselEnum

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
    actParamSize: Int)
    extends Module {

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
    val actMemDepthBits:      Int = (actWidth * actHeight * kernelDepth * actParamSize)
    val actMemDepthWords:     Int = (actMemDepthBits / actMemValidBits) + 1

    val collumnAddConstant: Int = actParamSize
    val rowAddConstant:     Int = ((actWidth - kernelSize) + (kernelSize - 1)) * actParamSize
    val chAddConstant:      Int = ((actWidth - kernelSize) + ((actHeight - kernelSize) * actWidth) + 1) * actParamSize
    val constantWireSize:   Int = if (reqWidth(chAddConstant) > 5) reqWidth(chAddConstant) else 5

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
    val subwordAddrReg = RegNext(subwordAddr)
    val data           = Wire(UInt(actParamSize.W))
    val dataReg        = RegNext(data)

    object swuState extends ChiselEnum {
        val sWAIT, sADDCOL, sADDROW, sADDCH, sEND = Value
    }

    val nstate = WireInit(swuState.sWAIT)
    val state  = RegNext(next = nstate, init = swuState.sWAIT)

    val colCnt = RegInit(0.U(reqWidth(kernelSize).W))
    val rowCnt = RegInit(0.U(reqWidth(kernelSize).W))
    val chCnt  = RegInit(0.U(reqWidth(kernelDepth).W))

    io.actRdEn := false.B
    when(io.start === true.B) {
        state       := swuState.sADDCOL
        bitAddr     := 0.U
        io.actRdEn  := true.B
        addConstant := 0.U
    }.otherwise {
        switch(state) {
            is(swuState.sWAIT) {
                addConstant := 0.U
                io.actRdEn  := false.B
            }
            is(swuState.sADDCOL) {
                addConstant := collumnAddConstant.U
                io.actRdEn  := true.B
                colCnt      := colCnt + 1.U
                when(colCnt === (kernelSize - 1).U) {
                    nstate := swuState.sADDCOL
                }
            }
            is(swuState.sADDROW) {
                addConstant := rowAddConstant.U
                io.actRdEn  := true.B
                rowCnt      := rowCnt + 1.U
                when(rowCnt === (kernelSize - 1).U) {
                    nstate := swuState.sADDCH
                }.otherwise {
                    nstate := swuState.sADDCOL
                }
            }
            is(swuState.sADDCH) {
                addConstant := chAddConstant.U
                io.actRdEn  := true.B
                chCnt       := chCnt + 1.U
                nstate      := swuState.sADDCOL
            }
            is(swuState.sEND) {
                addConstant := 0.U
                io.actRdEn  := false.B
            }
        }
    }

    // ADDRESS CALCULATIONS
    bitAddrNext    := bitAddr + addConstant
    // The correction factor corrects for the stuffing that happens (if 32 / bw is not whole) at the end of each word
    correction     := (addConstant >> 5) + ((addConstant(4, 0) + bitAddr(5, 0)) >> 5)
    bitAddrNextMod := bitAddrNext + correction * (memWordWidth - actMemValidBits).U
    bitAddr        := bitAddrNextMod

    io.actRdAddr := (bitAddr >> 5)
    subwordAddr  := bitAddr(5, 0)
    data         := io.actRdData(actMemValidBits - 1, 0).asTypeOf(Vec(actParamsPerWord, UInt(actParamSize.W)))(subwordAddrReg)

    io.shiftRegs    := false.B
    io.rowWriteMode := false.B
    io.rowAddr      := 0.U
    io.chAddr       := 0.U
    io.data         := 0.U
    io.valid        := false.B
}

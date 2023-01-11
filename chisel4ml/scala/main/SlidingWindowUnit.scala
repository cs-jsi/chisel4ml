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
  * kernelSize - Size of one side of a square 2d kernel (i.e. kernelSize=3 for a 3x3 kernel).
  * kernelDepth - The depth of a kernel/actMap, i.e. for a RGB image, the kernel depth is 3. (i.e. num of channels)
  * actWidth - the width of the activation map/image (in num of elements)
  * actHeight - the height of the activation map/image (in num of elements)
  * actParamSize - The bitwidth of each activation parameter.
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

  val actMemDepthBits:     Int = actWidth * actHeight * kernelDepth * actParamSize
  val actMemDepthWords:    Int = (actMemDepthBits / actMemValidBits) + 1
  val actMemDepthRealBits: Int = actMemDepthWords * memWordWidth


  val baseAddConstantBase:  Int = kernelSize * actParamSize
  val colAddConstantBase:   Int = actParamSize
  val rowAddConstantBase:   Int = (actWidth - kernelSize + 1) * actParamSize
  val chAddConstantBase:    Int = ((actWidth - kernelSize) + ((actHeight - kernelSize) * actWidth) + 1) * actParamSize
  val colModRowAddConstantBase: Int = actWidth * actParamSize
  val colModChAddConstantBase:  Int = (actWidth * actHeight * actParamSize) - ((kernelSize - 1) * colModRowAddConstantBase)

  val baseAddConstantMod:  Int = baseAddConstantBase / actMemValidBits
  val colAddConstantMod:   Int = colAddConstantBase  / actMemValidBits
  val rowAddConstantMod:   Int = rowAddConstantBase  / actMemValidBits
  val chAddConstantMod:    Int = chAddConstantBase   / actMemValidBits
  val colModRowAddConstantMod: Int = colModRowAddConstantBase / actMemValidBits
  val colModChAddConstantMod:  Int = colModChAddConstantBase  / actMemValidBits

  val baseAddConstant:  Int = baseAddConstantBase + (baseAddConstantMod * leftoverBits)
  val colAddConstant:   Int = colAddConstantBase +  (colAddConstantMod * leftoverBits)
  val rowAddConstant:   Int = rowAddConstantBase +  (rowAddConstantMod * leftoverBits)
  val chAddConstant:    Int = chAddConstantBase +   (chAddConstantMod  * leftoverBits)
  val colModRowAddConstant: Int = colModRowAddConstantBase + (colModRowAddConstantMod * leftoverBits)
  val colModChAddConstant:  Int = colModChAddConstantBase  + (colModChAddConstantMod * leftoverBits)
  val constantWireSize:     Int = if (reqWidth(colModChAddConstant) >= 5) reqWidth(colModChAddConstant) + 1 else 5

  val io = IO(new Bundle {
    // interface to the RollingRegisterFile module.
    val shiftRegs    = Output(Bool())
    val rowWriteMode = Output(Bool())
    val rowAddr      = Output(UInt(rowAddrWidth.W))
    val chAddr       = Output(UInt(chAddrWidth.W))
    val data         = Output(UInt(wrDataWidth.W))
    val valid        = Output(Bool())
    val imageValid   = Output(Bool())

    // interface to the activation memory
    val actRdEna  = Output(Bool())
    val actRdAddr = Output(UInt(reqWidth(actMemDepthWords).W))
    val actRdData = Input(UInt(memWordWidth.W))

    // control interface
    val start     = Input(Bool())
    val end       = Output(Bool())
  })


  val updateBase1     = WireInit(false.B)
  val updateBase2     = WireInit(false.B)
  val baseBitAddr     = RegInit(0.U(reqWidth(actMemDepthRealBits).W))
  val nbaseBitAddr    = WireInit(0.U(reqWidth(actMemDepthRealBits).W))
  val nbaseBitAddrMod = WireInit(0.U(reqWidth(actMemDepthRealBits).W))
  val bitAddr         = RegInit(0.U(reqWidth(actMemDepthRealBits).W)) // bit addressing, because params can be any width
  val nbitAddr        = Wire(UInt(reqWidth(actMemDepthRealBits).W))
  val nbitAddrMod     = Wire(UInt(reqWidth(actMemDepthRealBits).W))
  val addConstant     = WireInit(0.U(constantWireSize.W))
  val addConstantMod  = WireInit(0.U(constantWireSize.W))
  val subwordAddr     = Wire(UInt(reqWidth(memWordWidth).W))      // Which part of the memory word are we looking at?
  val ramDataAsVec    = Wire(Vec(actParamsPerWord, UInt(actParamSize.W)))
  val dataBuf         = RegInit(VecInit(Seq.fill(kernelSize)(0.U(actParamSize.W))))
  val dataIndex       = Wire(UInt())

  object swuState extends ChiselEnum {
    val sWAITSTART = Value(0.U)
    val sROWMODE   = Value(1.U)
    val sCOLMODE   = Value(2.U)
    val sEND       = Value(3.U)
    val sERROR     = Value(4.U)
  }

  object addState extends ChiselEnum {
    val sADDCOL    = Value(0.U)
    val sADDROW    = Value(1.U)
    val sADDCH     = Value(2.U)
  }

  val nstate = WireInit(swuState.sERROR)
  val state  = RegInit(swuState.sWAITSTART)
  val prevstate  = RegNext(next = state, init = swuState.sWAITSTART)

  val stall = RegInit(false.B)
  val prevstall = RegNext(next = stall, init = false.B)

  val naddstate = WireInit(addState.sADDCOL)
  val addstate  = RegInit(addState.sADDCOL)

  val colCnt = RegInit(0.U(reqWidth(kernelSize).W))
  val rowCnt = RegInit(0.U(reqWidth(kernelSize).W))
  val chCnt  = RegInit(0.U(reqWidth(kernelDepth).W))
  val horizCnt  = RegInit(0.U(reqWidth(actWidth).W))
  val nhorizCnt = WireInit(0.U(reqWidth(actWidth).W))
  val vertiCnt  = RegInit(0.U(reqWidth(actHeight).W))
  val nvertiCnt = WireInit(0.U(reqWidth(actHeight).W))
  val allCntZero = Wire(Bool())
  val rowAndChCntZero = Wire(Bool())
  val allCntMax = Wire(Bool())
  val rowAndChCntMax = Wire(Bool())

  ////// NEXT STATE LOGIC //////
  nstate := state
  when (state === swuState.sWAITSTART) {
    when (io.start) {
      nstate := swuState.sROWMODE
    }.otherwise {
      nstate := swuState.sWAITSTART
    }
  }.elsewhen (state === swuState.sERROR) {
      nstate := swuState.sERROR
  }.elsewhen (state === swuState.sEND) {
      nstate := swuState.sWAITSTART
  }.elsewhen (state === swuState.sROWMODE) {
    when (allCntMax) {
      nstate := swuState.sCOLMODE
    }. otherwise {
      nstate := swuState.sROWMODE
    }
  }.elsewhen (state === swuState.sCOLMODE) {
    when (rowAndChCntMax &&
          horizCnt === (actWidth - kernelSize).U &&
          vertiCnt === (actHeight - kernelSize).U) {
      nstate := swuState.sEND
    }.elsewhen (rowAndChCntMax && horizCnt === (actWidth - kernelSize).U) {
      nstate := swuState.sROWMODE
    }.otherwise {
      nstate := swuState.sCOLMODE
    }
  }
  when (!stall) {
    state := nstate
  }


  ///// NEXT ADD STATE LOGIC /////
  when (state === swuState.sROWMODE) {
    when (allCntMax) {
      naddstate := addState.sADDROW // state will change to COLMODE - so add mode should be addrow
    }.elsewhen (addstate === addState.sADDROW || addstate  === addState.sADDCH) {
      naddstate := addState.sADDCOL
    }.elsewhen (colCnt === (kernelSize - 2).U &&
                rowCnt === (kernelSize - 1).U) {
      naddstate := addState.sADDCH
    }.elsewhen (colCnt === (kernelSize - 2).U) {
      naddstate := addState.sADDROW
    }.otherwise {
      naddstate := addState.sADDCOL
    }
  }.elsewhen (state === swuState.sCOLMODE) {
    when (rowAndChCntMax &&
          horizCnt === (actWidth - kernelSize).U) {
      naddstate := addState.sADDCOL // state will change to ROWMODE - so add mode should be addcol
    }.elsewhen(rowCnt === (kernelSize - 2).U) {
      naddstate := addState.sADDCH
    }.otherwise {
      naddstate := addState.sADDROW
    }
  }
  when (!stall) {
    addstate := naddstate
  }

  /////// STALL LOGIC //////
  when (((bitAddr >> 5) =/= (nbitAddrMod >> 5)) && !stall) {
    stall := true.B
  }.elsewhen(stall) {
    stall := false.B
  }


  ////// CONSTANTS //////
  addConstant := 0.U
  when (state === swuState.sROWMODE) {
    switch(addstate) {
      is(addState.sADDCOL) {
        addConstant := colAddConstant.U
        addConstantMod := colAddConstantMod.U
      }
      is(addState.sADDROW) {
        addConstant := rowAddConstant.U
        addConstantMod := rowAddConstantMod.U
      }
      is(addState.sADDCH)  {
        addConstant := chAddConstant.U
        addConstantMod := chAddConstantMod.U
      }
    }
  }.otherwise {
    switch(addstate) {
      is(addState.sADDROW) {
        addConstant := colModRowAddConstant.U
        addConstantMod := colModRowAddConstantMod.U
      }
      is(addState.sADDCH)  {
        addConstant := colModChAddConstant.U
        addConstantMod := colModChAddConstantMod.U
      }
    }
  }

  ////// COUNTERS //////
  colCnt := colCnt
  rowCnt := rowCnt
  chCnt := chCnt
  when (!stall) {
    when (state === swuState.sWAITSTART ||
         (state =/= nstate)) {
      colCnt := 0.U
      rowCnt := 0.U
      chCnt  := 0.U
    }.elsewhen ((state === swuState.sROWMODE ||
                 state === swuState.sCOLMODE ||
                 state === swuState.sEND)) {
      switch (addstate) {
        is (addState.sADDCOL) {
          colCnt := colCnt + 1.U
        }
        is (addState.sADDROW) {
          rowCnt := rowCnt + 1.U
          colCnt := 0.U
        }
        is (addState.sADDCH) {
          when (chCnt === (kernelDepth - 1).U) {
            chCnt := 0.U
          }.otherwise {
            chCnt  := chCnt + 1.U
          }
          rowCnt := 0.U
          colCnt := 0.U
        }
      }
    }
  }

  rowAndChCntZero := (rowCnt === 0.U) && (chCnt === 0.U)
  allCntZero      := (colCnt === 0.U) && rowAndChCntZero
  if (kernelDepth > 1) {
    rowAndChCntMax := (rowCnt === (kernelSize - 1).U) && (chCnt === (kernelDepth - 1).U)
  } else {
    rowAndChCntMax := (rowCnt === (kernelSize - 1).U)
  }
  allCntMax := (colCnt === (kernelSize - 1).U) && rowAndChCntMax


  nbaseBitAddr := baseBitAddr
  nhorizCnt := horizCnt
  nvertiCnt := vertiCnt
  updateBase1 := false.B
  updateBase2 := false.B
  when (io.start && !RegNext(io.start)) {
    nhorizCnt := 0.U
    nvertiCnt := 0.U
    nbaseBitAddr := baseAddConstant.U
    updateBase1 := true.B
  }.elsewhen (state === swuState.sROWMODE && allCntMax) {
    nhorizCnt := horizCnt + 1.U
    nbaseBitAddr := baseBitAddr + actParamSize.U
    updateBase2 := true.B
  }.elsewhen (state === swuState.sCOLMODE) {
    when(horizCnt < (actWidth - kernelSize).U && rowAndChCntMax) {
      updateBase2 := true.B
      nhorizCnt := horizCnt + 1.U
      when (horizCnt =/= 0.U) {
        nbaseBitAddr := baseBitAddr + actParamSize.U
      }
    }.elsewhen(rowAndChCntMax) {
      updateBase1 := true.B
      nhorizCnt := 0.U
      nvertiCnt := vertiCnt + 1.U
      nbaseBitAddr := baseBitAddr + baseAddConstant.U
    }
  }
  when ((((nbaseBitAddr >> 5) - (baseBitAddr >> 5) =/= baseAddConstantMod.U) && updateBase1) ||
        nbaseBitAddr(4,0) === actMemValidBits.U) {
    nbaseBitAddrMod := nbaseBitAddr + leftoverBits.U
  }.otherwise {
    nbaseBitAddrMod := nbaseBitAddr
  }
  when (!stall && (updateBase1 || updateBase2)) {
    baseBitAddr := nbaseBitAddrMod
    horizCnt    := nhorizCnt
    vertiCnt    := nvertiCnt
  }

  ////// ADDRESS CALCULATIONS //////
  nbitAddr := bitAddr
  when((allCntMax && state === swuState.sROWMODE && !stall) ||
       (rowAndChCntMax && state === swuState.sCOLMODE && !stall) ) {
    nbitAddr := baseBitAddr
  }.elsewhen((state === swuState.sROWMODE || state === swuState.sCOLMODE) && !stall) {
    nbitAddr := bitAddr + addConstant
  }

  nbitAddrMod := nbitAddr
  when ((state === swuState.sROWMODE || state === swuState.sCOLMODE) &&
         !stall &&
         (((((nbitAddr >> 5) - (bitAddr >> 5)) =/= addConstantMod) && nbitAddr > bitAddr) ||
         (nbitAddr(4,0) === actMemValidBits.U))) {
    nbitAddrMod := nbitAddr + leftoverBits.U
  }.otherwise {
    nbitAddrMod := nbitAddr
  }
  bitAddr := nbitAddrMod

  ////// ACTIVATION MEMORY INTERFACE //////
  io.actRdEna  := (state === swuState.sROWMODE  || state === swuState.sCOLMODE)
  io.actRdAddr := (bitAddr >> 5)

  // Subword (bit) address is translated into indexes via a lookup table
  subwordAddr := bitAddr(4, 0)
  dataIndex := MuxLookup(subwordAddr,
                         0.U,
                         Seq.tabulate(actParamsPerWord)(_ * actParamSize).zipWithIndex.map(x => (x._1.U -> x._2.U)))
  ramDataAsVec := io.actRdData(actMemValidBits - 1, 0).asTypeOf(ramDataAsVec)
  when((prevstate === swuState.sROWMODE) ||
       (state === swuState.sROWMODE && prevstate === swuState.sWAITSTART)) {
    dataBuf(colCnt) := ramDataAsVec(dataIndex)
  }.elsewhen(prevstate === swuState.sCOLMODE) {
    dataBuf(rowCnt) := ramDataAsVec(dataIndex)
  }

  // //// ROLLING REGISTER FILE INTERFACE //////
  io.shiftRegs    := RegNext(io.imageValid)
  io.rowWriteMode := (prevstate === swuState.sROWMODE) && !io.shiftRegs
  io.rowAddr      := RegNext(rowCnt)
  if (kernelDepth > 1) {
    io.chAddr := RegNext(chCnt)
  } else {
    io.chAddr := 0.U
  }
  io.data         := dataBuf.asUInt
  when(io.rowWriteMode) {
    io.valid := RegNext(colCnt === (kernelSize - 1).U) && !prevstall
  }.otherwise {
    io.valid := RegNext(rowCnt === (kernelSize - 1).U) && rowCnt === 0.U && !prevstall // rowCnt reset
  }
  io.imageValid := (RegNext(allCntMax) && prevstate === swuState.sROWMODE && !prevstall) ||
                   (RegNext(rowAndChCntMax) && prevstate === swuState.sCOLMODE && !prevstall)

  ////// CONTROL INTERFACE ///////
  io.end := (state === swuState.sEND) && (prevstate =/= swuState.sEND) // in case of stall, we add prevstate
}

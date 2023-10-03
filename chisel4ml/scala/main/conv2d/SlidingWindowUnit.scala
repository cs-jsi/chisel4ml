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
import memories.SRAMRead
import chisel4ml.implicits._
import chisel4ml.MemWordSize

/** Sliding Window Unit */
class SlidingWindowUnit(input: lbir.QTensor, kernel: lbir.QTensor) extends Module {
  val inputValidBits:      Int = input.paramsPerWord * input.dtype.bitwidth
  val inputLefoverBits:    Int = MemWordSize.bits - inputValidBits
  val actMemDepthBits:     Int = input.numParams * input.dtype.bitwidth
  val actMemDepthWords:    Int = math.ceil(actMemDepthBits.toFloat / inputValidBits.toFloat).toInt
  val actMemDepthRealBits: Int = actMemDepthWords * MemWordSize.bits

  val baseAddConstantBase: Int = kernel.width * input.dtype.bitwidth
  val colAddConstantBase:  Int = input.dtype.bitwidth
  val rowAddConstantBase:  Int = (input.height - kernel.width + 1) * input.dtype.bitwidth
  val chAddConstantBase: Int =
    ((input.height - kernel.width) + ((input.width - kernel.width) * input.height) + 1) * input.dtype.bitwidth
  val colModRowAddConstantBase: Int = input.height * input.dtype.bitwidth
  val colModChAddConstantBase: Int =
    (input.height * input.width * input.dtype.bitwidth) - ((kernel.width - 1) * colModRowAddConstantBase)

  val baseAddConstantMod:      Int = baseAddConstantBase / inputValidBits
  val colAddConstantMod:       Int = colAddConstantBase / inputValidBits
  val rowAddConstantMod:       Int = rowAddConstantBase / inputValidBits
  val chAddConstantMod:        Int = chAddConstantBase / inputValidBits
  val colModRowAddConstantMod: Int = colModRowAddConstantBase / inputValidBits
  val colModChAddConstantMod:  Int = colModChAddConstantBase / inputValidBits

  val baseAddConstant:      Int = baseAddConstantBase + (baseAddConstantMod * inputLefoverBits)
  val colAddConstant:       Int = colAddConstantBase + (colAddConstantMod * inputLefoverBits)
  val rowAddConstant:       Int = rowAddConstantBase + (rowAddConstantMod * inputLefoverBits)
  val chAddConstant:        Int = chAddConstantBase + (chAddConstantMod * inputLefoverBits)
  val colModRowAddConstant: Int = colModRowAddConstantBase + (colModRowAddConstantMod * inputLefoverBits)
  val colModChAddConstant:  Int = colModChAddConstantBase + (colModChAddConstantMod * inputLefoverBits)
  val constantWireSize:     Int = if (log2Up(colModChAddConstant) >= 5) log2Up(colModChAddConstant) + 1 else 5

  val io = IO(new Bundle {
    // interface to the RollingRegisterFile module.
    val shiftRegs = Output(Bool())
    val rowWriteMode = Output(Bool())
    val rowAddr = Output(UInt(log2Up(kernel.width).W))
    val chAddr = Output(UInt(log2Up(kernel.numChannels).W))
    val data = Output(UInt((kernel.width * input.dtype.bitwidth).W))
    val valid = Output(Bool())
    val imageValid = Output(Bool())

    // interface to the activation memory
    val actMem = Flipped(new SRAMRead(actMemDepthWords, MemWordSize.bits))

    // control interface
    val start = Input(Bool())
    val end = Output(Bool())
  })

  val updateBase1 = WireInit(false.B)
  val updateBase2 = WireInit(false.B)
  val baseBitAddr = RegInit(0.U(log2Up(actMemDepthRealBits).W))
  val nbaseBitAddr = WireInit(0.U(log2Up(actMemDepthRealBits).W))
  val nbaseBitAddrMod = WireInit(0.U(log2Up(actMemDepthRealBits).W))
  val bitAddr = RegInit(0.U(log2Up(actMemDepthRealBits).W)) // bit addressing, because params can be any width
  val nbitAddr = Wire(UInt(log2Up(actMemDepthRealBits).W))
  val nbitAddrMod = Wire(UInt(log2Up(actMemDepthRealBits).W))
  val addConstant = WireInit(0.U(constantWireSize.W))
  val addConstantMod = WireInit(0.U(constantWireSize.W))
  val subwordAddr = Wire(UInt(log2Up(MemWordSize.bits).W)) // Which part of the memory word are we looking at?
  val ramDataAsVec = Wire(Vec(input.paramsPerWord, UInt(input.dtype.bitwidth.W)))
  val dataBuf = RegInit(VecInit(Seq.fill(kernel.width)(0.U(input.dtype.bitwidth.W))))
  val dataIndex = Wire(UInt())

  object swuState extends ChiselEnum {
    val sWAITSTART = Value(0.U)
    val sROWMODE = Value(1.U)
    val sCOLMODE = Value(2.U)
    val sEND = Value(3.U)
    val sERROR = Value(4.U)
  }

  object addState extends ChiselEnum {
    val sADDCOL = Value(0.U)
    val sADDROW = Value(1.U)
    val sADDCH = Value(2.U)
  }

  val nstate = WireInit(swuState.sERROR)
  val state = RegInit(swuState.sWAITSTART)
  val prevstate = RegNext(next = state, init = swuState.sWAITSTART)

  val stall = RegInit(false.B)
  val prevstall = RegNext(next = stall, init = false.B)

  val naddstate = WireInit(addState.sADDCOL)
  val addstate = RegInit(addState.sADDCOL)

  val colCnt = RegInit(0.U(log2Up(kernel.width).W))
  val rowCnt = RegInit(0.U(log2Up(kernel.width).W))
  val chCnt = RegInit(0.U(log2Up(kernel.numChannels).W))
  val horizCnt = RegInit(0.U(log2Up(input.height).W))
  val nhorizCnt = WireInit(0.U(log2Up(input.height).W))
  val vertiCnt = RegInit(0.U(log2Up(input.width).W))
  val nvertiCnt = WireInit(0.U(log2Up(input.width).W))
  val allCntZero = Wire(Bool())
  val rowAndChCntZero = Wire(Bool())
  val allCntMax = Wire(Bool())
  val rowAndChCntMax = Wire(Bool())

  ////// NEXT STATE LOGIC //////
  nstate := state
  when(state === swuState.sWAITSTART) {
    when(io.start) {
      nstate := swuState.sROWMODE
    }.otherwise {
      nstate := swuState.sWAITSTART
    }
  }.elsewhen(state === swuState.sERROR) {
    nstate := swuState.sERROR
  }.elsewhen(state === swuState.sEND) {
    nstate := swuState.sWAITSTART
  }.elsewhen(state === swuState.sROWMODE) {
    when(allCntMax) {
      nstate := swuState.sCOLMODE
    }.otherwise {
      nstate := swuState.sROWMODE
    }
  }.elsewhen(state === swuState.sCOLMODE) {
    when(
      rowAndChCntMax &&
        horizCnt === (input.height - kernel.width).U &&
        vertiCnt === (input.width - kernel.width).U
    ) {
      nstate := swuState.sEND
    }.elsewhen(rowAndChCntMax && horizCnt === (input.height - kernel.width).U) {
      nstate := swuState.sROWMODE
    }.otherwise {
      nstate := swuState.sCOLMODE
    }
  }
  when(!stall) {
    state := nstate
  }

  ///// NEXT ADD STATE LOGIC /////
  when(state === swuState.sROWMODE) {
    when(allCntMax) {
      naddstate := addState.sADDROW // state will change to COLMODE - so add mode should be addrow
    }.elsewhen(addstate === addState.sADDROW || addstate === addState.sADDCH) {
      naddstate := addState.sADDCOL
    }.elsewhen(
      colCnt === (kernel.width - 2).U &&
        rowCnt === (kernel.width - 1).U
    ) {
      naddstate := addState.sADDCH
    }.elsewhen(colCnt === (kernel.width - 2).U) {
      naddstate := addState.sADDROW
    }.otherwise {
      naddstate := addState.sADDCOL
    }
  }.elsewhen(state === swuState.sCOLMODE) {
    when(
      rowAndChCntMax &&
        horizCnt === (input.height - kernel.width).U
    ) {
      naddstate := addState.sADDCOL // state will change to ROWMODE - so add mode should be addcol
    }.elsewhen(rowCnt === (kernel.width - 2).U) {
      naddstate := addState.sADDCH
    }.otherwise {
      naddstate := addState.sADDROW
    }
  }
  when(!stall) {
    addstate := naddstate
  }

  /////// STALL LOGIC //////
  when(((bitAddr >> 5) =/= (nbitAddrMod >> 5)) && !stall) {
    stall := true.B
  }.elsewhen(stall) {
    stall := false.B
  }

  ////// CONSTANTS //////
  addConstant := 0.U
  when(state === swuState.sROWMODE) {
    switch(addstate) {
      is(addState.sADDCOL) {
        addConstant := colAddConstant.U
        addConstantMod := colAddConstantMod.U
      }
      is(addState.sADDROW) {
        addConstant := rowAddConstant.U
        addConstantMod := rowAddConstantMod.U
      }
      is(addState.sADDCH) {
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
      is(addState.sADDCH) {
        addConstant := colModChAddConstant.U
        addConstantMod := colModChAddConstantMod.U
      }
    }
  }

  ////// COUNTERS //////
  colCnt := colCnt
  rowCnt := rowCnt
  chCnt := chCnt
  when(!stall) {
    when(
      state === swuState.sWAITSTART ||
        (state =/= nstate)
    ) {
      colCnt := 0.U
      rowCnt := 0.U
      chCnt := 0.U
    }.elsewhen(
      (state === swuState.sROWMODE ||
        state === swuState.sCOLMODE ||
        state === swuState.sEND)
    ) {
      switch(addstate) {
        is(addState.sADDCOL) {
          colCnt := colCnt + 1.U
        }
        is(addState.sADDROW) {
          rowCnt := rowCnt + 1.U
          colCnt := 0.U
        }
        is(addState.sADDCH) {
          when(chCnt === (kernel.numChannels - 1).U) {
            chCnt := 0.U
          }.otherwise {
            chCnt := chCnt + 1.U
          }
          rowCnt := 0.U
          colCnt := 0.U
        }
      }
    }
  }

  rowAndChCntZero := (rowCnt === 0.U) && (chCnt === 0.U)
  allCntZero := (colCnt === 0.U) && rowAndChCntZero
  if (kernel.numChannels > 1) {
    rowAndChCntMax := (rowCnt === (kernel.width - 1).U) && (chCnt === (kernel.numChannels - 1).U)
  } else {
    rowAndChCntMax := (rowCnt === (kernel.width - 1).U)
  }
  allCntMax := (colCnt === (kernel.width - 1).U) && rowAndChCntMax

  nbaseBitAddr := baseBitAddr
  nhorizCnt := horizCnt
  nvertiCnt := vertiCnt
  updateBase1 := false.B
  updateBase2 := false.B
  when(io.start && !RegNext(io.start)) {
    nhorizCnt := 0.U
    nvertiCnt := 0.U
    nbaseBitAddr := baseAddConstant.U
    updateBase1 := true.B
  }.elsewhen(state === swuState.sROWMODE && allCntMax) {
    nhorizCnt := horizCnt + 1.U
    nbaseBitAddr := baseBitAddr + input.dtype.bitwidth.U
    updateBase2 := true.B
  }.elsewhen(state === swuState.sCOLMODE) {
    when(horizCnt < (input.height - kernel.width).U && rowAndChCntMax) {
      updateBase2 := true.B
      nhorizCnt := horizCnt + 1.U
      when(horizCnt =/= 0.U) {
        nbaseBitAddr := baseBitAddr + input.dtype.bitwidth.U
      }
    }.elsewhen(rowAndChCntMax) {
      updateBase1 := true.B
      nhorizCnt := 0.U
      nvertiCnt := vertiCnt + 1.U
      nbaseBitAddr := baseBitAddr + baseAddConstant.U
    }
  }
  when(
    (((nbaseBitAddr >> 5) - (baseBitAddr >> 5) =/= baseAddConstantMod.U) && updateBase1) ||
      nbaseBitAddr(4, 0) === inputValidBits.U
  ) {
    nbaseBitAddrMod := nbaseBitAddr + inputLefoverBits.U
  }.otherwise {
    nbaseBitAddrMod := nbaseBitAddr
  }
  when(io.start && !RegNext(io.start)) {
    baseBitAddr := nbaseBitAddr
  }.elsewhen(!stall && (updateBase1 || updateBase2)) {
    baseBitAddr := nbaseBitAddrMod
    horizCnt := nhorizCnt
    vertiCnt := nvertiCnt
  }

  ////// ADDRESS CALCULATIONS //////
  nbitAddr := bitAddr
  when(io.start && !RegNext(io.start)) {
    nbitAddr := 0.U
  }.elsewhen(
    (allCntMax && state === swuState.sROWMODE && !stall) ||
      (rowAndChCntMax && state === swuState.sCOLMODE && !stall)
  ) {
    nbitAddr := baseBitAddr
  }.elsewhen((state === swuState.sROWMODE || state === swuState.sCOLMODE) && !stall) {
    nbitAddr := bitAddr + addConstant
  }

  nbitAddrMod := nbitAddr
  when(
    (state === swuState.sROWMODE || state === swuState.sCOLMODE) &&
      !stall &&
      (((((nbitAddr >> 5) - (bitAddr >> 5)) =/= addConstantMod) && nbitAddr > bitAddr) ||
        (nbitAddr(4, 0) === inputValidBits.U))
  ) {
    nbitAddrMod := nbitAddr + inputLefoverBits.U
  }.otherwise {
    nbitAddrMod := nbitAddr
  }
  bitAddr := nbitAddrMod

  ////// ACTIVATION MEMORY INTERFACE //////
  io.actMem.enable := (state === swuState.sROWMODE || state === swuState.sCOLMODE)
  io.actMem.address := (bitAddr >> 5)

  // Subword (bit) address is translated into indexes via a lookup table
  subwordAddr := bitAddr(4, 0)
  dataIndex := MuxLookup(
    subwordAddr,
    0.U,
    Seq.tabulate(input.paramsPerWord)(_ * input.dtype.bitwidth).zipWithIndex.map(x => (x._1.U -> x._2.U))
  )
  ramDataAsVec := io.actMem.data(inputValidBits - 1, 0).asTypeOf(ramDataAsVec)
  when(
    (prevstate === swuState.sROWMODE) ||
      (state === swuState.sROWMODE && prevstate === swuState.sWAITSTART)
  ) {
    dataBuf(colCnt) := ramDataAsVec(dataIndex)
  }.elsewhen(prevstate === swuState.sCOLMODE) {
    dataBuf(rowCnt) := ramDataAsVec(dataIndex)
  }

  // //// ROLLING REGISTER FILE INTERFACE //////
  io.shiftRegs := RegNext(io.imageValid)
  io.rowWriteMode := (prevstate === swuState.sROWMODE) && !io.shiftRegs
  io.rowAddr := RegNext(rowCnt)
  if (kernel.numChannels > 1) {
    io.chAddr := RegNext(chCnt)
  } else {
    io.chAddr := 0.U
  }
  io.data := dataBuf.asUInt
  when(io.rowWriteMode) {
    io.valid := RegNext(colCnt === (kernel.width - 1).U) && !prevstall
  }.otherwise {
    io.valid := RegNext(rowCnt === (kernel.width - 1).U) && rowCnt === 0.U && !prevstall // rowCnt reset
  }
  io.imageValid := (RegNext(allCntMax) && prevstate === swuState.sROWMODE && !prevstall) ||
    (RegNext(rowAndChCntMax) && prevstate === swuState.sCOLMODE && !prevstall)

  ////// CONTROL INTERFACE ///////
  io.end := (state === swuState.sEND) && (prevstate =/= swuState.sEND) // in case of stall, we add prevstate
}

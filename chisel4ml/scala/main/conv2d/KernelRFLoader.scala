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

class KernelRFLoaderControlIO(l: lbir.Conv2DConfig) extends Bundle {
  val kernelDone = Output(Bool())
  val nextActive = if (l.depthwise) Some(Input(Bool())) else None
  val loadKernel = Flipped(Valid(UInt(log2Up(l.kernel.numKernels).W)))
}

/** KernelRFLoader
  */
class KernelRFLoader[W <: Bits](l: lbir.Conv2DConfig) extends Module {
  val kernelMemValidBits: Int = l.kernel.paramsPerWord * l.kernel.dtype.bitwidth

  val io = IO(new Bundle {
    val krf = Valid(l.kernel.getType[W])
    val rom = Flipped(new SRAMRead(depth = l.kernel.memDepth, width = MemWordSize.bits))
    val ctrl = new KernelRFLoaderControlIO(l)
  })

  object krlState extends ChiselEnum {
    val sWAIT = Value(0.U)
    val sFILLRF = Value(1.U)
    val sACTIVEFULL = Value(2.U)
    val sEND = Value(3.U)
  }
  val state = RegInit(krlState.sWAIT)

  val stall = Wire(Bool())
  val romDataAsVec = Wire(Vec(l.kernel.paramsPerWord, UInt(l.kernel.dtype.bitwidth.W)))
  val romBaseAddr = RegInit(0.U(log2Up(l.kernel.memDepth).W))
  val romBaseAddrWire = MuxLookup(
    io.ctrl.loadKernel.bits,
    0.U,
    (0 until l.kernel.numKernels).map(i => (i.U -> (i * l.kernel.memDepthOneKernel).U))
  )
  val (wordElemCnt, wordElemWrap) = Counter(0 until l.kernel.paramsPerWord, io.krf.valid, io.ctrl.loadKernel.valid)
  val (_, activeElemWrap) = Counter(0 until l.kernel.numActiveParams(l.depthwise), io.krf.valid)
  val (_, kernelElemWrap) = Counter(0 until l.kernel.numKernelParams, io.krf.valid, io.ctrl.loadKernel.valid)
  val (romAddrCntValue, _) =
    Counter(
      0 to l.kernel.memDepth,
      wordElemCnt === (l.kernel.paramsPerWord - 1).U || state === krlState.sEND,
      io.ctrl.loadKernel.valid
    )

  when(io.ctrl.loadKernel.valid) {
    romBaseAddr := romBaseAddrWire
  }

  ///////////////////////
  // NEXT STATE LOGIC  //
  ///////////////////////
  when(state === krlState.sWAIT && io.ctrl.loadKernel.valid) {
    state := krlState.sFILLRF
  }.elsewhen(state === krlState.sFILLRF && activeElemWrap) {
    if (l.depthwise) {
      when(kernelElemWrap) {
        state := krlState.sEND
      }.otherwise {
        state := krlState.sACTIVEFULL
      }
    } else {
      state := krlState.sEND
    }
  }.elsewhen(state === krlState.sACTIVEFULL && io.ctrl.nextActive.getOrElse(false.B)) {
    state := krlState.sFILLRF
  }.elsewhen(state === krlState.sEND) {
    state := krlState.sWAIT
  }

  romDataAsVec := io.rom.data.asTypeOf(romDataAsVec)

  ///////////////////////
  // MODULE INTERFACES //
  ///////////////////////
  io.ctrl.kernelDone := state === krlState.sEND

  // kernel ROM interface
  io.rom.enable := true.B // TODO
  io.rom.address := romAddrCntValue + romBaseAddr
  stall := RegNext(wordElemCnt === (l.kernel.paramsPerWord - 1).U || io.ctrl.loadKernel.valid)

  // kernel RF interface
  io.krf.bits := romDataAsVec(wordElemCnt).asTypeOf(io.krf.bits)
  io.krf.valid := state === krlState.sFILLRF && !stall
}

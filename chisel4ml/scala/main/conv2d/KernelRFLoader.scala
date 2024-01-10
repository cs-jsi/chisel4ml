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
import chisel4ml.implicits._
import memories.SRAMRead

class KernelRFLoaderControlIO(l: lbir.Conv2DConfig) extends Bundle {
  val lastActiveLoaded = Output(Bool())
  val nextActive = if (l.depthwise) Some(Input(Bool())) else None
  val loadKernel = Flipped(Valid(UInt(log2Up(l.kernel.numKernels).W)))
}

/** KernelRFLoader
  */
class KernelRFLoader[W <: Bits](l: lbir.Conv2DConfig) extends Module {
  val io = IO(new Bundle {
    val krf = Valid(l.kernel.getType[W])
    val rom = Flipped(new SRAMRead(depth = l.kernel.memDepth(32), width = 32))
    val ctrl = new KernelRFLoaderControlIO(l)
  })

  object KrlState extends ChiselEnum {
    val sWAIT = Value(0.U)
    val sFILLRF = Value(1.U)
    val sACTIVEFULL = Value(2.U)
    val sEND = Value(3.U)
  }
  val state = RegInit(KrlState.sWAIT)

  val stall = Wire(Bool())
  val romBaseAddr = RegInit(0.U(log2Up(l.kernel.memDepth(32)).W))
  val romBaseAddrWire = MuxLookup(
    io.ctrl.loadKernel.bits,
    0.U,
    (0 until l.kernel.numKernels).map(i => (i.U -> (i * l.kernel.memDepthOneKernel(32)).U))
  )
  val (wordElemCnt, wordElemWrap) = Counter(0 until l.kernel.paramsPerWord(), io.krf.valid, io.ctrl.loadKernel.valid)
  val (_, activeElemWrap) = Counter(0 until l.kernel.numActiveParams(l.depthwise), io.krf.valid)
  val (_, kernelElemWrap) = Counter(0 until l.kernel.numKernelParams, io.krf.valid, io.ctrl.loadKernel.valid)
  val (channelCounter, _) =
    Counter(0 until l.kernel.numChannels, io.ctrl.nextActive.getOrElse(false.B), io.ctrl.loadKernel.valid)
  val (romAddrCntValue, _) =
    Counter(
      0 to l.kernel.memDepth(32),
      wordElemWrap,
      state === KrlState.sEND
    )

  when(io.ctrl.loadKernel.valid) {
    romBaseAddr := romBaseAddrWire
  }

  ///////////////////////
  // NEXT STATE LOGIC  //
  ///////////////////////
  when(state === KrlState.sWAIT && io.ctrl.loadKernel.valid) {
    state := KrlState.sFILLRF
  }.elsewhen(state === KrlState.sFILLRF && activeElemWrap) {
    if (l.depthwise) {
      when(kernelElemWrap) {
        state := KrlState.sEND
      }.otherwise {
        state := KrlState.sACTIVEFULL
      }
    } else {
      state := KrlState.sEND
    }
  }.elsewhen(state === KrlState.sACTIVEFULL && io.ctrl.nextActive.getOrElse(true.B)) {
    state := KrlState.sFILLRF
  }.elsewhen(state === KrlState.sEND) {
    state := KrlState.sWAIT
  }

  io.ctrl.lastActiveLoaded := channelCounter === (l.kernel.numChannels - 1).U

  // kernel ROM interface
  io.rom.enable := true.B // TODO
  io.rom.address := romAddrCntValue + romBaseAddr
  stall := RegNext(wordElemCnt === (l.kernel.paramsPerWord() - 1).U || io.ctrl.loadKernel.valid)

  // kernel RF interface
  val validBits = l.kernel.paramsPerWord() * l.kernel.dtype.bitwidth
  val romDataAsVec = io.rom.data(validBits - 1, 0).asTypeOf(Vec(l.kernel.paramsPerWord(), l.kernel.getType[W]))
  io.krf.bits := romDataAsVec(wordElemCnt)
  io.krf.valid := state === KrlState.sFILLRF && !stall

  when(state =/= KrlState.sWAIT) {
    assert(io.ctrl.loadKernel.valid === false.B)
  }
}

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

class InputDataMover(input: lbir.QTensor) extends Module {
  object IDMState extends ChiselEnum {
    val sWAIT = Value(0.U)
    val sMOVEDATA = Value(1.U)
  }

  val io = IO(new Bundle {
    val nextElement = Decoupled(UInt(input.dtype.bitwidth.W))
    val actMem = Flipped(new SRAMRead(input.memDepth, MemWordSize.bits))
    val actMemWrittenTo = Input(UInt(input.memDepth.W))
    val start = Input(Bool())
    val done = Output(Bool())
  })

  val (elemCntValue, elemCntWrap) = Counter(0 until input.numParams, io.nextElement.fire)
  val (memLineCntValue, memlineCntWrap) = Counter(0 until input.paramsPerWord, io.nextElement.fire, io.done)
  val (addrCntValue, addrCntWrap) = Counter(0 until input.memDepth, memlineCntWrap || io.done)

  val state = RegInit(IDMState.sWAIT)
  when(io.start) {
    state := IDMState.sMOVEDATA
  }.elsewhen(elemCntWrap) {
    state := IDMState.sWAIT
  }

  io.actMem.address := addrCntValue
  io.actMem.enable := state === IDMState.sMOVEDATA

  val actMemAsVec = io.actMem
    .data(input.paramsPerWord * input.dtype.bitwidth - 1, 0)
    .asTypeOf(Vec(input.paramsPerWord, UInt(input.dtype.bitwidth.W)))
  io.nextElement.bits := actMemAsVec(memLineCntValue)
  io.nextElement.valid := (addrCntValue === RegNext(
    addrCntValue
  ) && addrCntValue <= io.actMemWrittenTo) && state === IDMState.sMOVEDATA

  io.done := state === IDMState.sWAIT && RegNext(state === IDMState.sMOVEDATA)
}

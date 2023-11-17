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

class InputDataMover[I <: Bits](input: lbir.QTensor) extends Module {
  object IDMState extends ChiselEnum {
    val sWAIT = Value(0.U)
    val sMOVEDATA = Value(1.U)
  }

  val io = IO(new Bundle {
    val nextElement = Decoupled(input.getType[I])
    val actMem = Flipped(new SRAMRead(input.memDepth, MemWordSize.bits))
    val actMemWrittenTo = Input(UInt(input.memDepth.W))
    val start = Input(Bool())
    val done = Output(Bool())
  })

  val (elementCounter, elementCounterWrap) = Counter(0 until input.numParams, io.nextElement.fire)
  val (wordSelectCounter, wordSelectCounterWrap) = Counter(0 until input.paramsPerWord, io.nextElement.fire, io.done)
  val (addressCounter, _) = Counter(0 until input.memDepth, wordSelectCounterWrap || io.done)

  val state = RegInit(IDMState.sWAIT)
  when(io.start) {
    state := IDMState.sMOVEDATA
  }.elsewhen(elementCounterWrap) {
    state := IDMState.sWAIT
  }

  io.actMem.address := addressCounter
  io.actMem.enable := state === IDMState.sMOVEDATA || io.start

  val actMemAsVec = io.actMem
    .data(input.paramsPerWord * input.dtype.bitwidth - 1, 0)
    .asTypeOf(Vec(input.paramsPerWord, input.getType[I]))
  io.nextElement.bits := actMemAsVec(wordSelectCounter)
  io.nextElement.valid := (addressCounter === RegNext(
    addressCounter
  ) && addressCounter <= io.actMemWrittenTo) && state === IDMState.sMOVEDATA

  io.done := state === IDMState.sWAIT && RegNext(state === IDMState.sMOVEDATA)
}

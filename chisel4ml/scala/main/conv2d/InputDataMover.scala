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
import chisel4ml.util.isStable
import org.chipsalliance.cde.config.Parameters
import chisel4ml.HasLBIRStreamParameters

/*
  Moves the entire tensor after obtainint the signal io.start.
  (this includes several channels)
 */
class InputDataMover[I <: Bits](implicit val p: Parameters) extends Module
with HasSequentialConvParameters
with HasLBIRStreamParameters {
  object IDMState extends ChiselEnum {
    val sWAIT = Value(0.U)
    val sMOVEDATA = Value(1.U)
  }

  val io = IO(new Bundle {
    val nextElement = Decoupled(cfg.input.getType[I])
    val actMem = Flipped(new SRAMRead(cfg.input.memDepth(), inWidth))
    val actMemWrittenTo = Input(UInt(log2Up(cfg.input.memDepth() + 1).W))
    val start = Input(Bool())
  })
  val (elementCounter, elementCounterWrap) = Counter(0 until cfg.input.numParams, io.nextElement.fire)
  val (wordSelectCounter, wordSelectCounterWrap) = Counter(0 until cfg.input.paramsPerWord(), io.nextElement.fire, io.start)
  val (addressCounter, _) = Counter(0 until cfg.input.memDepth(), wordSelectCounterWrap, io.start)
  dontTouch(elementCounterWrap)
  val state = RegInit(IDMState.sWAIT)
  when(io.start) {
    state := IDMState.sMOVEDATA
  }.elsewhen(elementCounterWrap) {
    state := IDMState.sWAIT
  }

  io.actMem.address := addressCounter
  io.actMem.enable := state === IDMState.sMOVEDATA || io.start

  val validBits = cfg.input.paramsPerWord() * cfg.input.dtype.bitwidth
  val actMemAsVec = io.actMem.data(validBits - 1, 0).asTypeOf(Vec(cfg.input.paramsPerWord(), cfg.input.getType[I]))
  io.nextElement.bits := actMemAsVec(wordSelectCounter)
  io.nextElement.valid := isStable(
    addressCounter
  ) && addressCounter < io.actMemWrittenTo && state === IDMState.sMOVEDATA

  when(state === IDMState.sMOVEDATA) {
    assert(io.start === false.B)
  }
}

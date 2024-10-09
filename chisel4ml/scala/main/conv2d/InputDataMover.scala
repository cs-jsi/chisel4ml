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
import chisel4ml.HasLBIRStreamParameters
import chisel4ml.implicits._
import chisel4ml.util.isStable
import lbir.Conv2DConfig
import memories.SRAMRead
import org.chipsalliance.cde.config.Parameters

/*
  Moves the entire tensor after obtainint the signal io.start.
  (this includes several channels)
 */
class InputDataMover[I <: Bits](implicit val p: Parameters)
    extends Module
    with HasSequentialConvParameters
    with HasLBIRStreamParameters[Conv2DConfig] {
  object IDMState extends ChiselEnum {
    val sWAIT = Value(0.U)
    val sMOVEDATA = Value(1.U)
  }

  val io = IO(new Bundle {
    val nextElement = Decoupled(cfg.input.getType[I])
    val actMem = Flipped(new SRAMRead(cfg.input.memDepth(inWidth), inWidth))
    val actMemWrittenTo = Input(UInt(log2Up(cfg.input.memDepth(inWidth) + 1).W))
    val start = Input(Bool())
  })
  val (elementCounter, elementCounterWrap) = Counter(0 until cfg.input.numParams, io.nextElement.fire)
  val state = RegInit(IDMState.sWAIT)
  val trueStart = io.start && (state === IDMState.sWAIT || elementCounterWrap)
  val (wordSelectCounter, wordSelectCounterWrap) = Counter(0 until numBeatsIn, io.nextElement.fire, trueStart)
  val (addressCounter, _) = Counter(0 until cfg.input.memDepth(inWidth), wordSelectCounterWrap, trueStart)
  dontTouch(elementCounterWrap)
  when(trueStart) {
    state := IDMState.sMOVEDATA
  }.elsewhen(elementCounterWrap) {
    state := IDMState.sWAIT
  }

  io.actMem.address := addressCounter
  io.actMem.enable := state === IDMState.sMOVEDATA || io.start

  val actMemAsVec = io.actMem.data.asTypeOf(Vec(numBeatsIn, cfg.input.getType[I]))
  io.nextElement.bits := actMemAsVec(wordSelectCounter)
  io.nextElement.valid := isStable(
    addressCounter
  ) && addressCounter < io.actMemWrittenTo && state === IDMState.sMOVEDATA
}

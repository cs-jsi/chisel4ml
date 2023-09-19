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

import _root_.chisel3._
import _root_.chisel4ml.lbir._
import _root_.chisel4ml.util.saturate

class DynamicNeuron[I <: Bits with Num[I], W <: Bits with Num[W], M <: Bits, S <: Bits, A <: Bits, O <: Bits](
  genIn:      I,
  numSynaps:  Int,
  genWeights: W,
  genAccu:    S,
  genThresh:  A,
  genOut:     O,
  mul:        (I, W) => M,
  add:        Vec[M] => S,
  actFn:      (S, A) => S)
    extends Module {

  def shiftAndRound(pAct: S, shift: UInt, shiftLeft: Bool, genAccu: S): S = {
    val sout = Wire(genAccu)
    when(shiftLeft) {
      sout := (pAct << shift).asUInt.asTypeOf(sout)
    }.otherwise {
      sout := ((pAct >> shift).asSInt + pAct(shift - 1.U).asSInt).asUInt.asTypeOf(sout)
    }
    sout
  }

  val io = IO(new Bundle {
    val in:        UInt = Input(UInt((numSynaps * genIn.getWidth).W))
    val weights:   UInt = Input(UInt((numSynaps * genWeights.getWidth).W))
    val thresh:    A = Input(genThresh)
    val shift:     UInt = Input(UInt(8.W)) // TODO: bitwidth?
    val shiftLeft: Bool = Input(Bool())
    val out:       O = Output(genOut)
  })

  val inVec = io.in.asTypeOf(Vec(numSynaps, genIn))
  val inWeights = io.weights.asTypeOf(Vec(numSynaps, genWeights))

  val muls = VecInit((inVec.zip(inWeights)).map { case (a, b) => mul(a, b) })
  val pAct = add(muls)
  val sAct = shiftAndRound(pAct, io.shift, io.shiftLeft, genAccu)
  io.out := saturate(actFn(sAct, io.thresh).asUInt, genOut.getWidth).asTypeOf(io.out)
}

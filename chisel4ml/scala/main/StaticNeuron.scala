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
package chisel4ml.combinational

import chisel3._
import chisel4ml.util.shiftAndRound

object StaticNeuron {
  def apply[I <: Bits, W <: Bits, M <: Bits, A <: Bits, O <: Bits](
    in:      Seq[I],
    weights: Seq[W],
    thresh:  A,
    mul:     (I, W) => M,
    add:     Vec[M] => A,
    actFn:   (A, A) => O,
    shift:   Int
  ): O = {
    val muls = VecInit((in.zip(weights)).map { case (a, b) => mul(a, b) })
    val pAct = add(muls)
    val sAct = shiftAndRound(pAct, shift)
    actFn(sAct, thresh)
  }
}

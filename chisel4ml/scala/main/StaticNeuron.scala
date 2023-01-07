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

import _root_.chisel3._
import _root_.chisel3.util._
import _root_.chisel4ml.lbir._

object StaticNeuron {
  def apply[I <: Bits, W <: Bits: WeightsProvider, M <: Bits, A <: Bits: ThreshProvider, O <: Bits](
      in:      Seq[I],
      weights: Seq[W],
      thresh:  A,
      mul:     (I, W) => M,
      add:     Vec[M] => A,
      actFn:   (A, A) => O,
      shift:   Int,
    ): O = {
    val muls = VecInit((in.zip(weights)).map { case (a, b) => mul(a, b) })
    val pAct = add(muls)
    val sAct = shiftAndRound(pAct, shift)
    actFn(sAct, thresh)
  }

  def shiftAndRound[A <: Bits: ThreshProvider](pAct: A, shift: Int): A = shift.compare(0) match {
    case 0 => pAct
    case -1 => {
      // Handles the case when the scale factor (shift) basically sets the output to zero always.
      if (-shift >= pAct.getWidth) {
        0.U.asTypeOf(pAct)
      } else {
        // We add the "cutt-off" bit to round the same way a convential rounding is done (1 >= 0.5, 0 < 0.5)
        ((pAct >> shift.abs).asSInt + Cat(0.S((pAct.getWidth - 1).W), pAct(shift.abs - 1)).asSInt)
          .asTypeOf(pAct)
      }
    }
    case 1 => (pAct << shift.abs).asTypeOf(pAct)
  }
}

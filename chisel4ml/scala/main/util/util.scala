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
package chisel4ml

import _root_.lbir._
import _root_.org.slf4j.LoggerFactory
import _root_.scala.math.{log, pow}
import chisel3._
import chisel3.util.Cat

package object util {
  val logger = LoggerFactory.getLogger("chisel4ml.util.")

  def log2(x: Int):   Int = (log(x.toFloat) / log(2.0)).toInt
  def log2(x: Float): Float = (log(x) / log(2.0)).toFloat

  def toBinary(i: Int, digits: Int = 8): String =
    String.format(s"%${digits}s", i.toBinaryString.takeRight(digits)).replace(' ', '0')
  def toBinaryB(i: BigInt, digits: Int = 8): String = String.format("%" + digits + "s", i.toString(2)).replace(' ', '0')

  def signedCorrect(x: Float, dtype: Datatype): Float = {
    if (dtype.signed && x > (pow(2, dtype.bitwidth - 1) - 1))
      x - pow(2, dtype.bitwidth).toFloat
    else
      x
  }

  def signFnU(act:     UInt, thresh: UInt, bw: Int = 1): Bool = act >= thresh
  def signFnS(act:     SInt, thresh: SInt, bw: Int = 1): Bool = act >= thresh
  def reluFnNoSat(act: SInt, thresh: SInt): UInt = Mux((act - thresh) > 0.S, (act - thresh).asUInt, 0.U)
  def reluFn(act:      SInt, thresh: SInt, bw: Int): UInt = saturateFnU(reluFnNoSat(act, thresh), bw)
  def linFn(act:       SInt, thresh: SInt): SInt = act - thresh

  def saturateFnU(x: UInt, bitwidth: Int): UInt = {
    val max = (pow(2, bitwidth) - 1).toInt.U
    Mux(x > max, max, x)
  }
  def saturateFnS(x: SInt, bitwidth: Int): SInt = {
    val max = (pow(2, bitwidth - 1) - 1).toInt.S
    val min = -pow(2, bitwidth - 1).toInt.S
    Mux(x > max, max, Mux(x < min, min, x))
  }

  def shiftAndRound[A <: Bits](pAct: A, shift: Int): A = shift.compare(0) match {
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

  def shiftAndRoundDynamic[S <: Bits](pAct: S, shift: UInt, shiftLeft: Bool, genAccu: S): S = {
    val sout = Wire(genAccu)
    when(shiftLeft) {
      sout := (pAct << shift).asUInt.asTypeOf(sout)
    }.otherwise {
      val shifted = (pAct >> shift).asSInt
      val carry = pAct(shift - 1.U).asUInt.zext
      sout := (shifted + carry).asUInt.asTypeOf(sout)
    }
    sout
  }
}

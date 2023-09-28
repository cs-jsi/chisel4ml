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

  def signFn(act:   UInt, thresh:   UInt): Bool = act >= thresh
  def signFn(act:   SInt, thresh:   SInt): Bool = act >= thresh
  def reluFn(act:   SInt, thresh:   SInt): UInt = Mux((act - thresh) > 0.S, (act - thresh).asUInt, 0.U)
  def linFn(act:    SInt, thresh:   SInt): SInt = act - thresh
  def noSaturate(x: Bool, bitwidth: Int): Bool = x
  def noSaturate(x: SInt, bitwidth: Int): SInt = Mux(
    x > (pow(2, bitwidth - 1) - 1).toInt.S,
    (pow(2, bitwidth - 1) - 1).toInt.S,
    Mux(x < -pow(2, bitwidth - 1).toInt.S, -pow(2, bitwidth - 1).toInt.S, x)
  )

  def saturate(x: UInt, bitwidth: Int): UInt =
    Mux(x > (pow(2, bitwidth) - 1).toInt.U, (pow(2, bitwidth) - 1).toInt.U, x) // TODO

  def mul(i: Bool, w: Bool): Bool = ~(i ^ w)
  def mul(i: UInt, w: Bool): SInt = Mux(w, i.zext, -(i.zext))
  def mul(i: UInt, w: SInt): SInt =
    if (w.litValue == 1.S.litValue) {
      i.zext
    } else if (w.litValue == -1.S.litValue) {
      -(i.zext)
    } else if (w.litValue == 0.S.litValue) {
      0.S
    } else {
      i * w
    }
  def mul(i: SInt, w: SInt): SInt = {
    if (w.litValue == 0.S.litValue) {
      0.S
    } else {
      i * w
    }
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
}

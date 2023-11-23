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

  def shiftAndRoundSInt(pAct: SInt, shift: UInt, shiftLeft: Bool): SInt = {
    val sout = Wire(SInt(pAct.getWidth.W))
    when(shiftLeft) {
      sout := (pAct << shift)
    }.otherwise {
      val shifted = (pAct >> shift).asSInt
      val sign = pAct(pAct.getWidth - 1)
      val nsign = !sign
      val fDec = pAct(shift.abs - 1.U) // first (most significant) decimal number
      val lDec = (pAct << (pAct.getWidth.U - (shift - 1.U)))(pAct.getWidth - 1, 0)
      val rest = VecInit(lDec.asBools).reduceTree(_ || _)
      val carry = (nsign && fDec) || (sign && fDec && rest)
      sout := (shifted + carry.asUInt.zext).asUInt.asTypeOf(sout)
    }
    sout
  }

  def shiftAndRoundSIntStatic(pAct: SInt, shift: Int): SInt = shift.compare(0) match {
    case 0 => pAct
    case 1 => pAct << shift
    case -1 => {
      val shifted = (pAct >> shift.abs).asSInt
      val sign = pAct(pAct.getWidth - 1)
      val nsign = !sign
      val fDec = pAct(shift.abs - 1) // first (most significnat) decimal number
      val rest = VecInit(pAct(shift.abs - 2, 0).asBools).reduceTree(_ || _)
      val carry = (nsign && fDec) || (sign && fDec && rest)
      shifted + carry.asUInt.zext
    }
  }

  def shiftAndRoundUInt(pAct: UInt, shift: UInt, shiftLeft: Bool): UInt = {
    val sout = Wire(UInt(pAct.getWidth.W))
    when(shiftLeft) {
      sout := (pAct << shift)
    }.otherwise {
      val shifted = (pAct >> shift)
      val carry = pAct(shift - 1.U).asUInt
      sout := shifted + carry
    }
    sout
  }

  def shiftAndRoundUIntStatic(pAct: UInt, shift: Int): UInt = shift.compare(0) match {
    case 0 => pAct
    case 1 =>
      pAct << shift
      val shifted = (pAct >> shift.abs).asUInt
      val sign = pAct(pAct.getWidth - 1)
      val nsign = !sign
      val fDec = pAct(shift.abs - 1) // first (most significnat) decimal number
      val rest = VecInit(pAct(shift.abs - 2, 0).asBools).reduceTree(_ || _)
      val carry = (nsign && fDec) || (sign && fDec && rest)
      shifted + carry.asUInt
  }

  def risingEdge(x: Bool) = x && !RegNext(x)

  def isStable[T <: Data](x: T): Bool = x === RegNext(x)
}

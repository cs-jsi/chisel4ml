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

  def toBinary(i: Int, digits: Int = 8): String = {
    val replacement = if (i >= 0) '0' else '1'
    String.format(s"%${digits}s", i.toBinaryString.takeRight(digits)).replace(' ', replacement)
  }
  def toBinaryB(i: BigInt, digits: Int = 8): String = {
    val replacement = if (i >= 0) '0' else '1'
    String.format("%" + digits + "s", i.toString(2)).replace(' ', replacement)
  }
  def signedCorrect(x: Float, dtype: Datatype): Float = {
    if (dtype.signed && x > (pow(2, dtype.bitwidth - 1) - 1))
      x - pow(2, dtype.bitwidth).toFloat
    else
      x
  }

  def saturateFnU(x: UInt, bitwidth: Int): UInt = {
    val max = (pow(2, bitwidth) - 1).toInt.U
    Mux(x > max, max, x)
  }
  def saturateFnS(x: SInt, bitwidth: Int): SInt = {
    val max = (pow(2, bitwidth - 1) - 1).toInt.S
    val min = -pow(2, bitwidth - 1).toInt.S
    Mux(x > max, max, Mux(x < min, min, x))
  }

  def shiftAndRoundSIntDynamic(roundingMode: String): (SInt, UInt, Bool) => SInt = roundingMode match {
    case "UP"        => shiftAndRoundSIntUp
    case "HALF_EVEN" => shiftAndRoundSIntDynamicHalfToEven
    case "ROUND"     => shiftAndRoundSIntDynamicHalfToEven
    case "NONE"      => (x: SInt, _: UInt, _: Bool) => x
    case _           => throw new NotImplementedError
  }

  def shiftAndRoundSIntUp(pAct: SInt, shift: UInt, shiftLeft: Bool): SInt = {
    val sout = Wire(SInt(pAct.getWidth.W))
    when(shiftLeft) {
      sout := (pAct << shift)
    }.otherwise {
      assert(pAct.getWidth.U > shift)
      assert(shift > 1.U)
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
  def shiftAndRoundSIntDynamicHalfToEven(pAct: SInt, shift: UInt, shiftLeft: Bool): SInt = {
    val sout = Wire(SInt(pAct.getWidth.W))
    when(shiftLeft) {
      sout := (pAct << shift)
    }.otherwise {
      assert(pAct.getWidth.U > shift)
      assert(shift > 1.U, f"$shift lower then 1")
      val shifted = (pAct >> shift).asSInt
      val sign = pAct(pAct.getWidth - 1)
      val nsign = !sign
      val fInt = pAct(shift)
      val fDec = pAct(shift - 1.U)
      val lDec = (pAct << (pAct.getWidth.U - (shift - 1.U)))(pAct.getWidth - 1, 0)
      val rest = VecInit(lDec.asBools).reduceTree(_ || _)
      val carry = (nsign && fInt && fDec) || (fDec && rest) || (sign && fInt && fDec)
      sout := shifted + carry.asUInt.zext
    }
    sout
  }

  def shiftAndRoundSIntStatic(roundingMode: String): (SInt, Int) => SInt = roundingMode match {
    case "UP"        => shiftAndRoundSIntStaticUp
    case "HALF_EVEN" => shiftAndRoundSIntStaticHalfToEven
    case "ROUND"     => shiftAndRoundSIntStaticHalfToEven
    case "NONE"      => (x: SInt, _: Int) => x
    case _           => throw new NotImplementedError
  }

  def shiftAndRoundSIntStaticUp(pAct: SInt, shift: Int): SInt = shift.compare(0) match {
    case 0 => pAct
    case 1 => pAct << shift
    case -1 =>
      if (pAct.getWidth > shift.abs) {
        val shifted = (pAct >> shift.abs).asSInt
        val sign = pAct(pAct.getWidth - 1)
        val nsign = !sign
        val fDec = pAct(shift.abs - 1) // first (most significnat) decimal number
        val rest = if (shift.abs > 1) VecInit(pAct(shift.abs - 2, 0).asBools).reduceTree(_ || _) else false.B
        val carry = (nsign && fDec) || (sign && fDec && rest)
        shifted + carry.asUInt.zext
      } else {
        0.S
      }
  }

  def shiftAndRoundSIntStaticHalfToEven(pAct: SInt, shift: Int): SInt = shift.compare(0) match {
    /*   +5.5 = 0101,1000 = 0110 = +6  | c=+1
     *   -5.5 = 1010,1000 = 1010 = -6  | c=0
     *
     *  +2.75 = 0010,1100 = 0011 = +3  | c=1
     *  -2.75 = 1101,0100 = 1101 = -3  | c=0
     *
     *   +2.5 = 0010,1000 = 0010 = +2  | c=0
     *   -2.5 = 1101,1000 = 1110 = -2  | c=+1
     *
     *  +1.25 = 0001,0100 = 0001 =  +1 | c=0
     *  -1.25 = 1110,1100 = 1111 =  -1 | c=1
     *
     *  +1.75 = 0001,1100 = 0010 = +2 | c=+1
     *  -1.75 = 1110,0100 = 1110 = -2 | c=0
     *
     */
    case 0 => pAct
    case 1 => pAct << shift
    case -1 =>
      if (pAct.getWidth > shift.abs) {
        val shifted = (pAct >> shift.abs).asSInt
        val sign = pAct(pAct.getWidth - 1)
        val nsign = !sign
        val fDec = pAct(shift.abs - 1)
        val fInt = pAct(shift.abs)
        val rest = if (shift.abs > 1) VecInit(pAct(shift.abs - 2, 0).asBools).reduceTree(_ || _) else false.B
        val carry = (nsign && fInt && fDec) || (fDec && rest) || (sign && fInt && fDec)
        shifted + carry.asUInt.zext
      } else {
        0.S
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
    case 1 => pAct << shift
    case -1 =>
      if (pAct.getWidth > shift) {
        val shifted = (pAct >> shift.abs).asUInt
        val sign = pAct(pAct.getWidth - 1)
        val nsign = !sign
        val fDec = pAct(shift.abs - 1) // first (most significnat) decimal number
        val rest = if (shift > 1) VecInit(pAct(shift.abs - 2, 0).asBools).reduceTree(_ || _) else true.B
        val carry = (nsign && fDec) || (sign && fDec && rest)
        shifted + carry.asUInt
      } else {
        0.U
      }
  }

  def risingEdge(x: Bool) = x && !RegNext(x)

  def isStable[T <: Data](x: T): Bool = x === RegNext(x)
}

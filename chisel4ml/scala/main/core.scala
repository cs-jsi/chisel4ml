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

import chisel3._
import chisel3.util._
import chisel4ml.util._

trait QuantizationContext[I <: Bits, W <: Bits, M <: Bits, A <: Bits, O <: Bits] extends Any {
  def mul:                  (I, W) => M
  def add:                  Vec[M] => A
  def actFn:                (A, A, Int) => O
  def shiftAndRoundStatic:  (A, Int) => A
  def shiftAndRoundDynamic: (A, UInt, Bool) => A
}

object BinarizedQuantizationContext extends QuantizationContext[Bool, Bool, Bool, UInt, Bool] {
  def mul = (i: Bool, w: Bool) => ~(i ^ w)
  def add = (x: Vec[Bool]) => PopCount(x.asUInt)
  def shiftAndRoundStatic:  (UInt, Int) => UInt = shiftAndRoundUIntStatic
  def shiftAndRoundDynamic: (UInt, UInt, Bool) => UInt = shiftAndRoundUInt
  def actFn:                (UInt, UInt, Int) => Bool = signFnU
}

class BinaryQuantizationContext(roundingMode: lbir.RoundingMode)
    extends QuantizationContext[UInt, Bool, SInt, SInt, Bool] {
  def mul = (i: UInt, w: Bool) => Mux(w, i.zext, -(i.zext))
  def add = (x: Vec[SInt]) => x.reduceTree(_ +& _)
  def shiftAndRoundStatic:  (SInt, Int) => SInt = shiftAndRoundSIntStatic(roundingMode)
  def shiftAndRoundDynamic: (SInt, UInt, Bool) => SInt = shiftAndRoundSInt
  def actFn = signFnS
}

class BinaryQuantizationContextSInt(roundingMode: lbir.RoundingMode)
    extends QuantizationContext[SInt, Bool, SInt, SInt, Bool] {
  def mul = (i: SInt, w: Bool) => Mux(w, i, -i)
  def add = (x: Vec[SInt]) => x.reduceTree(_ +& _)
  def shiftAndRoundStatic:  (SInt, Int) => SInt = shiftAndRoundSIntStatic(roundingMode)
  def shiftAndRoundDynamic: (SInt, UInt, Bool) => SInt = shiftAndRoundSInt
  def actFn = signFnS
}

// implementiraj s dsptools?
class UniformQuantizationContextSSU(act: (SInt, SInt, Int) => UInt, roundingMode: lbir.RoundingMode)
    extends QuantizationContext[SInt, SInt, SInt, SInt, UInt] {
  def mul = (i: SInt, w: SInt) => i * w
  def add = (x: Vec[SInt]) => x.reduceTree(_ +& _)
  def shiftAndRoundStatic:  (SInt, Int) => SInt = shiftAndRoundSIntStatic(roundingMode)
  def shiftAndRoundDynamic: (SInt, UInt, Bool) => SInt = shiftAndRoundSInt
  def actFn = act
}

class UniformQuantizationContextSSUReLU(roundingMode: lbir.RoundingMode)
    extends UniformQuantizationContextSSU(reluFn, roundingMode)

class UniformQuantizationComputeUSU(act: (SInt, SInt, Int) => UInt, roundingMode: lbir.RoundingMode)
    extends QuantizationContext[UInt, SInt, SInt, SInt, UInt] {
  def mul = (i: UInt, w: SInt) => i * w
  def add = (x: Vec[SInt]) => x.reduceTree(_ +& _)
  def shiftAndRoundStatic:  (SInt, Int) => SInt = shiftAndRoundSIntStatic(roundingMode)
  def shiftAndRoundDynamic: (SInt, UInt, Bool) => SInt = shiftAndRoundSInt
  def actFn = act
}

class UniformQuantizationContextUSUReLU(roundingMode: lbir.RoundingMode)
    extends UniformQuantizationComputeUSU(reluFn, roundingMode)

class UniformQuantizationContextSSS(act: (SInt, SInt, Int) => SInt, roundingMode: lbir.RoundingMode)
    extends QuantizationContext[SInt, SInt, SInt, SInt, SInt] {
  def mul = (i: SInt, w: SInt) => i * w
  def add = (x: Vec[SInt]) => x.reduceTree(_ +& _)
  def shiftAndRoundStatic:  (SInt, Int) => SInt = shiftAndRoundSIntStatic(roundingMode)
  def shiftAndRoundDynamic: (SInt, UInt, Bool) => SInt = shiftAndRoundSInt
  def actFn = act
}

class UniformQuantizationContextSSSNoAct(roundingMode: lbir.RoundingMode)
    extends UniformQuantizationContextSSS((i: SInt, t: SInt, bw: Int) => saturateFnS(i - t, bw), roundingMode)

class UniformQuantizationContextUSS(act: (SInt, SInt, Int) => SInt, roundingMode: lbir.RoundingMode)
    extends QuantizationContext[UInt, SInt, SInt, SInt, SInt] {
  def mul = (i: UInt, w: SInt) => i * w
  def add = (x: Vec[SInt]) => x.reduceTree(_ +& _)
  def shiftAndRoundStatic:  (SInt, Int) => SInt = shiftAndRoundSIntStatic(roundingMode)
  def shiftAndRoundDynamic: (SInt, UInt, Bool) => SInt = shiftAndRoundSInt
  def actFn = act
}

class UniformQuantizationContextUSSNoAct(roundingMode: lbir.RoundingMode)
    extends UniformQuantizationContextUSS((i: SInt, t: SInt, bw: Int) => saturateFnS(i - t, bw), roundingMode)

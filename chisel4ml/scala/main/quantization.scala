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

package quantization {
  trait QuantizationContext[I <: Bits, W <: Bits, M <: Bits, A <: Bits, O <: Bits] extends Any {
    def mul:                  (I, W) => M
    def add:                  Vec[M] => A
    def actFn:                (A, A, Int) => O
    def shiftAndRoundStatic:  (A, Int) => A
    def shiftAndRoundDynamic: (A, UInt, Bool) => A
  }

  object BinarizedQuantizationContext extends QuantizationContext[Bool, Bool, Bool, UInt, Bool] {
    override def mul = (i: Bool, w: Bool) => ~(i ^ w)
    override def add = (x: Vec[Bool]) => PopCount(x.asUInt)
    override def shiftAndRoundStatic:  (UInt, Int) => UInt = shiftAndRoundUIntStatic
    override def shiftAndRoundDynamic: (UInt, UInt, Bool) => UInt = shiftAndRoundUInt
    override def actFn:                (UInt, UInt, Int) => Bool = signFnU
  }

  class BinaryQuantizationContext(roundingMode: String) extends QuantizationContext[UInt, Bool, SInt, SInt, Bool] {
    override def mul = (i: UInt, w: Bool) => Mux(w, i.zext, -(i.zext))
    override def add = (x: Vec[SInt]) => x.reduceTree(_ +& _)
    override def shiftAndRoundStatic:  (SInt, Int) => SInt = shiftAndRoundSIntStatic(roundingMode)
    override def shiftAndRoundDynamic: (SInt, UInt, Bool) => SInt = shiftAndRoundSIntDynamic(roundingMode)
    override def actFn = signFnS
  }

  class BinaryQuantizationContextSInt(roundingMode: String) extends QuantizationContext[SInt, Bool, SInt, SInt, Bool] {
    override def mul = (i: SInt, w: Bool) => Mux(w, i, -i)
    override def add = (x: Vec[SInt]) => x.reduceTree(_ +& _)
    override def shiftAndRoundStatic:  (SInt, Int) => SInt = shiftAndRoundSIntStatic(roundingMode)
    override def shiftAndRoundDynamic: (SInt, UInt, Bool) => SInt = shiftAndRoundSIntDynamic(roundingMode)
    override def actFn = signFnS
  }

  // implementiraj s dsptools?
  class UniformQuantizationContextSSU(act: (SInt, SInt, Int) => UInt, roundingMode: String)
      extends QuantizationContext[SInt, SInt, SInt, SInt, UInt] {
    override def mul = (i: SInt, w: SInt) => i * w
    override def add = (x: Vec[SInt]) => x.reduceTree(_ +& _)
    override def shiftAndRoundStatic:  (SInt, Int) => SInt = shiftAndRoundSIntStatic(roundingMode)
    override def shiftAndRoundDynamic: (SInt, UInt, Bool) => SInt = shiftAndRoundSIntDynamic(roundingMode)
    override def actFn = act
  }

  class UniformQuantizationContextSSUReLU(roundingMode: String)
      extends UniformQuantizationContextSSU(reluFn, roundingMode)

  class UniformQuantizationComputeUSU(act: (SInt, SInt, Int) => UInt, roundingMode: String)
      extends QuantizationContext[UInt, SInt, SInt, SInt, UInt] {
    override def mul = (i: UInt, w: SInt) => i * w
    override def add = (x: Vec[SInt]) => x.reduceTree(_ +& _)
    override def shiftAndRoundStatic:  (SInt, Int) => SInt = shiftAndRoundSIntStatic(roundingMode)
    override def shiftAndRoundDynamic: (SInt, UInt, Bool) => SInt = shiftAndRoundSIntDynamic(roundingMode)
    override def actFn = act
  }

  class UniformQuantizationContextUSUReLU(roundingMode: String)
      extends UniformQuantizationComputeUSU(reluFn, roundingMode)

  class UniformQuantizationContextSSS(act: (SInt, SInt, Int) => SInt, roundingMode: String)
      extends QuantizationContext[SInt, SInt, SInt, SInt, SInt] {
    override def mul = (i: SInt, w: SInt) => i * w
    override def add = (x: Vec[SInt]) => x.reduceTree(_ +& _)
    override def shiftAndRoundStatic:  (SInt, Int) => SInt = shiftAndRoundSIntStatic(roundingMode)
    override def shiftAndRoundDynamic: (SInt, UInt, Bool) => SInt = shiftAndRoundSIntDynamic(roundingMode)
    override def actFn = act
  }

  class UniformQuantizationContextSSSNoAct(roundingMode: String)
      extends UniformQuantizationContextSSS((i: SInt, t: SInt, bw: Int) => saturateFnS(i - t, bw), roundingMode)

  class UniformQuantizationContextUSS(act: (SInt, SInt, Int) => SInt, roundingMode: String)
      extends QuantizationContext[UInt, SInt, SInt, SInt, SInt] {
    override def mul = (i: UInt, w: SInt) => i * w
    override def add = (x: Vec[SInt]) => x.reduceTree(_ +& _)
    override def shiftAndRoundStatic:  (SInt, Int) => SInt = shiftAndRoundSIntStatic(roundingMode)
    override def shiftAndRoundDynamic: (SInt, UInt, Bool) => SInt = shiftAndRoundSIntDynamic(roundingMode)
    override def actFn = act
  }

  class UniformQuantizationContextUSSNoAct(roundingMode: String)
      extends UniformQuantizationContextUSS((i: SInt, t: SInt, bw: Int) => saturateFnS(i - t, bw), roundingMode)

}

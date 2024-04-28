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
import spire.algebra.Ring
import spire.implicits._
import dsptools.numbers._

package quantization {
  trait QuantizationContext[I <: Bits, W <: Bits, M <: Bits, A <: Bits, O <: Bits] extends Any {
    def mul:           (I, W) => M
    def add:           Vec[M] => A
    def actFn:         (A, A, Int) => O
    def shiftAndRound: (A, UInt, Bool, lbir.RoundingMode) => A
    def genI(bitwidth: Int): I
    def genW(bitwidth: Int): W
    def genM(bitwidth: Int): M
    def genA(bitwidth: Int): A
    def genO(bitwidth: Int): O
    def ringA: Ring[A]
  }

  object BinarizedQuantizationContext extends QuantizationContext[Bool, Bool, Bool, UInt, Bool] {
    override def mul = (i: Bool, w: Bool) => ~(i ^ w)
    override def add = (x: Vec[Bool]) => PopCount(x.asUInt)
    override def shiftAndRound: (UInt, UInt, Bool, lbir.RoundingMode) => UInt = shiftAndRoundUInt
    override def actFn:         (UInt, UInt, Int) => Bool = signFnU
    override def genI(bitwidth: Int) = Bool()
    override def genW(bitwidth: Int) = Bool()
    override def genM(bitwidth: Int) = Bool()
    override def genA(bitwidth: Int) = UInt(bitwidth.W)
    override def genO(bitwidth: Int) = Bool()

    override def ringA = implicitly[Ring[UInt]]
  }

  trait HasRoundingMode {
    val roundingMode: lbir.RoundingMode
  }

  object BinaryQuantizationContextUInt extends QuantizationContext[UInt, Bool, SInt, SInt, Bool] {
    override def mul = (i: UInt, w: Bool) => Mux(w, i.zext, -(i.zext))
    override def add = (x: Vec[SInt]) => x.reduceTree(_ +& _)
    override def shiftAndRound: (SInt, UInt, Bool, lbir.RoundingMode) => SInt = shiftAndRoundSInt
    override def actFn = signFnS
    override def genI(bitwidth: Int) = UInt(bitwidth.W)
    override def genW(bitwidth: Int) = Bool()
    override def genM(bitwidth: Int) = SInt(bitwidth.W)
    override def genA(bitwidth: Int) = SInt(bitwidth.W)
    override def genO(bitwidth: Int) = Bool()

    override def ringA = implicitly[Ring[SInt]]
  }

  object BinaryQuantizationContextSInt extends QuantizationContext[SInt, Bool, SInt, SInt, Bool] {
    override def mul = (i: SInt, w: Bool) => Mux(w, i, -i)
    override def add = (x: Vec[SInt]) => x.reduceTree(_ +& _)
    override def shiftAndRound: (SInt, UInt, Bool, lbir.RoundingMode) => SInt = shiftAndRoundSInt
    override def actFn = signFnS
    override def genI(bitwidth: Int) = SInt(bitwidth.W)
    override def genW(bitwidth: Int) = Bool()
    override def genM(bitwidth: Int) = SInt(bitwidth.W)
    override def genA(bitwidth: Int) = SInt(bitwidth.W)
    override def genO(bitwidth: Int) = Bool()

    override def ringA = implicitly[Ring[SInt]]
  }

  // implementiraj s dsptools?
  object UniformQuantizationContextSSUReLU extends QuantizationContext[SInt, SInt, SInt, SInt, UInt] {
    override def mul = (i: SInt, w: SInt) => i * w
    override def add = (x: Vec[SInt]) => x.reduceTree(_ +& _)
    override def shiftAndRound: (SInt, UInt, Bool, lbir.RoundingMode) => SInt = shiftAndRoundSInt
    override def actFn = reluFn
    override def genI(bitwidth: Int) = SInt(bitwidth.W)
    override def genW(bitwidth: Int) = SInt(bitwidth.W)
    override def genM(bitwidth: Int) = SInt(bitwidth.W)
    override def genA(bitwidth: Int) = SInt(bitwidth.W)
    override def genO(bitwidth: Int) = UInt(bitwidth.W)

    override def ringA = implicitly[Ring[SInt]]
  }

  object UniformQuantizationContextUSUReLU extends QuantizationContext[UInt, SInt, SInt, SInt, UInt] {
    override def mul = (i: UInt, w: SInt) => i * w
    override def add = (x: Vec[SInt]) => x.reduceTree(_ +& _)
    override def shiftAndRound: (SInt, UInt, Bool, lbir.RoundingMode) => SInt = shiftAndRoundSInt
    override def actFn = reluFn
    override def genI(bitwidth: Int) = UInt(bitwidth.W)
    override def genW(bitwidth: Int) = SInt(bitwidth.W)
    override def genM(bitwidth: Int) = SInt(bitwidth.W)
    override def genA(bitwidth: Int) = SInt(bitwidth.W)
    override def genO(bitwidth: Int) = UInt(bitwidth.W)

    override def ringA = implicitly[Ring[SInt]]
  }

  object UniformQuantizationContextSSSNoAct extends QuantizationContext[SInt, SInt, SInt, SInt, SInt] {
    override def mul = (i: SInt, w: SInt) => i * w
    override def add = (x: Vec[SInt]) => x.reduceTree(_ +& _)
    override def shiftAndRound: (SInt, UInt, Bool, lbir.RoundingMode) => SInt = shiftAndRoundSInt
    override def actFn = (i: SInt, t: SInt, bw: Int) => saturateFnS(i - t, bw)
    override def genI(bitwidth: Int) = SInt(bitwidth.W)
    override def genW(bitwidth: Int) = SInt(bitwidth.W)
    override def genM(bitwidth: Int) = SInt(bitwidth.W)
    override def genA(bitwidth: Int) = SInt(bitwidth.W)
    override def genO(bitwidth: Int) = SInt(bitwidth.W)

    override def ringA = implicitly[Ring[SInt]]
  }

  object UniformQuantizationContextUSSNoAct extends QuantizationContext[UInt, SInt, SInt, SInt, SInt] {
    override def mul = (i: UInt, w: SInt) => i * w
    override def add = (x: Vec[SInt]) => x.reduceTree(_ +& _)
    override def shiftAndRound: (SInt, UInt, Bool, lbir.RoundingMode) => SInt = shiftAndRoundSInt
    override def actFn = (i: SInt, t: SInt, bw: Int) => saturateFnS(i - t, bw)
    override def genI(bitwidth: Int) = UInt(bitwidth.W)
    override def genW(bitwidth: Int) = SInt(bitwidth.W)
    override def genM(bitwidth: Int) = SInt(bitwidth.W)
    override def genA(bitwidth: Int) = SInt(bitwidth.W)
    override def genO(bitwidth: Int) = SInt(bitwidth.W)

    override def ringA = implicitly[Ring[SInt]]
  }

}

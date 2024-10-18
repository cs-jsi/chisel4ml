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
  abstract class IOContext {
    type I <: Bits
    type O <: Bits
  }

  object IOContextBB extends IOContext {
    type I = Bool
    type O = Bool
  }

  object IOContextUB extends IOContext {
    type I = UInt
    type O = Bool
  }

  object IOContextSB extends IOContext {
    type I = SInt
    type O = Bool
  }

  object IOContextSU extends IOContext {
    type I = SInt
    type O = UInt
  }

  object IOContextSS extends IOContext {
    type I = SInt
    type O = SInt
  }

  object IOContextUU extends IOContext {
    type I = UInt
    type O = UInt
  }

  object IOContextUS extends IOContext {
    type I = UInt
    type O = SInt
  }

  abstract class QuantizationContext {
    type W <: Bits
    type M <: Bits
    type A <: Bits
    val io: IOContext
    def mul:                  (io.I, W) => M
    def add:                  Vec[M] => A
    def addA:                 (A, A) => A
    def minA:                 (A, A) => A
    def zeroA:                A
    def actFn:                (A, A) => io.O
    def shiftAndRoundStatic:  (A, Int) => A
    def shiftAndRoundDynamic: (A, UInt, Bool) => A
    def gt:                   (io.I, io.I) => Bool
  }

  object BinarizedQuantizationContext extends QuantizationContext {
    type W = Bool
    type M = Bool
    type A = UInt
    override val io = IOContextBB
    override def mul = (i: Bool, w: Bool) => ~(i ^ w)
    override def add = (x: Vec[Bool]) => PopCount(x.asUInt)
    override def addA = (x: UInt, y: UInt) => x +& y
    override def minA = (x: UInt, y: UInt) => x -& y
    override def zeroA = 0.U
    override def shiftAndRoundStatic:  (UInt, Int) => UInt = shiftAndRoundUIntStatic
    override def shiftAndRoundDynamic: (UInt, UInt, Bool) => UInt = shiftAndRoundUInt
    override def actFn:                (UInt, UInt) => Bool = new SignFunction[UInt].actFn
    // i0 i1 out
    // 0  0   x
    // 0  1   0
    // 1  0   1
    // 1  1   x
    override def gt = (i0: Bool, _: Bool) => i0
  }

  class BinaryQuantizationContext(roundingMode: String) extends QuantizationContext {
    type W = Bool
    type M = SInt
    type A = SInt
    override val io = IOContextUB
    override def mul = (i: UInt, w: Bool) => Mux(w, i.zext, -(i.zext))
    override def add = (x: Vec[SInt]) => x.reduceTree(_ +& _)
    override def addA = (x: SInt, y: SInt) => x +& y
    override def minA = (x: SInt, y: SInt) => x -& y
    override def zeroA = 0.S
    override def shiftAndRoundStatic:  (SInt, Int) => SInt = shiftAndRoundSIntStatic(roundingMode)
    override def shiftAndRoundDynamic: (SInt, UInt, Bool) => SInt = shiftAndRoundSIntDynamic(roundingMode)
    override def actFn = new SignFunction[SInt].actFn
    override def gt = (i0: UInt, i1: UInt) => i0 > i1
  }

  class BinaryQuantizationContextSInt(roundingMode: String) extends QuantizationContext {
    type W = Bool
    type M = SInt
    type A = SInt
    val io = IOContextSB
    override def mul = (i: SInt, w: Bool) => Mux(w, i, -i)
    override def add = (x: Vec[SInt]) => x.reduceTree(_ +& _)
    override def addA = (x: SInt, y: SInt) => x +& y
    override def minA = (x: SInt, y: SInt) => x -& y
    override def zeroA = 0.S
    override def shiftAndRoundStatic:  (SInt, Int) => SInt = shiftAndRoundSIntStatic(roundingMode)
    override def shiftAndRoundDynamic: (SInt, UInt, Bool) => SInt = shiftAndRoundSIntDynamic(roundingMode)
    override def actFn = new SignFunction[SInt].actFn
    override def gt = (i0: SInt, i1: SInt) => i0 > i1
  }

  class UniformQuantizationContextSSU(act: lbir.Activation, outputBitwidth: Int, roundingMode: String)
      extends QuantizationContext {
    type W = SInt
    type M = SInt
    type A = SInt
    override val io = IOContextSU
    override def mul = (i: SInt, w: SInt) => i * w
    override def add = (x: Vec[SInt]) => x.reduceTree(_ +& _)
    override def addA = (x: SInt, y: SInt) => x +& y
    override def minA = (x: SInt, y: SInt) => x -& y
    override def zeroA = 0.S
    override def shiftAndRoundStatic:  (SInt, Int) => SInt = shiftAndRoundSIntStatic(roundingMode)
    override def shiftAndRoundDynamic: (SInt, UInt, Bool) => SInt = shiftAndRoundSIntDynamic(roundingMode)
    override def actFn = Utilities.activationToFunctionSU(act, outputBitwidth)
    override def gt = (i0: SInt, i1: SInt) => i0 > i1
  }

  class UniformQuantizationContextUSU(act: lbir.Activation, outputBitwidth: Int, roundingMode: String)
      extends QuantizationContext {
    type W = SInt
    type M = SInt
    type A = SInt
    override val io = IOContextUU
    override def mul = (i: UInt, w: SInt) => i * w
    override def add = (x: Vec[SInt]) => x.reduceTree(_ +& _)
    override def addA = (x: SInt, y: SInt) => x +& y
    override def minA = (x: SInt, y: SInt) => x -& y
    override def zeroA = 0.S
    override def shiftAndRoundStatic:  (SInt, Int) => SInt = shiftAndRoundSIntStatic(roundingMode)
    override def shiftAndRoundDynamic: (SInt, UInt, Bool) => SInt = shiftAndRoundSIntDynamic(roundingMode)
    override def actFn = Utilities.activationToFunctionSU(act, outputBitwidth)
    override def gt = (i0: UInt, i1: UInt) => i0 > i1
  }

  class UniformQuantizationContextSSS(act: lbir.Activation, outputBitwidth: Int, roundingMode: String)
      extends QuantizationContext {
    type W = SInt
    type M = SInt
    type A = SInt
    override val io = IOContextSS
    override def mul = (i: SInt, w: SInt) => i * w
    override def add = (x: Vec[SInt]) => x.reduceTree(_ +& _)
    override def addA = (x: SInt, y: SInt) => x +& y
    override def minA = (x: SInt, y: SInt) => x -& y
    override def zeroA = 0.S
    override def shiftAndRoundStatic:  (SInt, Int) => SInt = shiftAndRoundSIntStatic(roundingMode)
    override def shiftAndRoundDynamic: (SInt, UInt, Bool) => SInt = shiftAndRoundSIntDynamic(roundingMode)
    override def actFn = Utilities.activationToFunctionSS(act, outputBitwidth)
    override def gt = (i0: SInt, i1: SInt) => i0 > i1
  }

  class UniformQuantizationContextUSS(act: lbir.Activation, outputBitwidth: Int, roundingMode: String)
      extends QuantizationContext {
    type W = SInt
    type M = SInt
    type A = SInt
    val io = IOContextUS
    override def mul = (i: UInt, w: SInt) => i * w
    override def add = (x: Vec[SInt]) => x.reduceTree(_ +& _)
    override def addA = (x: SInt, y: SInt) => x +& y
    override def minA = (x: SInt, y: SInt) => x -& y
    override def zeroA = 0.S
    override def shiftAndRoundStatic:  (SInt, Int) => SInt = shiftAndRoundSIntStatic(roundingMode)
    override def shiftAndRoundDynamic: (SInt, UInt, Bool) => SInt = shiftAndRoundSIntDynamic(roundingMode)
    override def actFn = Utilities.activationToFunctionSS(act, outputBitwidth)
    override def gt = (i0: UInt, i1: UInt) => i0 > i1
  }

  // ACTIVATION FUNCTIONS
  trait ActivationFunction[A <: Bits, O <: Bits] {
    def actFn: (A, A) => O
  }

  class SignFunction[A <: Bits with Num[A]] extends ActivationFunction[A, Bool] {
    def actFn = (act: A, thresh: A) => act >= thresh
  }

  class ReluFunction(bitwidth: Int) extends ActivationFunction[SInt, UInt] {
    def reluNoSaturation(act: SInt, thresh: SInt): UInt = Mux((act - thresh) > 0.S, (act - thresh).asUInt, 0.U)
    def actFn = (act: SInt, thresh: SInt) => saturateFnU(reluNoSaturation(act, thresh), bitwidth)
  }

  class LinearFunction(bitwidth: Int) extends ActivationFunction[SInt, SInt] {
    def actFn = (act: SInt, thresh: SInt) => saturateFnS(act - thresh, bitwidth)
  }

  object Utilities {
    def activationToFunctionSU(act: lbir.Activation, bitwidth: Int): (SInt, SInt) => (UInt) = act match {
      case lbir.Activation.RELU => new ReluFunction(bitwidth).actFn
      case _                    => throw new RuntimeException
    }
    def activationToFunctionSS(act: lbir.Activation, bitwidth: Int): (SInt, SInt) => (SInt) = act match {
      case lbir.Activation.NO_ACTIVATION => new LinearFunction(bitwidth).actFn
      case _                             => throw new RuntimeException
    }
  }
}

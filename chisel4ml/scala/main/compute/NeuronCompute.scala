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
package chisel4ml.compute

import chisel3._
import chisel3.util._
import chisel4ml.util._
import dsptools.numbers._
import dsptools.numbers.implicits._
import lbir.Datatype.QuantizationType.{BINARY, UNIFORM}
import lbir.{IsActiveLayer, LayerWrap}

object NeuronCompute {
  def apply(l: LayerWrap with IsActiveLayer): NeuronCompute = (
    l.input.dtype.quantization,
    l.input.dtype.signed,
    l.kernel.dtype.quantization,
    l.kernel.dtype.signed,
    l.output.dtype.quantization,
    l.output.dtype.signed
  ) match {
    case (BINARY, _, BINARY, _, BINARY, _) => BinarizedNeuronCompute
    case (UNIFORM, false, BINARY, _, BINARY, _) =>
      new UnsignedBinaryNeuronCompute(l.output.roundingMode, l.input.dtype.bitwidth)
    case (UNIFORM, true, BINARY, _, BINARY, _) =>
      new SignedBinaryNeuronCompute(l.output.roundingMode, l.input.dtype.bitwidth)
    case (UNIFORM, true, UNIFORM, true, UNIFORM, false) =>
      new NeuronComputeSSU(l.activation, l.input.dtype.bitwidth, l.output.dtype.bitwidth, l.output.roundingMode)
    case (UNIFORM, false, UNIFORM, true, UNIFORM, false) =>
      new NeuronComputeUSU(l.activation, l.input.dtype.bitwidth, l.output.dtype.bitwidth, l.output.roundingMode)
    case (UNIFORM, true, UNIFORM, true, UNIFORM, true) =>
      new NeuronComputeSSS(l.activation, l.input.dtype.bitwidth, l.output.dtype.bitwidth, l.output.roundingMode)
    case (UNIFORM, false, UNIFORM, true, UNIFORM, true) =>
      new NeuronComputeUSS(l.activation, l.input.dtype.bitwidth, l.output.dtype.bitwidth, l.output.roundingMode)
    case _ =>
      throw new RuntimeException(
        f"Quantization type not supported: ${l.input.dtype.quantization}, ${l.input.dtype.signed}, ${l.kernel.dtype.quantization}, ${l.kernel.dtype.signed}, ${l.output.dtype.quantization}, ${l.output.dtype.signed}."
      )
  }
}

abstract class NeuronCompute {
  type I <: Bits
  type W <: Bits
  type M <: Bits
  type A <: Bits
  type O <: Bits
  def ringA:                Ring[A]
  def binA:                 BinaryRepresentation[A]
  def mul:                  (I, W) => M
  def add:                  Vec[M] => A
  def actFn:                (A, A) => O
  def shiftAndRoundStatic:  (A, Int) => A
  def shiftAndRoundDynamic: (A, UInt, Bool) => A
  def genI:                 I
  def genO:                 O
}

object BinarizedNeuronCompute extends NeuronCompute {
  type I = Bool
  type W = Bool
  type M = Bool
  type A = UInt

  type O = Bool
  override def ringA: Ring[A] = implicitly[Ring[UInt]]
  override def binA:  BinaryRepresentation[A] = implicitly[BinaryRepresentation[UInt]]
  override def mul = (i: Bool, w: Bool) => ~(i ^ w)
  override def add = (x: Vec[Bool]) => PopCount(x.asUInt)
  override def shiftAndRoundStatic:  (UInt, Int) => UInt = shiftAndRoundUIntStatic
  override def shiftAndRoundDynamic: (UInt, UInt, Bool) => UInt = shiftAndRoundUInt
  override def actFn:                (UInt, UInt) => Bool = new SignFunction[UInt].actFn

  override def genI = Bool()
  override def genO = Bool()
}

class UnsignedBinaryNeuronCompute(roundingMode: String, bitwidth: Int) extends NeuronCompute {
  type I = UInt
  type W = Bool
  type M = SInt
  type A = SInt
  type O = Bool
  override def ringA: Ring[A] = implicitly[Ring[SInt]]
  override def binA:  BinaryRepresentation[A] = implicitly[BinaryRepresentation[SInt]]
  override def mul = (i: UInt, w: Bool) => Mux(w, i.zext, -(i.zext))
  override def add = (x: Vec[SInt]) => x.reduceTree(_ +& _)
  override def shiftAndRoundStatic:  (SInt, Int) => SInt = shiftAndRoundSIntStatic(roundingMode)
  override def shiftAndRoundDynamic: (SInt, UInt, Bool) => SInt = shiftAndRoundSIntDynamic(roundingMode)
  override def actFn = new SignFunction[SInt].actFn

  override def genI = UInt(bitwidth.W)
  override def genO = Bool()
}

class SignedBinaryNeuronCompute(roundingMode: String, bitwidth: Int) extends NeuronCompute {
  type I = SInt
  type W = Bool
  type M = SInt
  type A = SInt
  type O = Bool
  override def ringA: Ring[A] = implicitly[Ring[SInt]]
  override def binA:  BinaryRepresentation[A] = implicitly[BinaryRepresentation[SInt]]
  override def mul = (i: SInt, w: Bool) => Mux(w, i, 0.S -& i)
  override def add = (x: Vec[SInt]) => x.reduceTree(_ +& _)
  override def shiftAndRoundStatic:  (SInt, Int) => SInt = shiftAndRoundSIntStatic(roundingMode)
  override def shiftAndRoundDynamic: (SInt, UInt, Bool) => SInt = shiftAndRoundSIntDynamic(roundingMode)
  override def actFn = new SignFunction[SInt].actFn

  override def genI = SInt(bitwidth.W)
  override def genO = Bool()
}

class NeuronComputeSSU(act: lbir.Activation, inputBitwidth: Int, outputBitwidth: Int, roundingMode: String)
    extends NeuronCompute {
  type I = SInt
  type W = SInt
  type M = SInt
  type A = SInt
  type O = UInt
  override def ringA: Ring[A] = implicitly[Ring[SInt]]
  override def binA:  BinaryRepresentation[A] = implicitly[BinaryRepresentation[SInt]]
  override def mul = (i: SInt, w: SInt) => i * w
  override def add = (x: Vec[SInt]) => x.reduceTree(_ +& _)
  override def shiftAndRoundStatic:  (SInt, Int) => SInt = shiftAndRoundSIntStatic(roundingMode)
  override def shiftAndRoundDynamic: (SInt, UInt, Bool) => SInt = shiftAndRoundSIntDynamic(roundingMode)
  override def actFn = Utilities.activationToFunctionSU(act, outputBitwidth)

  override def genI = SInt(inputBitwidth.W)
  override def genO = UInt(outputBitwidth.W)
}

class NeuronComputeUSU(act: lbir.Activation, inputBitwidth: Int, outputBitwidth: Int, roundingMode: String)
    extends NeuronCompute {

  type I = UInt
  type W = SInt
  type M = SInt
  type A = SInt

  type O = UInt
  override def ringA: Ring[A] = implicitly[Ring[SInt]]
  override def binA:  BinaryRepresentation[A] = implicitly[BinaryRepresentation[SInt]]
  override def mul = (i: UInt, w: SInt) => i * w
  override def add = (x: Vec[SInt]) => x.reduceTree(_ +& _)
  override def shiftAndRoundStatic:  (SInt, Int) => SInt = shiftAndRoundSIntStatic(roundingMode)
  override def shiftAndRoundDynamic: (SInt, UInt, Bool) => SInt = shiftAndRoundSIntDynamic(roundingMode)
  override def actFn = Utilities.activationToFunctionSU(act, outputBitwidth)

  override def genI = UInt(inputBitwidth.W)
  override def genO = UInt(outputBitwidth.W)
}

class NeuronComputeSSS(act: lbir.Activation, inputBitwidth: Int, outputBitwidth: Int, roundingMode: String)
    extends NeuronCompute {

  type I = SInt
  type W = SInt
  type M = SInt
  type A = SInt
  type O = SInt
  override def ringA: Ring[A] = implicitly[Ring[SInt]]
  override def binA:  BinaryRepresentation[A] = implicitly[BinaryRepresentation[SInt]]
  override def mul = (i: SInt, w: SInt) => i * w
  override def add = (x: Vec[SInt]) => x.reduceTree(_ +& _)
  override def shiftAndRoundStatic:  (SInt, Int) => SInt = shiftAndRoundSIntStatic(roundingMode)
  override def shiftAndRoundDynamic: (SInt, UInt, Bool) => SInt = shiftAndRoundSIntDynamic(roundingMode)
  override def actFn = Utilities.activationToFunctionSS(act, outputBitwidth)

  override def genI = SInt(inputBitwidth.W)
  override def genO = SInt(outputBitwidth.W)
}

class NeuronComputeUSS(act: lbir.Activation, inputBitwidth: Int, outputBitwidth: Int, roundingMode: String)
    extends NeuronCompute {
  type I = UInt
  type W = SInt
  type M = SInt
  type A = SInt
  type O = SInt
  override def ringA: Ring[A] = implicitly[Ring[SInt]]
  override def binA:  BinaryRepresentation[A] = implicitly[BinaryRepresentation[SInt]]
  override def mul = (i: UInt, w: SInt) => i * w
  override def add = (x: Vec[SInt]) => x.reduceTree(_ +& _)
  override def shiftAndRoundStatic:  (SInt, Int) => SInt = shiftAndRoundSIntStatic(roundingMode)
  override def shiftAndRoundDynamic: (SInt, UInt, Bool) => SInt = shiftAndRoundSIntDynamic(roundingMode)
  override def actFn = Utilities.activationToFunctionSS(act, outputBitwidth)

  override def genI = UInt(inputBitwidth.W)
  override def genO = SInt(outputBitwidth.W)
}

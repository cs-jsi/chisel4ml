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
import chisel4ml.HasLayerWrap
import chisel4ml.compute.ActivatableImplementations._
import chisel4ml.compute.MultipliableImplementations._
import chisel4ml.compute.ShiftRoundableImplementations._
import chisel4ml.compute.VectorAddableImplementations._
import chisel4ml.implicits._
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
    case (UNIFORM, false, UNIFORM, false, UNIFORM, false) => new NeuronComputeUIntUIntUInt(l)
    case (UNIFORM, false, UNIFORM, false, UNIFORM, true)  => new NeuronComputeUIntUIntSInt(l)
    case (UNIFORM, false, UNIFORM, false, BINARY, _)      => new NeuronComputeUIntUIntBool(l)

    case (UNIFORM, false, UNIFORM, true, UNIFORM, false) => new NeuronComputeUIntSIntUInt(l)
    case (UNIFORM, false, UNIFORM, true, UNIFORM, true)  => new NeuronComputeUIntSIntSInt(l)
    case (UNIFORM, false, UNIFORM, true, BINARY, _)      => new NeuronComputeUIntSIntBool(l)

    case (UNIFORM, false, BINARY, _, UNIFORM, false) => new NeuronComputeUIntBoolUInt(l)
    case (UNIFORM, false, BINARY, _, UNIFORM, true)  => new NeuronComputeUIntBoolSInt(l)
    case (UNIFORM, false, BINARY, _, BINARY, _)      => new NeuronComputeUIntBoolBool(l)

    case (UNIFORM, true, UNIFORM, false, UNIFORM, false) => new NeuronComputeSIntUIntUInt(l)
    case (UNIFORM, true, UNIFORM, false, UNIFORM, true)  => new NeuronComputeSIntUIntSInt(l)
    case (UNIFORM, true, UNIFORM, false, BINARY, _)      => new NeuronComputeSIntUIntBool(l)

    case (UNIFORM, true, UNIFORM, true, UNIFORM, false) => new NeuronComputeSIntSIntUInt(l)
    case (UNIFORM, true, UNIFORM, true, UNIFORM, true)  => new NeuronComputeSIntSIntSInt(l)
    case (UNIFORM, true, UNIFORM, true, BINARY, _)      => new NeuronComputeSIntSIntBool(l)

    case (UNIFORM, true, BINARY, _, UNIFORM, false) => new NeuronComputeSIntBoolUInt(l)
    case (UNIFORM, true, BINARY, _, UNIFORM, true)  => new NeuronComputeSIntBoolSInt(l)
    case (UNIFORM, true, BINARY, _, BINARY, _)      => new NeuronComputeSIntBoolBool(l)

    case (BINARY, _, UNIFORM, false, UNIFORM, false) => new NeuronComputeBoolUIntUInt(l)
    case (BINARY, _, UNIFORM, false, UNIFORM, true)  => new NeuronComputeBoolUIntSInt(l)
    case (BINARY, _, UNIFORM, false, BINARY, _)      => new NeuronComputeBoolUIntBool(l)

    case (BINARY, _, UNIFORM, true, UNIFORM, false) => new NeuronComputeBoolSIntUInt(l)
    case (BINARY, _, UNIFORM, true, UNIFORM, true)  => new NeuronComputeBoolSIntSInt(l)
    case (BINARY, _, UNIFORM, true, BINARY, _)      => new NeuronComputeBoolSIntBool(l)

    case (BINARY, _, BINARY, _, UNIFORM, false) => new NeuronComputeBoolBoolUInt(l)
    case (BINARY, _, BINARY, _, UNIFORM, true)  => new NeuronComputeBoolBoolSInt(l)
    case (BINARY, _, BINARY, _, BINARY, _)      => new NeuronComputeBoolBoolBool(l)

    case _ =>
      throw new RuntimeException(
        f"Quantization type not supported: ${l.input.dtype.quantization}, ${l.input.dtype.signed}, ${l.kernel.dtype.quantization}, ${l.kernel.dtype.signed}, ${l.output.dtype.quantization}, ${l.output.dtype.signed}."
      )
  }
}

abstract class NeuronCompute extends HasLayerWrap {
  type I <: Bits
  type W <: Bits
  type M <: Bits
  type A <: Bits
  type O <: Bits
  def rngA:              Ring[A]
  def binA:              BinaryRepresentation[A]
  def mul:               (I, W) => M
  def addVec:            Vec[M] => A
  def shiftRound:        (A, Int) => A
  def shiftRoundDynamic: (A, UInt, Bool) => A
  def actFn:             (A, A) => O
  def genI: I = cfg.input.getType[I]
  def genO: O = cfg.output.getType[O]
}

class NeuronComputeUIntUIntUInt(val cfg: LayerWrap with IsActiveLayer)
    extends NeuronCompute
    with MultipliableUIntUInt
    with VectorAddableUIntUInt
    with ShiftRoundableUInt
    with ActivatableUIntUInt {
  type I = UInt
  type W = UInt
  type M = UInt
  type A = UInt
  type O = UInt
  override def rngA = Ring[UInt]
  override def binA = BinaryRepresentation[UInt]
}

class NeuronComputeUIntUIntSInt(val cfg: LayerWrap with IsActiveLayer)
    extends NeuronCompute
    with MultipliableUIntUInt
    with VectorAddableUIntSInt
    with ShiftRoundableSInt
    with ActivatableSIntSInt {
  type I = UInt
  type W = UInt
  type M = UInt
  type A = SInt
  type O = SInt
  override def rngA = Ring[SInt]
  override def binA = BinaryRepresentation[SInt]
}

class NeuronComputeUIntUIntBool(val cfg: LayerWrap with IsActiveLayer)
    extends NeuronCompute
    with MultipliableUIntUInt
    with VectorAddableUIntSInt
    with ShiftRoundableSInt
    with ActivatableSIntBool {
  type I = UInt
  type W = UInt
  type M = UInt
  type A = SInt
  type O = Bool
  override def rngA = Ring[SInt]
  override def binA = BinaryRepresentation[SInt]
}

class NeuronComputeUIntSIntUInt(val cfg: LayerWrap with IsActiveLayer)
    extends NeuronCompute
    with MultipliableUIntSInt
    with VectorAddableSIntSInt
    with ShiftRoundableSInt
    with ActivatableSIntUInt {
  type I = UInt
  type W = SInt
  type M = SInt
  type A = SInt
  type O = UInt
  override def rngA = Ring[SInt]
  override def binA = BinaryRepresentation[SInt]
}

class NeuronComputeUIntSIntSInt(val cfg: LayerWrap with IsActiveLayer)
    extends NeuronCompute
    with MultipliableUIntSInt
    with VectorAddableSIntSInt
    with ShiftRoundableSInt
    with ActivatableSIntSInt {
  type I = UInt
  type W = SInt
  type M = SInt
  type A = SInt
  type O = SInt
  override def rngA = Ring[SInt]
  override def binA = BinaryRepresentation[SInt]
}

class NeuronComputeUIntSIntBool(val cfg: LayerWrap with IsActiveLayer)
    extends NeuronCompute
    with MultipliableUIntSInt
    with VectorAddableSIntSInt
    with ShiftRoundableSInt
    with ActivatableSIntBool {
  type I = UInt
  type W = SInt
  type M = SInt
  type A = SInt
  type O = Bool
  override def rngA = Ring[SInt]
  override def binA = BinaryRepresentation[SInt]
}

class NeuronComputeUIntBoolUInt(val cfg: LayerWrap with IsActiveLayer)
    extends NeuronCompute
    with MultipliableUIntBool
    with VectorAddableSIntSInt
    with ShiftRoundableSInt
    with ActivatableSIntUInt {
  type I = UInt
  type W = Bool
  type M = SInt
  type A = SInt
  type O = UInt
  override def rngA = Ring[SInt]
  override def binA = BinaryRepresentation[SInt]
}
class NeuronComputeUIntBoolSInt(val cfg: LayerWrap with IsActiveLayer)
    extends NeuronCompute
    with MultipliableUIntBool
    with VectorAddableSIntSInt
    with ShiftRoundableSInt
    with ActivatableSIntSInt {
  type I = UInt
  type W = Bool
  type M = SInt
  type A = SInt
  type O = SInt
  override def rngA = Ring[SInt]
  override def binA = BinaryRepresentation[SInt]
}
class NeuronComputeUIntBoolBool(val cfg: LayerWrap with IsActiveLayer)
    extends NeuronCompute
    with MultipliableUIntBool
    with VectorAddableSIntSInt
    with ShiftRoundableSInt
    with ActivatableSIntBool {
  type I = UInt
  type W = Bool
  type M = SInt
  type A = SInt
  type O = Bool
  override def rngA = Ring[SInt]
  override def binA = BinaryRepresentation[SInt]
}

class NeuronComputeSIntUIntUInt(val cfg: LayerWrap with IsActiveLayer)
    extends NeuronCompute
    with MultipliableSIntUInt
    with VectorAddableSIntSInt
    with ShiftRoundableSInt
    with ActivatableSIntUInt {
  type I = SInt
  type W = UInt
  type M = SInt
  type A = SInt
  type O = UInt
  override def rngA = Ring[SInt]
  override def binA = BinaryRepresentation[SInt]
}

class NeuronComputeSIntUIntSInt(val cfg: LayerWrap with IsActiveLayer)
    extends NeuronCompute
    with MultipliableSIntUInt
    with VectorAddableSIntSInt
    with ShiftRoundableSInt
    with ActivatableSIntSInt {
  type I = SInt
  type W = UInt
  type M = SInt
  type A = SInt
  type O = SInt
  override def rngA = Ring[SInt]
  override def binA = BinaryRepresentation[SInt]
}

class NeuronComputeSIntUIntBool(val cfg: LayerWrap with IsActiveLayer)
    extends NeuronCompute
    with MultipliableSIntUInt
    with VectorAddableSIntSInt
    with ShiftRoundableSInt
    with ActivatableSIntBool {
  type I = SInt
  type W = UInt
  type M = SInt
  type A = SInt
  type O = Bool
  override def rngA = Ring[SInt]
  override def binA = BinaryRepresentation[SInt]
}

class NeuronComputeSIntSIntUInt(val cfg: LayerWrap with IsActiveLayer)
    extends NeuronCompute
    with MultipliableSIntSInt
    with VectorAddableSIntSInt
    with ShiftRoundableSInt
    with ActivatableSIntUInt {
  type I = SInt
  type W = SInt
  type M = SInt
  type A = SInt
  type O = UInt
  override def rngA = Ring[SInt]
  override def binA = BinaryRepresentation[SInt]
}

class NeuronComputeSIntSIntSInt(val cfg: LayerWrap with IsActiveLayer)
    extends NeuronCompute
    with MultipliableSIntSInt
    with VectorAddableSIntSInt
    with ShiftRoundableSInt
    with ActivatableSIntSInt {
  type I = SInt
  type W = SInt
  type M = SInt
  type A = SInt
  type O = SInt
  override def rngA = Ring[SInt]
  override def binA = BinaryRepresentation[SInt]
}

class NeuronComputeSIntSIntBool(val cfg: LayerWrap with IsActiveLayer)
    extends NeuronCompute
    with MultipliableSIntSInt
    with VectorAddableSIntSInt
    with ShiftRoundableSInt
    with ActivatableSIntBool {
  type I = SInt
  type W = SInt
  type M = SInt
  type A = SInt
  type O = Bool
  override def rngA = Ring[SInt]
  override def binA = BinaryRepresentation[SInt]
}

class NeuronComputeSIntBoolUInt(val cfg: LayerWrap with IsActiveLayer)
    extends NeuronCompute
    with MultipliableSIntBool
    with VectorAddableSIntSInt
    with ShiftRoundableSInt
    with ActivatableSIntUInt {
  type I = SInt
  type W = Bool
  type M = SInt
  type A = SInt
  type O = UInt
  override def rngA = Ring[SInt]
  override def binA = BinaryRepresentation[SInt]
}

class NeuronComputeSIntBoolSInt(val cfg: LayerWrap with IsActiveLayer)
    extends NeuronCompute
    with MultipliableSIntBool
    with VectorAddableSIntSInt
    with ShiftRoundableSInt
    with ActivatableSIntSInt {
  type I = SInt
  type W = Bool
  type M = SInt
  type A = SInt
  type O = SInt
  override def rngA = Ring[SInt]
  override def binA = BinaryRepresentation[SInt]
}

class NeuronComputeSIntBoolBool(val cfg: LayerWrap with IsActiveLayer)
    extends NeuronCompute
    with MultipliableSIntBool
    with VectorAddableSIntSInt
    with ShiftRoundableSInt
    with ActivatableSIntBool {
  type I = SInt
  type W = Bool
  type M = SInt
  type A = SInt
  type O = Bool
  override def rngA = Ring[SInt]
  override def binA = BinaryRepresentation[SInt]
}

class NeuronComputeBoolUIntUInt(val cfg: LayerWrap with IsActiveLayer)
    extends NeuronCompute
    with MultipliableBoolUInt
    with VectorAddableSIntSInt
    with ShiftRoundableSInt
    with ActivatableSIntUInt {
  type I = Bool
  type W = UInt
  type M = SInt
  type A = SInt
  type O = UInt
  override def rngA = Ring[SInt]
  override def binA = BinaryRepresentation[SInt]
}

class NeuronComputeBoolUIntSInt(val cfg: LayerWrap with IsActiveLayer)
    extends NeuronCompute
    with MultipliableBoolUInt
    with VectorAddableSIntSInt
    with ShiftRoundableSInt
    with ActivatableSIntSInt {
  type I = Bool
  type W = UInt
  type M = SInt
  type A = SInt
  type O = SInt
  override def rngA = Ring[SInt]
  override def binA = BinaryRepresentation[SInt]
}

class NeuronComputeBoolUIntBool(val cfg: LayerWrap with IsActiveLayer)
    extends NeuronCompute
    with MultipliableBoolUInt
    with VectorAddableSIntSInt
    with ShiftRoundableSInt
    with ActivatableSIntBool {
  type I = Bool
  type W = UInt
  type M = SInt
  type A = SInt
  type O = Bool
  override def rngA = Ring[SInt]
  override def binA = BinaryRepresentation[SInt]
}

class NeuronComputeBoolSIntUInt(val cfg: LayerWrap with IsActiveLayer)
    extends NeuronCompute
    with MultipliableBoolSInt
    with VectorAddableSIntSInt
    with ShiftRoundableSInt
    with ActivatableSIntUInt {
  type I = Bool
  type W = SInt
  type M = SInt
  type A = SInt
  type O = UInt
  override def rngA = Ring[SInt]
  override def binA = BinaryRepresentation[SInt]
}

class NeuronComputeBoolSIntSInt(val cfg: LayerWrap with IsActiveLayer)
    extends NeuronCompute
    with MultipliableBoolSInt
    with VectorAddableSIntSInt
    with ShiftRoundableSInt
    with ActivatableSIntSInt {
  type I = Bool
  type W = SInt
  type M = SInt
  type A = SInt
  type O = SInt
  override def rngA = Ring[SInt]
  override def binA = BinaryRepresentation[SInt]
}

class NeuronComputeBoolSIntBool(val cfg: LayerWrap with IsActiveLayer)
    extends NeuronCompute
    with MultipliableBoolSInt
    with VectorAddableSIntSInt
    with ShiftRoundableSInt
    with ActivatableSIntBool {
  type I = Bool
  type W = SInt
  type M = SInt
  type A = SInt
  type O = Bool
  override def rngA = Ring[SInt]
  override def binA = BinaryRepresentation[SInt]
}

class NeuronComputeBoolBoolUInt(val cfg: LayerWrap with IsActiveLayer)
    extends NeuronCompute
    with MultipliableBoolBool
    with VectorAddableBoolUInt
    with ShiftRoundableUInt
    with ActivatableUIntUInt {
  type I = Bool
  type W = Bool
  type M = Bool
  type A = UInt
  type O = UInt
  override def rngA = Ring[UInt]
  override def binA = BinaryRepresentation[UInt]
}

class NeuronComputeBoolBoolSInt(val cfg: LayerWrap with IsActiveLayer)
    extends NeuronCompute
    with MultipliableBoolBool
    with VectorAddableBoolSInt
    with ShiftRoundableSInt
    with ActivatableSIntSInt {
  type I = Bool
  type W = Bool
  type M = Bool
  type A = SInt
  type O = SInt
  override def rngA = Ring[SInt]
  override def binA = BinaryRepresentation[SInt]
}

class NeuronComputeBoolBoolBool(val cfg: LayerWrap with IsActiveLayer)
    extends NeuronCompute
    with MultipliableBoolBool
    with VectorAddableBoolUInt
    with ShiftRoundableUInt
    with ActivatableUIntBool {
  type I = Bool
  type W = Bool
  type M = Bool
  type A = UInt
  type O = Bool
  override def rngA = Ring[UInt]
  override def binA = BinaryRepresentation[UInt]
}

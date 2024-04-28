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
import chisel4ml.implicits._
import chisel4ml.quantization._
import chisel4ml.conv2d._
import spire.algebra.Ring
import spire.implicits._

object Neuron {
  def apply[I <: Bits, W <: Bits, M <: Bits, A <: Bits, O <: Bits](
    in:             Seq[I],
    weights:        Seq[W],
    thresh:         A,
    shift:          Int,
    outputBitwidth: Int,
    useThresh:      Boolean,
    roundingMode:   lbir.RoundingMode
  )(qc:             QuantizationContext[I, W, M, A, O]
  ): O = if (useThresh) {
    NeuronWithBias[I, W, M, A, O](in, weights, thresh, shift, outputBitwidth, roundingMode)(qc)
  } else {
    NeuronWithoutBias[I, W, M, A, O](in, weights, thresh, shift, outputBitwidth, roundingMode)(qc)
  }
}

object NeuronWithBias {
  def apply[I <: Bits, W <: Bits, M <: Bits, A <: Bits, O <: Bits](
    in:             Seq[I],
    weights:        Seq[W],
    thresh:         A,
    shift:          Int,
    outputBitwidth: Int,
    roundingMode:   lbir.RoundingMode
  )(qc:             QuantizationContext[I, W, M, A, O]
  ): O = {
    val muls = VecInit((in.zip(weights)).map { case (i, w) => qc.mul(i, w) })
    require(shift <= 0)
    val threshAdjusted = (thresh << shift.abs).asSInt.asInstanceOf[A]
    val pAct = qc.ringA.minus(qc.add(muls), threshAdjusted)
    val sAct = qc.shiftAndRound(pAct, shift.abs.U, (shift > 0).B, roundingMode)
    qc.actFn(sAct, qc.ringA.zero, outputBitwidth)
  }
}

object NeuronWithoutBias {
  def apply[I <: Bits, W <: Bits, M <: Bits, A <: Bits, O <: Bits](
    in:             Seq[I],
    weights:        Seq[W],
    thresh:         A,
    shift:          Int,
    outputBitwidth: Int,
    roundingMode:   lbir.RoundingMode
  )(qc:             QuantizationContext[I, W, M, A, O]
  ): O = {
    val muls = VecInit((in.zip(weights)).map { case (i, w) => qc.mul(i, w) })
    val pAct = qc.add(muls)
    val sAct = qc.shiftAndRound(pAct, shift.abs.U, (shift > 0).B, roundingMode)
    qc.actFn(sAct, thresh, outputBitwidth)
  }
}

class DynamicNeuron[I <: Bits, W <: Bits, M <: Bits, A <: Bits, O <: Bits](
  l:  lbir.Conv2DConfig,
  qc: QuantizationContext[I, W, M, A, O])
    extends Module {
  val io = IO(new Bundle {
    val in = Flipped(Decoupled(Vec(l.kernel.numActiveParams(l.depthwise), l.input.getType[I])))
    val weights = Flipped(Valid(new KernelSubsystemIO[W, A](l.kernel, l.thresh, l.depthwise)))
    val out = Decoupled(l.output.getType[O])
  })
  val inWeights = io.weights.bits.activeKernel.asTypeOf(Vec(l.kernel.numActiveParams(l.depthwise), l.kernel.getType[W]))

  val muls = VecInit((io.in.bits.zip(inWeights)).map { case (i, w) => qc.mul(i, w) })
  assert((!io.weights.bits.threshShift.shiftLeft) || (io.weights.bits.threshShift.shift === 0.U))

  val maxBits: Int = log2Up(l.thresh.values.map(_.abs).max.toInt) + l.kernel.dtype.shift.map(_.abs).max.toInt + 1
  val threshAdjusted =
    (io.weights.bits.threshShift.thresh << io.weights.bits.threshShift.shift)(maxBits - 1, 0).zext.asInstanceOf[A]
  val pAct = qc.ringA.minus(qc.add(muls), threshAdjusted)
  val sAct = qc.shiftAndRound(
    pAct,
    io.weights.bits.threshShift.shift,
    io.weights.bits.threshShift.shiftLeft,
    l.roundingMode
  )
  io.out.bits := qc.actFn(sAct, qc.ringA.zero, l.output.dtype.bitwidth)

  io.out.valid := io.in.valid && io.weights.valid
  io.in.ready := io.out.ready && io.weights.valid
}

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
import chisel4ml.conv2d._
import chisel4ml.implicits._
import chisel4ml.quantization._

trait Transformation {
  def apply(
    qc:      QuantizationContext
  )(in:      Seq[qc.I],
    weights: Seq[qc.W],
    thresh:  qc.A,
    shift:   Int
  ): qc.O
}

trait TransformationIO {
  def apply(qc: QuantizationContext)(in: Seq[qc.I]): qc.O
}

object NeuronWithBias extends Transformation {
  def apply(
    qc:      QuantizationContext
  )(in:      Seq[qc.I],
    weights: Seq[qc.W],
    thresh:  qc.A,
    shift:   Int
  ): qc.O = {
    val muls = VecInit((in.zip(weights)).map { case (i, w) => qc.mul(i, w) })
    require(shift <= 0)
    val threshAdjusted = (thresh << shift.abs).asSInt.asInstanceOf[qc.A]
    val pAct = qc.minA(qc.add(muls), threshAdjusted)
    val sAct = qc.shiftAndRoundStatic(pAct, shift)
    qc.actFn(sAct, qc.zeroA)
  }
}

object NeuronWithoutBias extends Transformation {
  def apply(
    qc:      QuantizationContext
  )(in:      Seq[qc.I],
    weights: Seq[qc.W],
    thresh:  qc.A,
    shift:   Int
  ): qc.O = {
    val muls = VecInit((in.zip(weights)).map { case (i, w) => qc.mul(i, w) })
    val pAct = qc.add(muls)
    val sAct = qc.shiftAndRoundStatic(pAct, shift)
    qc.actFn(sAct, thresh)
  }
}

object MaximumTransformationIO extends TransformationIO {
  def getMaximum(qc: QuantizationContext)(in0: qc.I, in1: qc.I): qc.I = {
    val out: qc.I = Wire(chiselTypeOf(in0))
    when(qc.gt(in0, in1)) {
      out := in0
    }.otherwise {
      out := in1
    }
    out.asInstanceOf
  }
  def apply(qc: QuantizationContext)(in: Seq[qc.I]): qc.O = {
    VecInit(in).reduceTree(getMaximum(qc)(_, _)).asInstanceOf[qc.O]
  }
}

class DynamicNeuron(l: lbir.Conv2DConfig, val qc: QuantizationContext) extends Module {
  val io = IO(new Bundle {
    val in = Flipped(Decoupled(Vec(l.kernel.numActiveParams(l.depthwise), l.input.getType[qc.I])))
    val weights = Flipped(Valid(new KernelSubsystemIO[qc.W, qc.A](l.kernel, l.thresh, l.depthwise)))
    val out = Decoupled(l.output.getType[qc.O])
  })
  val inWeights =
    io.weights.bits.activeKernel.asTypeOf(Vec(l.kernel.numActiveParams(l.depthwise), l.kernel.getType[qc.W]))

  val muls = VecInit((io.in.bits.zip(inWeights)).map { case (i, w) => qc.mul(i, w) })
  assert((!io.weights.bits.threshShift.shiftLeft) || (io.weights.bits.threshShift.shift === 0.U))
  val pAct = qc.addA(qc.add(muls), io.weights.bits.threshShift.bias)
  val sAct = qc.shiftAndRoundDynamic(
    pAct,
    io.weights.bits.threshShift.shift,
    io.weights.bits.threshShift.shiftLeft
  )
  io.out.bits := qc.actFn(sAct, qc.zeroA)

  io.out.valid := io.in.valid && io.weights.valid
  io.in.ready := io.out.ready && io.weights.valid
}

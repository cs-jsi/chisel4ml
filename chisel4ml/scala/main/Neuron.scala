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

object NeuronWithBias {
  def apply(
    qc:             QuantizationContext
  )(in:             Seq[qc.I],
    weights:        Seq[qc.W],
    thresh:         qc.A,
    shift:          Int,
    outputBitwidth: Int
  ): qc.O = {
    val muls = VecInit((in.zip(weights)).map { case (i, w) => qc.mul(i, w) })
    require(shift <= 0)
    val threshAdjusted = (thresh << shift.abs).asSInt.asInstanceOf[qc.A]
    val pAct = qc.minA(qc.add(muls), threshAdjusted)
    val sAct = qc.shiftAndRoundStatic(pAct, shift)
    qc.actFn(sAct, qc.zeroA(outputBitwidth), outputBitwidth)
  }
}

object NeuronWithoutBias {
  def apply(
    qc:             QuantizationContext
  )(in:             Seq[qc.I],
    weights:        Seq[qc.W],
    thresh:         qc.A,
    shift:          Int,
    outputBitwidth: Int
  ): qc.O = {
    val muls = VecInit((in.zip(weights)).map { case (i, w) => qc.mul(i, w) })
    val pAct = qc.add(muls)
    val sAct = qc.shiftAndRoundStatic(pAct, shift)
    qc.actFn(sAct, thresh, outputBitwidth)
  }
}

class DynamicNeuron(
  l:      lbir.Conv2DConfig,
  val qc: QuantizationContext)
    extends Module {
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
  io.out.bits := qc.actFn(sAct, qc.zeroA(l.output.dtype.bitwidth), l.output.dtype.bitwidth)

  io.out.valid := io.in.valid && io.weights.valid
  io.in.ready := io.out.ready && io.weights.valid
}

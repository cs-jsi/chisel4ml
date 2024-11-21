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
package chisel4ml.sequential

import chisel3._
import chisel3.util._
import chisel4ml.compute.NeuronCompute
import chisel4ml.implicits._
import dsptools.{DspContext, Grow}

// TODO:  make generic (instead of tied to conv2d)
class DynamicNeuron(val nc: NeuronCompute)(l: lbir.Conv2DConfig) extends Module {
  val io = IO(new Bundle {
    val in = Flipped(Decoupled(Vec(l.kernel.numActiveParams(l.depthwise), nc.genI)))
    val weights = Flipped(Valid(new KernelSubsystemIO[nc.W, nc.A](l.kernel, l.thresh, l.depthwise)))
    val out = Decoupled(nc.genO)
  })
  val inWeights =
    io.weights.bits.activeKernel.asTypeOf(Vec(l.kernel.numActiveParams(l.depthwise), l.kernel.getType[nc.W]))

  val muls = VecInit((io.in.bits.zip(inWeights)).map { case (i, w) => nc.mul(i, w) })
  assert((!io.weights.bits.threshShift.shiftLeft) || (io.weights.bits.threshShift.shift === 0.U))
  val pAct = DspContext.withOverflowType(Grow) {
    nc.rngA.plusContext(nc.addVec(muls), io.weights.bits.threshShift.bias)
  }
  val sAct = nc.shiftRoundDynamic(
    pAct,
    io.weights.bits.threshShift.shift,
    io.weights.bits.threshShift.shiftLeft
  )
  io.out.bits := nc.actFn(sAct, nc.rngA.zero)

  io.out.valid := io.in.valid && io.weights.valid
  io.in.ready := io.out.ready && io.weights.valid
}

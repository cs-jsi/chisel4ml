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
package chisel4ml.conv2d

import chisel3._
import chisel3.util._
import chisel4ml.util.{saturate, shiftAndRoundDynamic}
import chisel4ml.implicits._

class DynamicNeuron[I <: Bits with Num[I], W <: Bits with Num[W], M <: Bits, S <: Bits, A <: Bits, O <: Bits](
  l:          lbir.Conv2DConfig,
  genIn:      I,
  genWeights: W,
  genAccu:    S,
  genThresh:  A,
  genOut:     O,
  mul:        (I, W) => M,
  add:        Vec[M] => S,
  actFn:      (S, A) => O)
    extends Module {
  val io = IO(new Bundle {
    val in = Flipped(Decoupled(Vec(l.kernel.numActiveParams(l.depthwise), UInt())))
    val weights = Flipped(Valid(new KernelSubsystemIO(l.kernel, genThresh)))
    val out = Decoupled(genOut)
  })

  val inVec = io.in.asTypeOf(Vec(l.kernel.numActiveParams(l.depthwise), genIn))
  val inWeights = io.weights.bits.activeKernel.asTypeOf(Vec(l.kernel.numActiveParams(l.depthwise), genWeights))

  val muls = VecInit((inVec.zip(inWeights)).map { case (a, b) => mul(a, b) })
  val pAct = add(muls)
  dontTouch(io.weights.bits.threshShift.shift)
  dontTouch(io.weights.bits.threshShift.shiftLeft)
  val sAct =
    shiftAndRoundDynamic(pAct, io.weights.bits.threshShift.shift, io.weights.bits.threshShift.shiftLeft, genAccu)
  io.out.bits := saturate(
    actFn(sAct, io.weights.bits.threshShift.thresh).asUInt,
    genOut.getWidth,
    l.output.dtype.signed
  ).asTypeOf(io.out.bits)

  io.out.valid := io.in.valid && io.weights.valid
  io.in.ready := io.out.ready
}

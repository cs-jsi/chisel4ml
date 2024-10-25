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
package chisel4ml.combinational

import chisel3._
import chisel4ml.compute.NeuronCompute
import dsptools.{DspContext, Grow}

trait NeuronOperation {
  def apply(
    qc:      NeuronCompute
  )(in:      Seq[qc.I],
    weights: Seq[qc.W],
    thresh:  qc.A,
    shift:   Int
  ): qc.O
}

object NeuronWithBias extends NeuronOperation {
  def apply(
    nc:      NeuronCompute
  )(in:      Seq[nc.I],
    weights: Seq[nc.W],
    thresh:  nc.A,
    shift:   Int
  ): nc.O = {
    val muls = VecInit((in.zip(weights)).map { case (i, w) => nc.mul(i, w) })
    require(shift <= 0)
    val threshAdjusted = nc.binA.shl(thresh, shift.abs)
    val pAct = DspContext.withOverflowType(Grow) {
      nc.ringA.minusContext(nc.add(muls), threshAdjusted)
    }
    val sAct = nc.shiftAndRoundStatic(pAct, shift)
    nc.actFn(sAct, nc.ringA.zero)
  }
}

object NeuronWithoutBias extends NeuronOperation {
  def apply(
    nc:      NeuronCompute
  )(in:      Seq[nc.I],
    weights: Seq[nc.W],
    thresh:  nc.A,
    shift:   Int
  ): nc.O = {
    val muls = VecInit((in.zip(weights)).map { case (i, w) => nc.mul(i, w) })
    val pAct = nc.add(muls)
    val sAct = nc.shiftAndRoundStatic(pAct, shift)
    nc.actFn(sAct, thresh)
  }
}

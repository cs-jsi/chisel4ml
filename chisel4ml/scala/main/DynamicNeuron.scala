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

import _root_.chisel3._
import _root_.chisel3.util._

import _root_.chisel4ml.util._
import _root_.chisel4ml.combinational.StaticNeuron
import _root_.chisel4ml.lbir._

class DynamicNeuron[I <: Bits, W <: Bits: WeightsProvider, M <: Bits, A <: Bits: ThreshProvider, O <: Bits](
    genIn:      I,
    numIn:      Int,
    genWeights: W,
    numWeights: Int,
    genThresh:  A,
    genOut:     O,
    mul:        (I, W) => M,
    add:        Vec[M] => A,
    actFn:      (A, A) => O)
    extends Module {

    def shiftAndRound[A <: Bits: ThreshProvider](pAct: A, shift: UInt, shiftLeft: Bool, genThresh: A): A = {
        val out = Wire(genThresh)
        when(shiftLeft) {
            out := (pAct << shift).asTypeOf(pAct)
        }.otherwise {
            out := ((pAct >> shift).asSInt + pAct(shift - 1.U).asSInt).asTypeOf(pAct)
        }
        out
    }

    val io = IO(new Bundle {
        val in:        Vec[I] = Input(Vec(numIn, genIn))
        val weights:   Vec[W] = Input(Vec(numWeights, genWeights))
        val thresh:    A      = Input(genThresh)
        val shift:     UInt   = Input(UInt(8.W)) // TODO: bitwidth?
        val shiftLeft: Bool   = Input(Bool())
        val out:       O      = Output(genOut)
    })

    val muls = VecInit((io.in.zip(io.weights)).map { case (a, b) => mul(a, b) })
    val pAct = add(muls)
    val sAct = shiftAndRound(pAct, io.shift, io.shiftLeft, genThresh)
    actFn(sAct, io.thresh)
}

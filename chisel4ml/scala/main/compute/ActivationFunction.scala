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
import chisel4ml.util._
import dsptools.numbers.implicits._

// ACTIVATION FUNCTIONS
trait ActivationFunction[A <: Data, O <: Data] {
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

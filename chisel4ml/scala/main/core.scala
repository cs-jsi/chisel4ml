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
import chisel4ml.util.{mul, reluFn, signFn}

trait QuantizationCompute[I <: Bits, W <: Bits, M <: Bits, A <: Bits, O <: Bits] extends Any {
  def mul:   (I, W) => M
  def add:   Vec[M] => A
  def actFn: (A, A) => O
}

object BinarizedQuantizationCompute extends QuantizationCompute[Bool, Bool, Bool, UInt, Bool] {
  def mul = (i: Bool, w: Bool) => ~(i ^ w)
  def add = (x: Vec[Bool]) => PopCount(x.asUInt)
  def actFn: (UInt, UInt) => Bool = signFn
}

object BinaryQuantizationCompute extends QuantizationCompute[UInt, Bool, SInt, SInt, Bool] {
  def mul = (i: UInt, w: Bool) => Mux(w, i.zext, -(i.zext))
  def add = (x: Vec[SInt]) => x.reduceTree(_ +& _)
  def actFn = signFn
}

object BinaryQuantizationComputeS extends QuantizationCompute[SInt, Bool, SInt, SInt, Bool] {
  def mul = (i: SInt, w: Bool) => Mux(w, i, -i)
  def add = (x: Vec[SInt]) => x.reduceTree(_ +& _)
  def actFn = signFn
}

// implementiraj s dsptools?
//trait UniformQuantizationCompute[I <: Bits, W <: Bits,] extends Any with QuantizationCompute[]
class UniformQuantizationComputeSSU(act: (SInt, SInt) => UInt)
    extends QuantizationCompute[SInt, SInt, SInt, SInt, UInt] {
  def mul = (i: SInt, w: SInt) => i * w
  def add = (x: Vec[SInt]) => x.reduceTree(_ + _)
  def actFn = act
}

object UniformQuantizationComputeSSUReLU extends UniformQuantizationComputeSSU(reluFn)

class UniformQuantizationComputeUSU(act: (SInt, SInt) => UInt)
    extends QuantizationCompute[UInt, SInt, SInt, SInt, UInt] {
  def mul = (i: UInt, w: SInt) => i * w
  def add = (x: Vec[SInt]) => x.reduceTree(_ + _)
  def actFn = act
}

object UniformQuantizationComputeUSUReLU extends UniformQuantizationComputeUSU(reluFn)

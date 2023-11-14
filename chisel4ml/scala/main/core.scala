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
import chisel4ml.util.{reluFn, saturateFnS, signFnS, signFnU}

trait QuantizationCompute[I <: Bits, W <: Bits, M <: Bits, A <: Bits, O <: Bits] extends Any {
  def mul:   (I, W) => M
  def add:   Vec[M] => A
  def actFn: (A, A, Int) => O
}

object BinarizedQuantizationCompute extends QuantizationCompute[Bool, Bool, Bool, UInt, Bool] {
  def mul = (i: Bool, w: Bool) => ~(i ^ w)
  def add = (x: Vec[Bool]) => PopCount(x.asUInt)
  def actFn: (UInt, UInt, Int) => Bool = signFnU
}

object BinaryQuantizationCompute extends QuantizationCompute[UInt, Bool, SInt, SInt, Bool] {
  def mul = (i: UInt, w: Bool) => Mux(w, i.zext, -(i.zext))
  def add = (x: Vec[SInt]) => x.reduceTree(_ +& _)
  def actFn = signFnS
}

object BinaryQuantizationComputeS extends QuantizationCompute[SInt, Bool, SInt, SInt, Bool] {
  def mul = (i: SInt, w: Bool) => Mux(w, i, -i)
  def add = (x: Vec[SInt]) => x.reduceTree(_ +& _)
  def actFn = signFnS
}

// implementiraj s dsptools?
class UniformQuantizationComputeSSU(act: (SInt, SInt, Int) => UInt)
    extends QuantizationCompute[SInt, SInt, SInt, SInt, UInt] {
  def mul = (i: SInt, w: SInt) => i * w
  def add = (x: Vec[SInt]) => x.reduceTree(_ +& _)
  def actFn = act
}

object UniformQuantizationComputeSSUReLU extends UniformQuantizationComputeSSU(reluFn)

class UniformQuantizationComputeUSU(act: (SInt, SInt, Int) => UInt)
    extends QuantizationCompute[UInt, SInt, SInt, SInt, UInt] {
  def mul = (i: UInt, w: SInt) => i * w
  def add = (x: Vec[SInt]) => x.reduceTree(_ +& _)
  def actFn = act
}

object UniformQuantizationComputeUSUReLU extends UniformQuantizationComputeUSU(reluFn)

class UniformQuantizationComputeSSS(act: (SInt, SInt, Int) => SInt)
    extends QuantizationCompute[SInt, SInt, SInt, SInt, SInt] {
  def mul = (i: SInt, w: SInt) => i * w
  def add = (x: Vec[SInt]) => x.reduceTree(_ +& _)
  def actFn = act
}

object UniformQuantizationComputeSSSNoAct
    extends UniformQuantizationComputeSSS((i: SInt, t: SInt, bw: Int) => saturateFnS(i - t, bw))

class UniformQuantizationComputeUSS(act: (SInt, SInt, Int) => SInt)
    extends QuantizationCompute[UInt, SInt, SInt, SInt, SInt] {
  def mul = (i: UInt, w: SInt) => i * w
  def add = (x: Vec[SInt]) => x.reduceTree(_ +& _)
  def actFn = act
}

object UniformQuantizationComputeUSSNoAct
    extends UniformQuantizationComputeUSS((i: SInt, t: SInt, bw: Int) => saturateFnS(i - t, bw))

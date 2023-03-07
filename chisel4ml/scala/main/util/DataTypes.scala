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
package chisel4ml.lbir

import _root_.lbir._
import chisel3._

final class LbirDataTransforms
object LbirDataTransforms {
  def transformWeights[T <: Bits: WeightsProvider](tensor: QTensor): Seq[Seq[T]] =
    WeightsProvider.transformWeights[T](tensor)

  def transformThresh[T <: Bits: ThreshProvider](tensor: QTensor, fanIn: Int): Seq[T] =
    ThreshProvider.transformThresh[T](tensor, fanIn)
}

trait ThreshProvider[T <: Bits] {
  def instance(tensor: QTensor, fanIn: Int): Seq[T]
}

object ThreshProvider {
  def transformThresh[T <: Bits: ThreshProvider](tensor: QTensor, fanIn: Int): Seq[T] =
    implicitly[ThreshProvider[T]].instance(tensor, fanIn)

  // Binarized neurons
  implicit object ThreshProviderUInt extends ThreshProvider[UInt] {
    def instance(tensor: QTensor, fanIn: Int): Seq[UInt] =
      tensor.values.map(x => (fanIn + x) / 2).map(_.ceil).map(_.toInt.U)
  }

  implicit object ThreshProviderSInt extends ThreshProvider[SInt] {
    def instance(tensor: QTensor, fanIn: Int): Seq[SInt] =
      tensor.values.map(_.toInt.S(tensor.dtype.get.bitwidth.W))
  }
}

trait WeightsProvider[T <: Bits] {
  def instance(tensor: QTensor): Seq[Seq[T]]
}

object WeightsProvider {
  def transformWeights[T <: Bits: WeightsProvider](tensor: QTensor): Seq[Seq[T]] =
    implicitly[WeightsProvider[T]].instance(tensor)

  implicit object WeightsProviderBool extends WeightsProvider[Bool] {
    def instance(tensor: QTensor): Seq[Seq[Bool]] =
      tensor.values.map(_ > 0).map(_.B).grouped(tensor.shape(3)).toSeq.transpose
  }

  implicit object WeightsProviderSInt extends WeightsProvider[SInt] {
    def instance(tensor: QTensor): Seq[Seq[SInt]] =
      tensor.values.map(_.toInt.S).grouped(tensor.shape(3)).toSeq.transpose
  }
}

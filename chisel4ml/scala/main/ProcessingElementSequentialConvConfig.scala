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

import _root_.chisel4ml.lbir._
import _root_.chisel4ml.memory.{ROM, SRAM}
import _root_.chisel4ml.util._
import _root_.lbir.Layer
import _root_.scala.math
import _root_.services.GenerateCircuitParams.Options
import chisel3._


case class ProcessingElementSequentialConvConfig(layer: lbir.Layer) {
  val kernel = TensorConfig(layer.weights.get)
  val thresh = ThreshConfig(layer.thresh.get)
  val input  = TensorConfig(layer.input.get)
  val result = TensorConfig(layer.output.get)

  require(kernel.numChannels == input.numChannels)
  require(kernel.numKernels == result.numChannels)
  require(kernel.height == kernel.width)
  require(result.numKernels == 1)
}

case class TensorConfig(qtensor: lbir.QTensor) {
  val numKernels:      Int = qtensor.shape(0)
  val numChannels:     Int = qtensor.shape(1)
  val height:          Int = qtensor.shape(2)
  val width:           Int = qtensor.shape(3)
  val numParams:       Int = qtensor.shape.reduce(_ * _)
  val numKernelParams: Int = numChannels * height * width
  val paramBitwidth:   Int = qtensor.dtype.get.bitwidth
  val mem                  = MemoryConfig(qtensor, numParams = numParams, paramBitwidth = paramBitwidth)
}

case class ThreshConfig(qtensor: lbir.QTensor) {
  val numParams:     Int = qtensor.shape.reduce(_ * _)
  val paramBitwidth: Int = qtensor.dtype.get.bitwidth
  val mem                = MemoryConfig(qtensor, numParams = numParams, paramBitwidth = paramBitwidth)
}

case class MemoryConfig(qtensor: lbir.QTensor, numParams: Int, paramBitwidth: Int) {
  val paramsPerWord: Int = MemWordSize.bits / paramBitwidth
  val depth:         Int = math.ceil(numParams.toFloat / paramsPerWord.toFloat).toInt
}

case object MemWordSize {
  val bits = 32
}

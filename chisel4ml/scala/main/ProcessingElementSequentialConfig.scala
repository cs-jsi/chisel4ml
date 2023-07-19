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
import _root_.scala.math
import _root_.lbir.{Layer, QTensor}

case class ProcessingElementSequentialConfig(layer: Layer) {
  val kernel = TensorConfig(layer.weights.get)
  val thresh = ThreshConfig(layer.thresh.get)
  val input  = TensorConfig(layer.input.get)
  val result = TensorConfig(layer.output.get)

  require(kernel.numChannels == input.numChannels)
  if (layer.ltype == Layer.Type.CONV2D) {
    require(kernel.numKernels == result.numChannels)
  }
}

case class TensorConfig(qtensor: QTensor) {
  val numKernels:      Int = if (qtensor.shape.length == 4) qtensor.shape(0) else -1
  val numChannels:     Int = if (qtensor.shape.length == 4) qtensor.shape(1) else -1
  val height:          Int = if (qtensor.shape.length == 4) qtensor.shape(2) else -1
  val width:           Int = if (qtensor.shape.length == 4) qtensor.shape(3) else -1
  val numParams:       Int = qtensor.shape.reduce(_ * _)
  val numKernelParams: Int = numChannels * height * width
  val paramBitwidth:   Int = qtensor.dtype.get.bitwidth
  val mem                  = MemoryConfig(qtensor,
                                          numKernelParams = numKernelParams,
                                          paramBitwidth = paramBitwidth,
                                          numKernels = numKernels)
}

case class ThreshConfig(qtensor: QTensor) {
  val numParams:       Int = qtensor.shape.reduce(_ * _)
  val numKernels:      Int = qtensor.shape(0)
  val numKernelParams: Int = numParams / numKernels
  val paramBitwidth:   Int = qtensor.dtype.get.bitwidth
  val mem                  = MemoryConfig(qtensor,
                                          numKernelParams = numKernelParams,
                                          paramBitwidth = paramBitwidth,
                                          numKernels = numKernels)
}

case class MemoryConfig(qtensor: QTensor, numKernelParams: Int, paramBitwidth: Int, numKernels: Int) {
  val paramsPerWord: Int = MemWordSize.bits / paramBitwidth
  val depth:         Int = math.ceil(numKernelParams.toFloat / paramsPerWord.toFloat).toInt * numKernels
}

case object MemWordSize {
  val bits = 32
}

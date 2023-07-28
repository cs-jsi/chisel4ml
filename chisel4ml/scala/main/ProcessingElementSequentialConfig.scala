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

case class ProcessingElementSequentialConfigConv(layer: Layer) {
  val kernel = TensorConfig(layer.weights.get)
  val thresh = ThreshConfig(layer.thresh.get)
  val input  = TensorConfig(layer.input.get)
  val result = TensorConfig(layer.output.get)

  require(kernel.numChannels == input.numChannels,
      s"""Number of channels of the kernels should be the same as the number of channels of the input.
      | Instead kernel ${kernel.numChannels} != input ${input.numChannels}. Kernel shape: ${kernel.shape},
      | input shape: ${input.shape}.""".stripMargin.replaceAll("\n",""))
  require(kernel.numKernels == result.numChannels,
      s"""Number of kernels in the kernel should be equal to the number of channels in the result. Instead the kernel
      | has ${kernel.numKernels} kernels, result ${result.numChannels} channels. Kernel shape: ${kernel.shape},
      | result shape: ${result.shape}.""".stripMargin.replaceAll("\n", ""))
}

case class ProcessingElementSequentialConfigMaxPool(layer: Layer) {
  val input  = TensorConfig(layer.input.get)
  val result = TensorConfig(layer.output.get)
}

case class TensorConfig(qtensor: QTensor) {
  val width:           Int = qtensor.shape.last
  val height:          Int = qtensor.shape.reverse(1)
  val numChannels:     Int = if (qtensor.shape.length >= 3) qtensor.shape.reverse(2) else 1
  val numKernels:      Int = if (qtensor.shape.length == 4) qtensor.shape(0) else 1

  val shape                = (numKernels, numChannels, height, width)
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

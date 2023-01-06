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

import scala.reflect.runtime.{universe => ru}

import chisel3._
import chisel3.util._

import _root_.chisel4ml.bus.AXIStream
import _root_.chisel4ml.memory.{ROM, SRAM}
import _root_.chisel4ml.util._
import _root_.chisel4ml.lbir._
import _root_.chisel4ml.implicits._
import _root_.lbir.Layer
import _root_.services.GenerateCircuitParams.Options
import _root_.scala.math

/** A sequential processing element for convolutions.
  *
  * This hardware module can handle two-dimensional convolutions of various types, and also can adjust the aritmetic
  * units depending on the quantization type. It does not take advantage of sparsity. It uses the filter stationary
  * approach and streams in the activations for each filter sequentialy. The compute unit computes one whole neuron at
  * once. The reason for this is that it simplifies the design, which would otherwise require complex control logic /
  * program code. This design, of course, comes at a price of utilization of the arithmetic units, which is low. But
  * thanks to the low bitwidths of parameters this should be an acceptable trade-off.
  */
class ProcessingElementSequentialConv[
    I <: Bits,
    W <: Bits: WeightsProvider,
    M <: Bits,
    A <: Bits: ThreshProvider,
    O <: Bits
  ](layer:      Layer,
    options:    Options,
    genIn:      I,
    genWeights: W,
    genThresh:  A,
    genOut:     O,
    mul:        (I, W) => M,
    add:        Vec[M] => A,
    actFn:      (A, A) => O)
    extends ProcessingElementSequential(layer, options) {
    val kernelParamSize:     Int = layer.weights.get.dtype.get.bitwidth
    val kernelParamsPerWord: Int = memWordWidth / kernelParamSize
    val kernelNumParams:     Int = layer.weights.get.shape.reduce(_ * _)
    val kernelMemDepth:      Int = math.ceil(kernelNumParams.toFloat / kernelParamsPerWord.toFloat).toInt
    val kernelMem = Module(
      new ROM(
        depth = kernelMemDepth,
        width = memWordWidth,
        memFile = genHexMemoryFile(layer.weights.get, layout = "CDHW")
      )
    )

    val actParamSize:     Int = layer.input.get.dtype.get.bitwidth
    val actParamsPerWord: Int = memWordWidth / actParamSize
    val actNumParams:     Int = layer.input.get.shape.reduce(_ * _)
    val actMemDepth:      Int = math.ceil(actNumParams.toFloat / actParamsPerWord.toFloat).toInt
    val actMem = Module(new SRAM(depth = actMemDepth, width = memWordWidth))

    val numOfKernels: Int = layer.weights.get.shape(0)
    val kernelDepth:  Int = layer.weights.get.shape(1)
    val kernelSize:   Int = layer.weights.get.shape(2)
    val kernelRegFile = Module(new KernelRegisterFile(kernelSize, kernelDepth, kernelParamSize))

    val actRegFile = Module(new RollingRegisterFile(kernelSize, kernelDepth, kernelParamSize))

    val resParamSize:     Int = layer.output.get.dtype.get.bitwidth
    val resParamsPerWord: Int = memWordWidth / resParamSize
    val resNumParams:     Int = layer.output.get.shape.reduce(_ * _)
    val resMemDepth:      Int = math.ceil(resNumParams.toFloat / resParamsPerWord.toFloat).toInt
    val resMem = Module(new SRAM(depth = resMemDepth, width = memWordWidth))

    val dynamicNeuron = Module(
      new DynamicNeuron[I, W, M, A, O](
        genIn = genIn,
        numSynaps = kernelNumParams,
        genWeights = genWeights,
        genThresh = genThresh,
        genOut = genOut,
        mul = mul,
        add = add,
        actFn = actFn
      )
    )

    // val swu = Module(new SlidingWindowUnit())
}

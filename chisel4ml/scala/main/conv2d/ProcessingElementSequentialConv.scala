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
package chisel4ml.conv2d

import chisel4ml._
import chisel4ml.implicits._
import lbir.Activation._
import lbir.Conv2DConfig
import lbir.Datatype.QuantizationType._
import org.slf4j.LoggerFactory
import services.LayerOptions
import chisel3._
import interfaces.amba.axis._
import chisel4ml.QuantizationContext
import chisel4ml.UniformQuantizationContextSSSNoAct

/** A sequential processing element for convolutions.
  *
  * This hardware module can handle two-dimensional convolutions of various types, and also can adjust the aritmetic
  * units depending on the quantization type. It does not take advantage of sparsity. It uses the filter stationary
  * approach and streams in the activations for each filter sequentialy. The compute unit computes one whole neuron at
  * once. The reason for this is that it simplifies the design, which would otherwise require complex control logic /
  * program code. This design, of course, comes at a price of utilization of the arithmetic units, which is low. But
  * thanks to the low bitwidths of parameters this should be an acceptable trade-off.
  */

class ProcessingElementSequentialConv[I <: Bits, W <: Bits, M <: Bits, A <: Bits, O <: Bits](
  layer:   Conv2DConfig,
  options: LayerOptions
)(qc:      QuantizationContext[I, W, M, A, O])
    extends Module
    with LBIRStream {
  val logger = LoggerFactory.getLogger("ProcessingElementSequentialConv")

  logger.info(
    s"""Generated new depthwise: ${layer.depthwise}: ProcessingElementSequentialConv with input shape:${layer.input.shape}, input dtype:
          | ${layer.input.dtype}. Number of kernel parameters is ${layer.kernel.numKernelParams}."""
  )

  val inStream = IO(Flipped(AXIStream(UInt(options.busWidthIn.W))))
  val outStream = IO(AXIStream(UInt(options.busWidthOut.W)))

  val dynamicNeuron = Module(new DynamicNeuron(layer, qc))
  val ctrl = Module(new PeSeqConvController(layer))
  val kernelSubsystem = Module(new KernelSubsystem(layer))
  val inputSubsytem = Module(new InputActivationsSubsystem[I](layer, options))
  val rmb = Module(new ResultMemoryBuffer(layer.output, options, layer.output.getType[O]))

  inputSubsytem.io.inStream <> inStream
  dynamicNeuron.io.in <> inputSubsytem.io.inputActivationsWindow
  dynamicNeuron.io.weights <> kernelSubsystem.io.weights
  rmb.io.result <> dynamicNeuron.io.out
  outStream <> rmb.io.outStream

  ctrl.io.activeDone := inputSubsytem.io.activeDone
  kernelSubsystem.io.ctrl <> ctrl.io.kernelCtrl
}

object ProcessingElementSequentialConv {
  def apply(layer: Conv2DConfig, options: LayerOptions) = (
    layer.input.dtype.quantization,
    layer.input.dtype.signed,
    layer.kernel.dtype.quantization,
    layer.activation
  ) match {
    case (UNIFORM, true, UNIFORM, RELU) =>
      new ProcessingElementSequentialConv[SInt, SInt, SInt, SInt, UInt](layer, options)(
        UniformQuantizationContextSSUReLU
      )
    case (UNIFORM, false, UNIFORM, RELU) =>
      new ProcessingElementSequentialConv[UInt, SInt, SInt, SInt, UInt](layer, options)(
        UniformQuantizationContextUSUReLU
      )
    case (UNIFORM, true, UNIFORM, NO_ACTIVATION) =>
      new ProcessingElementSequentialConv[SInt, SInt, SInt, SInt, SInt](layer, options)(
        UniformQuantizationContextSSSNoAct
      )
    case (UNIFORM, false, UNIFORM, NO_ACTIVATION) =>
      new ProcessingElementSequentialConv[UInt, SInt, SInt, SInt, SInt](layer, options)(
        UniformQuantizationContextUSSNoAct
      )
    case _ => throw new RuntimeException()
  }
}

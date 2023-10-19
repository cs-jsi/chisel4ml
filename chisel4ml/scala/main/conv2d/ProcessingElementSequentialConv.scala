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

import chisel4ml.LBIRStream
import chisel4ml.implicits._
import chisel4ml.util.{linFn, reluFn}
import lbir.Activation._
import lbir.Conv2DConfig
import lbir.Datatype.QuantizationType._
import org.slf4j.LoggerFactory
import services.LayerOptions
import chisel3._
import interfaces.amba.axis._

import scala.reflect.runtime.universe._

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
  I <: Bits with Num[I]: TypeTag,
  W <: Bits with Num[W]: TypeTag,
  M <: Bits,
  S <: Bits: TypeTag,
  A <: Bits: TypeTag,
  O <: Bits: TypeTag
](layer:   Conv2DConfig,
  options: LayerOptions,
  mul:     (I, W) => M,
  add:     Vec[M] => S,
  actFn:   (S, A) => O)
    extends Module
    with LBIRStream {
  val logger = LoggerFactory.getLogger("ProcessingElementSequentialConv")

  def genType[T <: Bits: TypeTag](bitwidth: Int): T = {
    val tpe = implicitly[TypeTag[T]].tpe
    val hwType =
      if (tpe =:= typeOf[UInt]) UInt(bitwidth.W)
      else if (tpe =:= typeOf[SInt]) SInt(bitwidth.W)
      else throw new NotImplementedError
    hwType.asInstanceOf[T]
  }

  val genIn = genType[I](layer.input.dtype.bitwidth)
  val genWeights = genType[W](layer.kernel.dtype.bitwidth)
  val genAccu = genType[S](layer.input.dtype.bitwidth + layer.kernel.dtype.bitwidth)
  val genThresh = genType[A](layer.thresh.dtype.bitwidth)
  val genOut = genType[O](layer.output.dtype.bitwidth)

  logger.info(s"""Generated new ProcessingElementSequentialConv with input shape:${layer.input.shape}, input dtype:
          | ${layer.input.dtype}. Number of kernel parameters is ${layer.kernel.numKernelParams}.""")

  val inStream = IO(Flipped(AXIStream(UInt(options.busWidthIn.W))))
  val outStream = IO(AXIStream(UInt(options.busWidthOut.W)))

  val dynamicNeuron = Module(
    new DynamicNeuron(
      kernel = layer.kernel,
      genIn = genIn,
      genWeights = genWeights,
      genAccu = genAccu,
      genThresh = genThresh,
      genOut = genOut,
      mul = mul,
      add = add,
      actFn = actFn
    )
  )
  val ctrl = Module(new PeSeqConvController(layer))
  val kernelSubsystem = Module(new KernelSubsystem(layer.kernel, layer.thresh, genThresh))
  val inputSubsytem = Module(new InputActivationsSubsystem(layer.input, layer.kernel, layer.output, options))
  val resultSubsystem = Module(new ResultSubsystem(layer.output, options, genOut))

  inputSubsytem.io.inStream <> inStream
  dynamicNeuron.io.in <> inputSubsytem.io.inputActivationsWindow
  dynamicNeuron.io.weights <> kernelSubsystem.io.weights
  resultSubsystem.io.result <> dynamicNeuron.io.out
  outStream <> resultSubsystem.io.outStream

  kernelSubsystem.io.loadKernel := ctrl.io.loadKernel
}

object ProcessingElementSequentialConv {
  def apply(layer: Conv2DConfig, options: LayerOptions) = (
    layer.input.dtype.quantization,
    layer.input.dtype.signed,
    layer.kernel.dtype.quantization,
    layer.activation
  ) match {
    case (UNIFORM, true, UNIFORM, RELU) =>
      new ProcessingElementSequentialConv[SInt, SInt, SInt, SInt, SInt, UInt](
        layer,
        options,
        mul = (x: SInt, y: SInt) => x * y,
        add = (x: Vec[SInt]) => x.reduceTree(_ +& _),
        actFn = reluFn
      )
    case (UNIFORM, false, UNIFORM, RELU) =>
      new ProcessingElementSequentialConv[UInt, SInt, SInt, SInt, SInt, UInt](
        layer,
        options,
        mul = (x: UInt, y: SInt) => x * y,
        add = (x: Vec[SInt]) => x.reduceTree(_ +& _),
        actFn = reluFn
      )
    case (UNIFORM, true, UNIFORM, NO_ACTIVATION) =>
      new ProcessingElementSequentialConv[SInt, SInt, SInt, SInt, SInt, SInt](
        layer,
        options,
        mul = (x: SInt, y: SInt) => x * y,
        add = (x: Vec[SInt]) => x.reduceTree(_ +& _),
        actFn = linFn
      )
    case _ => throw new RuntimeException()
  }
}

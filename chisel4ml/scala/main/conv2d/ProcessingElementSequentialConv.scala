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
import chisel3._
import interfaces.amba.axis._
import chisel4ml.quantization._
import org.chipsalliance.cde.config.Parameters
import spire.implicits._
import dsptools.numbers._
import org.chipsalliance.cde.config.{Config, Field}
import chisel4ml.logging.HasLogger
import chisel4ml.logging.HasParameterLogging

/** A sequential processing element for convolutions.
  *
  * This hardware module can handle two-dimensional convolutions of various types, and also can adjust the aritmetic
  * units depending on the quantization type. It does not take advantage of sparsity. It uses the filter stationary
  * approach and streams in the activations for each filter sequentialy. The compute unit computes one whole neuron at
  * once. The reason for this is that it simplifies the design, which would otherwise require complex control logic /
  * program code. This design, of course, comes at a price of utilization of the arithmetic units, which is low. But
  * thanks to the low bitwidths of parameters this should be an acceptable trade-off.
  */

case object Conv2DConfigField extends Field[Conv2DConfig]

trait HasSequentialConvParameters extends HasLBIRStreamParameters[Conv2DConfig] {
  val p: Parameters
  override val cfg = p(Conv2DConfigField)
}

class ProcessingElementSequentialConv[I <: Bits, W <: Bits, M <: Bits, A <: Bits: Ring, O <: Bits](
  qc: QuantizationContext[I, W, M, A, O]
)(
  implicit val p: Parameters)
    extends Module
    with HasLBIRStream[Vec[UInt]]
    with HasLBIRStreamParameters[Conv2DConfig]
    with HasLBIRConfig[Conv2DConfig]
    with HasSequentialConvParameters
    with HasLogger
    with HasParameterLogging {
  logParameters

  val inStream = IO(Flipped(AXIStream(Vec(numBeatsIn, UInt(cfg.input.dtype.bitwidth.W)))))
  val outStream = IO(AXIStream(Vec(numBeatsOut, UInt(cfg.output.dtype.bitwidth.W))))

  val dynamicNeuron = Module(new DynamicNeuron[I, W, M, A, O](cfg, qc))
  val ctrl = Module(new PeSeqConvController(cfg))
  val kernelSubsystem = Module(new KernelSubsystem[W, A](cfg))
  val inputSubsytem = Module(new InputActivationsSubsystem[I])
  val rmb = Module(new ResultMemoryBuffer[O])

  inputSubsytem.io.inStream <> inStream
  dynamicNeuron.io.in <> inputSubsytem.io.inputActivationsWindow
  dynamicNeuron.io.weights <> kernelSubsystem.io.weights
  rmb.io.result <> dynamicNeuron.io.out
  outStream <> rmb.io.outStream

  ctrl.io.activeDone := inputSubsytem.io.activeDone
  kernelSubsystem.io.ctrl <> ctrl.io.kernelCtrl
}

object ProcessingElementSequentialConv {
  def apply(cfg: Conv2DConfig) = {
    implicit val p: Parameters = new Config((_, _, _) => {
      case Conv2DConfigField => cfg
      case LBIRNumBeatsIn    => 4
      case LBIRNumBeatsOut   => 4
    })
    (cfg.input.dtype.quantization, cfg.input.dtype.signed, cfg.kernel.dtype.quantization, cfg.activation) match {
      case (UNIFORM, true, UNIFORM, RELU) =>
        new ProcessingElementSequentialConv[SInt, SInt, SInt, SInt, UInt](
          new UniformQuantizationContextSSUReLU(cfg.output.roundingMode)
        )
      case (UNIFORM, false, UNIFORM, RELU) =>
        new ProcessingElementSequentialConv[UInt, SInt, SInt, SInt, UInt](
          new UniformQuantizationContextUSUReLU(cfg.output.roundingMode)
        )
      case (UNIFORM, true, UNIFORM, NO_ACTIVATION) =>
        new ProcessingElementSequentialConv[SInt, SInt, SInt, SInt, SInt](
          new UniformQuantizationContextSSSNoAct(cfg.output.roundingMode)
        )
      case (UNIFORM, false, UNIFORM, NO_ACTIVATION) =>
        new ProcessingElementSequentialConv[UInt, SInt, SInt, SInt, SInt](
          new UniformQuantizationContextUSSNoAct(cfg.output.roundingMode)
        )
      case _ => throw new RuntimeException()
    }
  }
}

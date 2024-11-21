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

import chisel3._
import chisel4ml._
import chisel4ml.compute.NeuronCompute
import chisel4ml.logging.{HasLogger, HasParameterLogging}
import interfaces.amba.axis._
import lbir.{Conv2DConfig, LayerWrap}
import org.chipsalliance.cde.config.Parameters
import services.Accelerator

/** A sequential processing element for convolutions.
  *
  * This hardware module can handle two-dimensional convolutions of various types, and also can adjust the aritmetic
  * units depending on the quantization type. It does not take advantage of sparsity. It uses the filter stationary
  * approach and streams in the activations for each filter sequentialy. The compute unit computes one whole neuron at
  * once. The reason for this is that it simplifies the design, which would otherwise require complex control logic /
  * program code. This design, of course, comes at a price of utilization of the arithmetic units, which is low. But
  * thanks to the low bitwidths of parameters this should be an acceptable trade-off.
  */

trait HasSequentialConvParameters extends HasAXIStreamParameters with HasLayerWrapSeq {
  val p: Parameters
  val cfg: Conv2DConfig = LayerWrap.LayerWrapTypeMapper.toCustom(_cfg.head.asMessage).get.asInstanceOf[Conv2DConfig]
  require(_cfg.length == 1)
}

class ProcessingElementSequentialConv(
  val nc: NeuronCompute
)(
  implicit val p: Parameters)
    extends Module
    with HasAXIStream
    with HasAXIStreamParameters
    with HasSequentialConvParameters
    with HasLogger
    with HasParameterLogging {
  logParameters

  val inStream = IO(Flipped(AXIStream(nc.genI, numBeatsIn)))
  val outStream = IO(AXIStream(nc.genO, numBeatsOut))

  val dynamicNeuron = Module(new DynamicNeuron(nc)(cfg))
  val ctrl = Module(new PeSeqConvController(cfg))
  val kernelSubsystem = Module(new KernelSubsystem[nc.W, nc.A](cfg))
  val inputSubsytem = Module(new InputActivationsSubsystem[nc.I])
  val rmb = Module(new ResultMemoryBuffer[nc.O])

  inputSubsytem.io.inStream <> inStream
  dynamicNeuron.io.in <> inputSubsytem.io.inputActivationsWindow
  dynamicNeuron.io.weights <> kernelSubsystem.io.weights
  rmb.io.result <> dynamicNeuron.io.out
  outStream <> rmb.io.outStream

  ctrl.io.activeDone := inputSubsytem.io.activeDone
  kernelSubsystem.io.ctrl <> ctrl.io.kernelCtrl
}

object ProcessingElementSequentialConv {
  def apply(
    accel: Accelerator
  )(
    implicit p: Parameters
  ): Module with HasAXIStream = {
    require(accel.name == "ProcessingElementSequentialConv")
    require(accel.layers.length == 1)
    accel.layers.head.get match {
      case l: Conv2DConfig => new ProcessingElementSequentialConv(NeuronCompute(l))
      case _ => throw new RuntimeException
    }
  }
}

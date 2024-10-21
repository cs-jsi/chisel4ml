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
import chisel4ml.conv2d.ProcessingElementSequentialConv
import chisel4ml.quantization._
import chisel4ml.{HasLBIRStream, MaxPool2D}
import lbir.Datatype.QuantizationType.{BINARY, UNIFORM}
import lbir.{HasInputOutputQTensor, IsActiveLayer, LayerWrap}
import org.chipsalliance.cde.config.Parameters
import services.Accelerator

object AcceleratorGenerator {
  def apply(accel: Accelerator): Module with HasLBIRStream = {
    implicit val defaults: Parameters = Parameters.empty.alterPartial({
      case LayerWrapIOField => accel.layers.map(_.get).map((_, this.getIOContext(_)))
    })
    (accel.name, accel.layers) match {
      case ("MaxPool2D", _)   => Module(new MaxPool2D())
      case ("FFTWrapper", _)  => Module(new FFTWrapper())
      case ("LMFEWrapper", _) => Module(new LMFEWrapper())
      case (accelName, accelLayers: Seq[Option[LayerWrap with HasInputOutputQTensor with IsActiveLayer]]) => {
        val qcList = accelLayers.map(_.get).map(this.getQuantizationContext(_))
        accelName match {
          case "ProcessingElementSequentialConv" => Module(new ProcessingElementSequentialConv(qcList.head))
          case "ProcessingElementWrapSimpleToSequential" =>
            Module(
              new ProcessingElementCombToSeq(
                accel.layers.head.get.input,
                accel.layers.last.get.output,
                new ProcessingPipelineSimple(accel.layers.map(_.get))
              )
            )
          case _ => throw new RuntimeException
        }
      }
      case _ => throw new RuntimeException
    }
  }
  def getIOContext(l: LayerWrap with HasInputOutputQTensor): QuantizationContext =
    (
      l.input.dtype.quantization,
      l.input.dtype.signed,
      l.output.dtype.quantization,
      l.output.dtype.signed
    ) match {
      case (BINARY, _, BINARY, _)      => BinarizedQuantizationContext
      case (UNIFORM, false, BINARY, _) => new BinaryQuantizationContext(l.output.roundingMode)
      case (UNIFORM, true, BINARY, _)  => new BinaryQuantizationContextSInt(l.output.roundingMode)
      case (UNIFORM, true, UNIFORM, false) =>
        new UniformQuantizationContextSSU(lbir.Activation.NO_ACTIVATION, l.output.dtype.bitwidth, l.output.roundingMode)
      case (UNIFORM, false, UNIFORM, false) =>
        new UniformQuantizationContextUSU(lbir.Activation.NO_ACTIVATION, l.output.dtype.bitwidth, l.output.roundingMode)
      case (UNIFORM, false, UNIFORM, true) =>
        new UniformQuantizationContextUSS(lbir.Activation.NO_ACTIVATION, l.output.dtype.bitwidth, l.output.roundingMode)
      case (UNIFORM, true, UNIFORM, true) =>
        new UniformQuantizationContextSSS(lbir.Activation.NO_ACTIVATION, l.output.dtype.bitwidth, l.output.roundingMode)
      case _ => throw new RuntimeException
    }

  def getQuantizationContext(l: LayerWrap with IsActiveLayer): QuantizationContext =
    (
      l.input.dtype.quantization,
      l.input.dtype.signed,
      l.kernel.dtype.quantization,
      l.kernel.dtype.signed,
      l.output.dtype.quantization,
      l.output.dtype.signed
    ) match {
      case (BINARY, _, BINARY, _, BINARY, _)      => BinarizedQuantizationContext
      case (UNIFORM, false, BINARY, _, BINARY, _) => new BinaryQuantizationContext(l.output.roundingMode)
      case (UNIFORM, true, BINARY, _, BINARY, _)  => new BinaryQuantizationContextSInt(l.output.roundingMode)
      case (UNIFORM, true, UNIFORM, true, UNIFORM, false) =>
        new UniformQuantizationContextSSU(l.activation, l.output.dtype.bitwidth, l.output.roundingMode)
      case (UNIFORM, false, UNIFORM, true, UNIFORM, false) =>
        new UniformQuantizationContextUSU(l.activation, l.output.dtype.bitwidth, l.output.roundingMode)
      case (UNIFORM, true, UNIFORM, true, UNIFORM, true) =>
        new UniformQuantizationContextSSS(l.activation, l.output.dtype.bitwidth, l.output.roundingMode)
      case (UNIFORM, false, UNIFORM, true, UNIFORM, true) =>
        new UniformQuantizationContextUSS(l.activation, l.output.dtype.bitwidth, l.output.roundingMode)
      case _ => throw new RuntimeException
    }
}

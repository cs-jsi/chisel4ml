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

import chisel4ml.{HasLBIRStream, LBIRNumBeatsIn, LBIRNumBeatsOut}
import chisel4ml.conv2d.ProcessingElementSequentialConv
import chisel4ml.{MaxPool2D, MaxPool2DConfigField}
import lbir.{Conv2DConfig, DenseConfig, FFTConfig, LMFEConfig, LayerWrap, MaxPool2DConfig}
import lbir.Datatype.QuantizationType.UNIFORM
import lbir.Datatype.QuantizationType.BINARY
import chisel3._
import org.chipsalliance.cde.config.{Config, Parameters}
import chisel4ml.quantization._
import chisel4ml.conv2d.Conv2DConfigField
import lbir.IsActiveLayer

object LayerGenerator {
  def apply(layerWrap: LayerWrap): Module with HasLBIRStream = {
    implicit val defaults: Parameters = new Config((_, _, _) => {
      case LBIRNumBeatsIn  => 4
      case LBIRNumBeatsOut => 4
    })
    layerWrap match {
      case l: MaxPool2DConfig =>
        Module(new MaxPool2D()(defaults.alterPartial({
          case MaxPool2DConfigField => l
        })))
      case l: FFTConfig =>
        Module(new FFTWrapper()(defaults.alterPartial({
          case FFTConfigField  => l
          case LBIRNumBeatsIn  => 1
          case LBIRNumBeatsOut => 1
        })))
      case l: LMFEConfig =>
        Module(new LMFEWrapper()(defaults.alterPartial({
          case LMFEConfigField => l
          case LBIRNumBeatsIn  => 1
        })))
      case l: IsActiveLayer =>
        l match {
          case l: DenseConfig =>
            Module(new ProcessingElementWrapSimpleToSequential(this.getQuantizationContext(l))(defaults.alterPartial({
              case DenseConfigField => l
            })))
          case l: Conv2DConfig =>
            Module(new ProcessingElementSequentialConv(this.getQuantizationContext(l))(defaults.alterPartial({
              case Conv2DConfigField => l
            })))
          case _ => throw new RuntimeException
        }
      case _ => throw new RuntimeException(f"Unsupported layer type: $layerWrap")
    }
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

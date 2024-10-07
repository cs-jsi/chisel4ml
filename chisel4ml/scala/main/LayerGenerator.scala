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
import chisel3._
import org.chipsalliance.cde.config.{Config, Parameters}

object LayerGenerator {
  def apply(layerWrap: LayerWrap): Module with HasLBIRStream = {
    implicit val defaults: Parameters = new Config((_, _, _) => {
      case LBIRNumBeatsIn  => 4
      case LBIRNumBeatsOut => 4
    })
    /*val qc = (layerWrap.input.dtype.quantization, layerWrap.input.dtype.signed, layerWrap.output.dtype.quantization, layerWrap.output.dtype.signed) match {
      case (UNIFORM, true, UNIFORM, false) => new UniformQuantizationContextSSUReLU(layerWrap.output.roundingMode)
      case (UNIFORM, false, UNIFORM, false) =>new UniformQuantizationContextUSUReLU(layerWrap.output.roundingMode)
      case (UNIFORM, true, UNIFORM, true) => new UniformQuantizationContextSSSNoAct(layerWrap.output.roundingMode)
      case (UNIFORM, false, UNIFORM, true) => new UniformQuantizationContextUSSNoAct(layerWrap.output.roundingMode)
      case _ => throw new RuntimeException()
    }*/

    layerWrap match {
      case l: DenseConfig =>
        Module(new ProcessingElementWrapSimpleToSequential()(defaults.alterPartial({
          case DenseConfigField => l
        })))
      case l: Conv2DConfig => Module(ProcessingElementSequentialConv(l))
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
      case _ => throw new RuntimeException(f"Unsupported layer type: $layerWrap")
    }

  }
}

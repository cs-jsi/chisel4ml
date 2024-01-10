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
import chisel4ml.sequential.MaxPool2D
import lbir.{Conv2DConfig, DenseConfig, FFTConfig, LMFEConfig, LayerWrap, MaxPool2DConfig}
import chisel3._
import org.chipsalliance.cde.config.{Config, Parameters, Field}

case object SupportsMultipleBeats extends Field[Boolean](true)

object LayerGenerator {
  def apply(layerWrap: LayerWrap): Module with HasLBIRStream = {
    implicit val defaults: Parameters = new Config((site, _, _) => {
      case LayerWrap => layerWrap
      case LBIRNumBeatsIn => if (site(SupportsMultipleBeats) == true) 4 else 1
      case LBIRNumBeatsOut => if (site(SupportsMultipleBeats) == true) 4 else 1
    })
    layerWrap match {
      case _: DenseConfig => Module(new ProcessingElementWrapSimpleToSequential)
      case l: Conv2DConfig => Module(ProcessingElementSequentialConv(l))
      case _: MaxPool2DConfig => Module(new MaxPool2D)
      case _: FFTConfig => Module(new FFTWrapper()(defaults.alterPartial({case SupportsMultipleBeats => false})))
      case _: LMFEConfig => Module(new LMFEWrapper()(defaults.alterPartial({case SupportsMultipleBeats => false})))
      case _ => throw new RuntimeException(f"Unsupported layer type: $layerWrap")
    } 

  }
}

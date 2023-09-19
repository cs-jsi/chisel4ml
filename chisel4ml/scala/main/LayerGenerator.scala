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
import _root_.lbir.LayerWrap
import _root_.lbir.{Conv2DConfig, DenseConfig, FFTConfig, LMFEConfig, MaxPool2DConfig}
import _root_.services.LayerOptions
import _root_.chisel4ml.LBIRStream
import _root_.chisel4ml.sequential.{MaxPool2D, ProcessingElementSequentialConv}

object LayerGenerator {
  // TODO: Rewrite the generation procedure to something more sensisble
  def apply(layer_wrap: LayerWrap, options: LayerOptions): Module with LBIRStream = {
    layer_wrap match {
      case l: DenseConfig     => Module(new ProcessingElementWrapSimpleToSequential(l, options))
      case l: Conv2DConfig    => Module(ProcessingElementSequentialConv(l, options))
      case l: MaxPool2DConfig => Module(new MaxPool2D(l, options))
      case l: FFTConfig       => Module(new FFTWrapper(l, options))
      case l: LMFEConfig      => Module(new LMFEWrapper(l, options))
      case _ => throw new RuntimeException(f"Unsupported layer type: $layer_wrap")
    }

  }
}

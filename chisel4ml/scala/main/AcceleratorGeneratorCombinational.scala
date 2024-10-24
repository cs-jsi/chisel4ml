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
import chisel4ml.AcceleratorGenerator.{getIOContext, getQuantizationContext}
import chisel4ml.quantization._
import lbir.{HasInputOutputQTensor, IsActiveLayer, LayerWrap, MaxPool2DConfig}

object AcceleratorGeneratorCombinational {
  def apply(layer: LayerWrap with HasInputOutputQTensor): Module with LBIRStreamSimple = layer match {
    case l: MaxPool2DConfig =>
      Module(new ProcessingElementCombinationalIO(getIOContext(Seq(l)))(l, MaximumTransformationIO))
    case l: IsActiveLayer => {
      val qc = getQuantizationContext(l)
      qc match {
        case BinarizedQuantizationContext => Module(new ProcessingElementCombinational(qc)(l, NeuronWithoutBias))
        case _                            => Module(new ProcessingElementCombinational(qc)(l, NeuronWithBias))
      }
    }
    case _ => throw new RuntimeException(f"Unsupported layer type")
  }
}

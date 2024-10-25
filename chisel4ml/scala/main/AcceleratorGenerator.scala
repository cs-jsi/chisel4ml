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
import chisel4ml.HasAXIStream
import chisel4ml.sequential.{FFTWrapper, MaxPool2D, ProcessingElementSequentialConv}
import org.chipsalliance.cde.config.Parameters
import services.Accelerator

object AcceleratorGenerator {
  def apply(accel: Accelerator): Module with HasAXIStream = {
    implicit val defaults: Parameters = Parameters.empty.alterPartial({
      case LayerWrapSeqField => accel.layers.map(_.get)
    })
    accel.name match {
      case "MaxPool2D"                       => Module(MaxPool2D(accel))
      case "FFTWrapper"                      => Module(new FFTWrapper)
      case "LMFEWrapper"                     => Module(new LMFEWrapper)
      case "ProcessingElementSequentialConv" => Module(ProcessingElementSequentialConv(accel))
      case "ProcessingElementCombToSeq"      => Module(ProcessingElementCombToSeq(accel))
    }
  }
}

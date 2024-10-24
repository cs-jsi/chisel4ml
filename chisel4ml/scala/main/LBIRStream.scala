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
import chisel4ml.quantization.IOContext
import interfaces.amba.axis._
import lbir.{IsActiveLayer, LayerWrap}
import org.chipsalliance.cde.config.{Field, Parameters}

case object NumBeatsInField extends Field[Int](default = 4)
case object NumBeatsOutField extends Field[Int](default = 4)
case object LayerWrapSeqField extends Field[Seq[LayerWrap]]()
case object IOContextField extends Field[IOContext]()

trait HasLBIRStreamParameters {
  val p: Parameters
  val numBeatsIn = p(NumBeatsInField)
  val numBeatsOut = p(NumBeatsOutField)
  require(numBeatsIn > 0)
  require(numBeatsOut > 0)
  val _cfg = p(LayerWrapSeqField)
}
trait HasIOContext extends HasLBIRStreamParameters {
  val p: Parameters
  val ioc = p(IOContextField)
}
trait HasActiveLayer {
  val cfg: LayerWrap with IsActiveLayer
}
trait HasQuantizationContext extends HasIOContext with HasActiveLayer {
  val p: Parameters
  val qc = AcceleratorGenerator.getQuantizationContext(cfg)
}

trait HasLBIRStream {
  val inStream:  AXIStreamIO[Data]
  val outStream: AXIStreamIO[Data]
}

trait LBIRStreamSimple {
  val in:  Vec[Data]
  val out: Vec[Data]
}

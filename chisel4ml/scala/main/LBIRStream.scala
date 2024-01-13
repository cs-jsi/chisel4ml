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
import interfaces.amba.axis._
import org.chipsalliance.cde.config.{Field, Parameters}
import lbir.LayerWrap

case object LBIRNumBeatsIn extends Field[Int]
case object LBIRNumBeatsOut extends Field[Int]

trait HasLBIRConfig[+T <: LayerWrap] {
  val cfg: T
}

trait HasLBIRStreamParameters[T <: LayerWrap] extends HasLBIRConfig[T] {
  val p: Parameters
  val numBeatsIn = p(LBIRNumBeatsIn)
  val numBeatsOut = p(LBIRNumBeatsOut)
  def inWidth = numBeatsIn * cfg.input.dtype.bitwidth
  def outWidth = numBeatsOut * cfg.output.dtype.bitwidth
  require(numBeatsIn > 0)
  require(numBeatsOut > 0)
}

trait HasLBIRStream[T <: Data] {
  val inStream:  AXIStreamIO[T]
  val outStream: AXIStreamIO[T]
}

trait LBIRStreamSimple {
  val in:  Vec[Bits]
  val out: Vec[Bits]
}

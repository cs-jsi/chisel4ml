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

case object LBIRStreamWidthIn extends Field[Int]
case object LBIRStreamWidthOut extends Field[Int]

trait HasLBIRStreamParameters {
  val p: Parameters
  val inWidth = p(LBIRStreamWidthIn)
  val outWidth = p(LBIRStreamWidthOut)
  require(inWidth > 0)
  require(outWidth > 0)
}

trait LBIRStream {
  val inStream:  AXIStreamIO[UInt]
  val outStream: AXIStreamIO[UInt]
}

trait LBIRStreamSimple {
  val in:  Vec[Bits]
  val out: Vec[Bits]
}

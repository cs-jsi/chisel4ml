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
package chisel4ml.compute

import chisel3._
import lbir.Datatype.QuantizationType.{BINARY, UNIFORM}
import lbir.MaxPool2DConfig

object OrderCompute {
  def apply(l: MaxPool2DConfig): OrderCompute = (l.input.dtype.quantization, l.input.dtype.signed) match {
    case (BINARY, _)      => BoolOrderComputable
    case (UNIFORM, true)  => new SIntOrderComputable(l.input.dtype.bitwidth)
    case (UNIFORM, false) => new UIntOrderComputable(l.input.dtype.bitwidth)
    case _                => throw new RuntimeException
  }
}

abstract class OrderCompute {
  type T <: Data
  def gte:  (T, T) => Bool
  def genT: T
}

class UIntOrderComputable(val bitwidth: Int) extends OrderCompute {
  type T = UInt
  def gte = (x: UInt, y: UInt) => x > y
  def genT = UInt(bitwidth.W)
}

class SIntOrderComputable(val bitwidth: Int) extends OrderCompute {
  type T = SInt
  def gte = (x: SInt, y: SInt) => x > y
  def genT = SInt(bitwidth.W)
}

object BoolOrderComputable extends OrderCompute {
  type T = Bool
  def gte = (x: Bool, _: Bool) => x
  def genT = Bool()
}

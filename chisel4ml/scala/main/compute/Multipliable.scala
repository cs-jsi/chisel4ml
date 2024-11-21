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

/* Defines a trait of multipliable type combinations
 */
trait Multipliable[I <: Bits, W <: Bits, M <: Bits] {
  def mul: (I, W) => M
}
package MultipliableImplementations {
  // UInt, SInt, Bool
  trait MultipliableUIntUInt extends Multipliable[UInt, UInt, UInt] {
    override def mul = (i: UInt, w: UInt) => i * w
  }
  trait MultipliableUIntSInt extends Multipliable[UInt, SInt, SInt] {
    override def mul = (i: UInt, w: SInt) => i * w
  }
  trait MultipliableUIntBool extends Multipliable[UInt, Bool, SInt] {
    override def mul = (i: UInt, w: Bool) => Mux(w, i.zext, -(i.zext))
  }

  trait MultipliableSIntUInt extends Multipliable[SInt, UInt, SInt] {
    override def mul = (i: SInt, w: UInt) => i * w
  }
  trait MultipliableSIntSInt extends Multipliable[SInt, SInt, SInt] {
    override def mul = (i: SInt, w: SInt) => i * w
  }
  trait MultipliableSIntBool extends Multipliable[SInt, Bool, SInt] {
    override def mul = (i: SInt, w: Bool) => Mux(w, i, 0.S -& i)
  }

  trait MultipliableBoolUInt extends Multipliable[Bool, UInt, SInt] {
    override def mul = (i: Bool, w: UInt) => Mux(i, w.zext, -w.zext)
  }
  trait MultipliableBoolSInt extends Multipliable[Bool, SInt, SInt] {
    override def mul = (i: Bool, w: SInt) => Mux(i, w, 0.S -& w)
  }
  trait MultipliableBoolBool extends Multipliable[Bool, Bool, Bool] {
    override def mul = (i: Bool, w: Bool) => ~(i ^ w)
  }
}

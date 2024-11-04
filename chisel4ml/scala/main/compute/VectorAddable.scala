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
import chisel3.util._

trait VectorAddable[M <: Bits, A <: Bits] {
  def addVec: Vec[M] => A
}

package VectorAddableImplementations {

  trait VectorAddableBoolUInt extends VectorAddable[Bool, UInt] {
    def addVec = (x: Vec[Bool]) => PopCount(x.asUInt)
  }

  trait VectorAddableBoolSInt extends VectorAddable[Bool, SInt] {
    def addVec = (x: Vec[Bool]) => PopCount(x.asUInt).zext -& PopCount(~x.asUInt).zext
  }

  trait VectorAddableSIntSInt extends VectorAddable[SInt, SInt] {
    def addVec = (x: Vec[SInt]) => x.reduceTree(_ +& _)
  }

  trait VectorAddableUIntUInt extends VectorAddable[UInt, UInt] {
    def addVec = (x: Vec[UInt]) => x.reduceTree(_ +& _)
  }

  trait VectorAddableUIntSInt extends VectorAddable[UInt, SInt] {
    def addVec = (x: Vec[UInt]) => x.reduceTree(_ +& _).zext
  }
}

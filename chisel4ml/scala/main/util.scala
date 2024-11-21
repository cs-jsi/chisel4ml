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

import _root_.lbir._
import _root_.org.slf4j.LoggerFactory
import _root_.scala.math.{log, pow}
import chisel3._

package object util {
  val logger = LoggerFactory.getLogger("chisel4ml.util.")

  def log2(x: Int):   Int = (log(x.toFloat) / log(2.0)).toInt
  def log2(x: Float): Float = (log(x) / log(2.0)).toFloat

  def toBinary(i: Int, digits: Int = 8): String = {
    val replacement = if (i >= 0) '0' else '1'
    String.format(s"%${digits}s", i.toBinaryString.takeRight(digits)).replace(' ', replacement)
  }
  def toBinaryB(i: BigInt, digits: Int = 8): String = {
    val replacement = if (i >= 0) '0' else '1'
    String.format("%" + digits + "s", i.toString(2)).replace(' ', replacement)
  }
  def signedCorrect(x: Float, dtype: Datatype): Float = {
    if (dtype.signed && x > (pow(2, dtype.bitwidth - 1) - 1))
      x - pow(2, dtype.bitwidth).toFloat
    else
      x
  }

  def saturateFnU(x: UInt, bitwidth: Int): UInt = {
    val max = (pow(2, bitwidth) - 1).toInt.U
    Mux(x > max, max, x)
  }
  def saturateFnS(x: SInt, bitwidth: Int): SInt = {
    val max = (pow(2, bitwidth - 1) - 1).toInt.S
    val min = -pow(2, bitwidth - 1).toInt.S
    Mux(x > max, max, Mux(x < min, min, x))
  }

  def risingEdge(x: Bool) = x && !RegNext(x)

  def isStable[T <: Data](x: T): Bool = x === RegNext(x)
}

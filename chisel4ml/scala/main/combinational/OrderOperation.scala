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
package chisel4ml.combinational

import chisel3._
import chisel4ml.compute.OrderCompute

/** Defines an interface for a comparing values
  *
  * Used in the OrderProcessingUnit to create a MaxPool layer.
  *
  * @param oc
  *   A OrderCompute object that defines the quantization details.
  */
trait OrderOperation {

  /** Given an order what is the output value from the input sequence? (e.g for maxpool give largest value)
    */
  def apply(oc: OrderCompute)(in: Seq[oc.T]): oc.T
}

/** An OrderOperation implementation
  *
  * Used in the OrderProcessingUnit to create a MaxPool layer.
  *
  * @param oc
  *   A OrderCompute object that defines the quantization details.
  */
object MaxPoolOperation extends OrderOperation {
  def max(oc: OrderCompute)(x: oc.T, y: oc.T): oc.T = {
    val out = Wire(chiselTypeOf(x))
    when(oc.gte(x, y)) {
      out := x
    }.otherwise {
      out := y
    }
    out
  }
  def apply(oc: OrderCompute)(in: Seq[oc.T]): oc.T = {
    VecInit(in).reduceTree(this.max(oc)(_, _))
  }
}

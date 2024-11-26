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
package lbir

import chisel3._
import chisel4ml.implicits._
import interfaces.amba.axis._

/** Extends the AXIStreamDriver with ability to write and read QTensor objects to AXI Stream.
  *
  * Extends `axiDrive` with functions `enqueueQTensor` and `dequeueQTensor`. This function is called implicitly (see
  * implicit def axiStreamToLBIRDriver).
  *
  * @param axiDrive
  *   The AXIStreamDriver to extends.
  */
class AXIStreamLBIRDriver[T <: Data](val axiDrive: AXIStreamDriver[T]) {

  /** Drives a AXIStreamIO with a LBIR QTensor.
    *
    * The qtensor is serialized as per the layout specification of the QTensor (See LBIR proto for more information on
    * this)
    *
    * @param qt
    *   QTensor to drive to the AXI Stream bus.
    * @param clock
    *   The clock used to drive the bus.
    */
  def enqueueQTensor(qt: QTensor, clock: Clock): Unit = {
    val transactions = qt.toLBIRTransactions[T](axiDrive.getBusWidth())
    axiDrive.enqueuePacket(transactions, clock)
  }

  /** Reads a AXIStreamIO to obtain an LBIR QTensor.
    *
    * The qtensor is deserialized as per the layout specification of the QTensor (See LBIR proto for more information on
    * this)
    *
    * @param qt
    *   QTensor shape we are expecting (empty qtensor with only shape in datatype).
    * @param clock
    *   The clock used to drive the bus.
    */
  def dequeueQTensor(stencil: QTensor, clock: Clock): QTensor = {
    axiDrive.dequeuePacket(clock).toQTensor(stencil, axiDrive.getBusWidth())
  }
}

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


class AXIStreamLBIRDriver(val axiDrive: AXIStreamDriver[UInt]) {
    def enqueueQTensor(qt: QTensor, clock: Clock): Unit = {
        val busWidth = axiDrive.getBusWidth()
        val qtBinaryString = qt.toBinaryString.drop(1) // remove the "b" in "b010101"
        val multipleOfBusWidth = busWidth * math.ceil(qt.values.length.toFloat / busWidth.toFloat).toInt
        val paddedString = qtBinaryString.reverse.padTo(multipleOfBusWidth, '0').reverse
        axiDrive.enqueuePacket(Seq(("b"+paddedString).U(busWidth.W)), clock)
    }

    def dequeueQTensor(clock: Clock): QTensor = {
        QTensor(values=axiDrive.dequeuePacket(clock).map(x => ((x * 2) - 1).toFloat))
    }
}

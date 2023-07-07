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
package chisel4ml.tests

import _root_.chisel4ml.sequential._
import _root_.org.slf4j.LoggerFactory
import chisel3._
import chiseltest._
import org.scalatest.flatspec.AnyFlatSpec

class KernelRegisterFileTests extends AnyFlatSpec with ChiselScalatestTester {
  val logger = LoggerFactory.getLogger(classOf[KernelRegisterFileTests])

  behavior.of("KernelRegisterFile module")
  it should "show appropirate window as it cycles through the input image" in {
    test(new KernelRegisterFile(2, 2, 4)) { dut =>
      dut.io.chAddr.poke(0.U)
      dut.io.rowAddr.poke(0.U)
      dut.io.colAddr.poke(0.U)
      dut.io.inData.poke("b0000".U)
      dut.io.inValid.poke(true.B)
      dut.clock.step()
      dut.io.chAddr.poke(0.U)
      dut.io.rowAddr.poke(0.U)
      dut.io.colAddr.poke(1.U)
      dut.io.inData.poke("b0001".U)
      dut.clock.step()
      dut.io.chAddr.poke(0.U)
      dut.io.rowAddr.poke(1.U)
      dut.io.colAddr.poke(0.U)
      dut.io.inData.poke("b0010".U)
      dut.clock.step()
      dut.io.chAddr.poke(0.U)
      dut.io.rowAddr.poke(1.U)
      dut.io.colAddr.poke(1.U)
      dut.io.inData.poke("b0011".U)
      dut.clock.step()
      dut.io.chAddr.poke(1.U)
      dut.io.rowAddr.poke(0.U)
      dut.io.colAddr.poke(0.U)
      dut.io.inData.poke("b0100".U)
      dut.clock.step()
      dut.io.chAddr.poke(1.U)
      dut.io.rowAddr.poke(0.U)
      dut.io.colAddr.poke(1.U)
      dut.io.inData.poke("b0101".U)
      dut.clock.step()
      dut.io.chAddr.poke(1.U)
      dut.io.rowAddr.poke(1.U)
      dut.io.colAddr.poke(0.U)
      dut.io.inData.poke("b0110".U)
      dut.clock.step()
      dut.io.chAddr.poke(1.U)
      dut.io.rowAddr.poke(1.U)
      dut.io.colAddr.poke(1.U)
      dut.io.inData.poke("b0111".U)
      dut.clock.step()
      dut.io.outData.expect("b0111_0110_0101_0100_0011_0010_0001_0000".U)
      dut.clock.step()
    }
  }
}

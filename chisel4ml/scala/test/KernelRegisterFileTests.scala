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

import org.scalatest.flatspec.AnyFlatSpec
import chisel3._
import chiseltest._

import _root_.chisel4ml.sequential._

class KernelRegisterFileTests extends AnyFlatSpec with ChiselScalatestTester {
    behavior of "KernelRegisterFile module"
    it should "show appropirate window as it cycles through the input image" in {
        test(new KernelRegisterFile(2, 1, 4)) { dut =>
            dut.io.flushRegs.poke(false.B)
            dut.io.shiftRegs.poke(false.B)
            dut.io.rowWriteMode.poke(true.B)
            dut.io.kernelAddr.poke(0.U)
            dut.io.rowAddr.poke(0.U)
            dut.io.inData.poke(0.U)
            dut.io.inValid.poke(false.B)
            dut.clock.step()

            dut.io.rowAddr.poke(0.U)
            dut.io.inValid.poke(true.B)
            dut.io.inData.poke("b0001_0000".U)
            dut.clock.step()

            /* 0  1
             *
             * 0  0
             */
            dut.io.outData.expect("b0000_0000_0001_0000".U)
            dut.io.rowAddr.poke(1.U)
            dut.io.inValid.poke(true.B)
            dut.io.inData.poke("b0011_0010".U)
            dut.clock.step()

            /* 0  1
             *
             * 2  3
             */
            dut.io.outData.expect("b0011_0010_0001_0000".U)
            dut.io.shiftRegs.poke(true.B)
            dut.io.rowWriteMode.poke(false.B)
            dut.io.inData.poke("b0101_0100".U)
            dut.io.inValid.poke(true.B)
            dut.clock.step()

            /* 1  4
             *
             * 3  5
             */
            dut.io.outData.expect("b0101_0011_0100_0001".U)
            dut.io.inValid.poke(false.B)
        }
    }
}

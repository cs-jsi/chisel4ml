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

import _root_.chisel4ml.tests.SlidingWindowUnitTestBed
import _root_.lbir.Datatype.QuantizationType.UNIFORM
import _root_.org.slf4j.LoggerFactory
import chisel3._
import chiseltest._
import org.scalatest.flatspec.AnyFlatSpec

class SlidingWindowUnitTests extends AnyFlatSpec with ChiselScalatestTester {
  val logger = LoggerFactory.getLogger(classOf[SlidingWindowUnitTests])

  val dtype = new lbir.Datatype(quantization = UNIFORM, bitwidth = 5, signed = false, shift = Seq(0), offset = Seq(0))
  val testParameters = lbir.QTensor(
    dtype = Option(dtype),
    shape = Seq(1, 2, 3, 3),
    values = Seq(1, 2, 3,
                 4, 5, 6,
                 7, 8, 9,
                 10, 11, 12,
                 13, 14, 15,
                 16, 17, 18),
  )

  behavior.of("SlidingWindowUnit module")
  it should "show appropirate window as it cycles through the input image" in {
    test(new SlidingWindowUnitTestBed(kernelSize = 2,
                                      kernelDepth = 2,
                                      actWidth = 3,
                                      actHeight = 3,
                                      actParamSize = 5,
                                      parameters = testParameters)) { dut =>
          //                  data,            rowAddr, chAddr, rowWriteMode
          val testVec = Seq(("b00010_00001".U(10.W), 0, 0,      true), // (1, 2)
                            ("b00101_00100".U(10.W), 1, 0,      true), // (4, 5)
                            ("b01011_01010".U(10.W), 0, 1,      true), // (10, 11)
                            ("b01110_01101".U(10.W), 1, 1,      true), // (13, 14)
                            ("b00110_00011".U(10.W), 1, 0,      false), // (3, 6)
                            ("b01111_01100".U(10.W), 1, 1,      false), // (12, 15)
                            ("b00101_00100".U(10.W), 0, 0,      true), // (4, 5)
                            ("b01000_00111".U(10.W), 1, 0,      true), // (7, 8)
                            ("b01110_01101".U(10.W), 0, 1,      true), // (13, 14)
                            ("b10001_10000".U(10.W), 1, 1,      true), // (16, 17)
                            ("b01001_00110".U(10.W), 1, 0,      false), // (6, 9)
                            ("b10010_01111".U(10.W), 1, 1,      false), // (15, 18)
                            )
          dut.clock.step()
          dut.io.start.poke(true.B)
          dut.clock.step()
          dut.io.start.poke(false.B)
          for (expected <- testVec) {
            while(!dut.io.rrfInValid.peek().litToBoolean) { dut.clock.step() } // wait for valid
            dut.io.rrfInData.expect(expected._1)
            dut.io.rrfRowAddr.expect(expected._2)
            dut.io.rrfChAddr.expect(expected._3)
            dut.io.rrfRowWrMode.expect(expected._4)
            dut.clock.step()
          }
    }
  }
}

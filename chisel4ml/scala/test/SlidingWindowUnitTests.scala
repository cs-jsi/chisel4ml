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
    values = Seq(1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18),
  )

  behavior.of("SlidingWindowUnit module")
  it should "show appropirate window as it cycles through the input image" in {
    test(
      new SlidingWindowUnitTestBed(
        kernelSize = 2,
        kernelDepth = 2,
        actWidth = 3,
        actHeight = 3,
        actParamSize = 5,
        parameters = testParameters,
      ),
    ) { dut =>
      dut.clock.step()
      dut.io.start.poke(true.B)
      dut.clock.step()
      dut.io.start.poke(false.B)
      dut.clock.step(20)
    }
  }
}

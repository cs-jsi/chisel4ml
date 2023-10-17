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

import chisel3._
import chisel3.experimental.VecLiterals._
import scala.language.implicitConversions
import chiseltest._
import chisel4ml.conv2d.ShiftRegisterConvolver
import org.scalatest.flatspec.AnyFlatSpec
import org.slf4j.LoggerFactory
import _root_.lbir.Datatype.QuantizationType.UNIFORM

class ShiftRegisterConvolverTests extends AnyFlatSpec with ChiselScalatestTester {
  val logger = LoggerFactory.getLogger(classOf[ShiftRegisterConvolverTests])

  val dtype = new lbir.Datatype(quantization = UNIFORM, bitwidth = 5, signed = false, shift = Seq(0), offset = Seq(0))
  val inputParams = lbir.QTensor(
    dtype = dtype,
    shape = Seq(1, 1, 3, 3),
    values = Seq(1, 2, 3, 4, 5, 6, 7, 8, 9)
  )
  val kernelParams = lbir.QTensor(
    dtype = new lbir.Datatype(quantization = UNIFORM),
    shape = Seq(1, 1, 2, 2)
  )
  val goldenVec = Seq(
    Vec.Lit(1.U(5.W), 2.U(5.W), 4.U(5.W), 5.U(5.W)),
    Vec.Lit(2.U(5.W), 3.U(5.W), 5.U(5.W), 6.U(5.W)),
    Vec.Lit(4.U(5.W), 5.U(5.W), 7.U(5.W), 8.U(5.W)),
    Vec.Lit(5.U(5.W), 6.U(5.W), 8.U(5.W), 9.U(5.W))
  )

  behavior.of("ShiftRegisterConvolver module")
  it should "show appropirate window as it cycles through the input image" in {
    test(new ShiftRegisterConvolver(input = inputParams, kernel = kernelParams)) { dut =>
      dut.io.nextElement.initSource()
      dut.io.nextElement.setSourceClock(dut.clock)
      dut.io.inputActivationsWindow.initSink()
      dut.io.inputActivationsWindow.setSinkClock(dut.clock)

      dut.reset.poke(true.B)
      dut.clock.step()
      dut.reset.poke(false.B)
      fork {
        dut.io.nextElement.enqueueSeq(inputParams.values.map(_.toInt.U))
      }.fork {
        dut.io.inputActivationsWindow.expectDequeueSeq(goldenVec)
      }.join()
    }
  }

}

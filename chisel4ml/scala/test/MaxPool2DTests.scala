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

import chisel4ml.compute.OrderCompute
import chisel4ml.implicits._
import chisel4ml.sequential._
import chisel4ml.{LayerWrapSeqField, NumBeatsInField, NumBeatsOutField}
import chiseltest._
import lbir.Datatype.QuantizationType.UNIFORM
import org.chipsalliance.cde.config.Config
import org.scalatest.flatspec.AnyFlatSpec
import org.slf4j.LoggerFactory

class MaxPool2DTests extends AnyFlatSpec with ChiselScalatestTester {
  val logger = LoggerFactory.getLogger(classOf[MaxPool2DTests])

  val dtype = new lbir.Datatype(quantization = UNIFORM, bitwidth = 4, signed = false, shift = Seq(0), offset = Seq(0))
  val testParameters = lbir.QTensor(
    dtype = dtype,
    shape = Seq(1, 2, 4, 4),
    values =
      Seq(1, 2, 3, 3, 4, 5, 6, 6, 7, 8, 9, 9, 7, 8, 9, 9, 10, 11, 12, 12, 13, 14, 15, 15, 14, 13, 12, 11, 10, 9, 8, 7)
  )
  val stencil = lbir.QTensor(
    dtype = dtype,
    shape = Seq(1, 2, 2, 2)
  )
  val expectedOutput = lbir.QTensor(
    dtype = dtype,
    shape = Seq(1, 2, 2, 2),
    values = Seq(5, 6, 8, 9, 14, 15, 14, 12)
  )
  val layer = lbir.MaxPool2DConfig(
    input = testParameters,
    output = stencil
  )

  behavior.of("MaxPool2D module")
  it should "compute max pooling for stride 2" in {
    val cfg = new Config((_, _, _) => {
      case LayerWrapSeqField => Seq(layer)
      case NumBeatsInField   => 4
      case NumBeatsOutField  => 4
    })
    test(new MaxPool2D(OrderCompute(layer))(cfg)).withAnnotations(Seq(WriteVcdAnnotation)) { dut =>
      var res: lbir.QTensor = lbir.QTensor()
      fork {
        dut.inStream.enqueueQTensor(testParameters, dut.clock)
      }.fork {
        res = dut.outStream.dequeueQTensor(stencil, dut.clock)
      }.join()
      assert(res.values == expectedOutput.values)
    }
  }
}

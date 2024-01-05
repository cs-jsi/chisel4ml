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

import _root_.chisel4ml.implicits._
import org.chipsalliance.cde.config.Config
import chisel4ml.sequential.MaxPool2DConfigField
import lbir.MaxPool2DConfig
import _root_.lbir.Datatype.QuantizationType.UNIFORM
import _root_.org.slf4j.LoggerFactory
import _root_.services._
import chisel4ml.sequential.MaxPool2D
import chiseltest._
import org.scalatest.flatspec.AnyFlatSpec

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
  val options = LayerOptions(
    busWidthIn = 32,
    busWidthOut = 32
  )

  behavior.of("MaxPool2D module")
  it should "compute max pooling for stride 2" in {
    val cfg = new Config((site, here, up) => {
        case MaxPool2DConfigField => layer
    })
    test(new MaxPool2D(options)(cfg)).withAnnotations(Seq(WriteVcdAnnotation)) { dut =>
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

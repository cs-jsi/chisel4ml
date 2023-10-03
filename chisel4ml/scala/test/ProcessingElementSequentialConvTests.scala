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

import _root_.chisel4ml.conv2d.ProcessingElementSequentialConv
import _root_.chisel4ml.implicits._
import _root_.lbir.Datatype.QuantizationType.UNIFORM
import _root_.org.slf4j.LoggerFactory
import _root_.services._
import chisel3._
import chiseltest._
import firrtl.transforms.NoCircuitDedupAnnotation
import memories.MemoryGenerator
import org.scalatest.flatspec.AnyFlatSpec
import org.scalatest.{BeforeAndAfterEachTestData, TestData}

import java.nio.file.Paths

class ProcessingElementSequentialConvTests
    extends AnyFlatSpec
    with ChiselScalatestTester
    with BeforeAndAfterEachTestData {
  val logger = LoggerFactory.getLogger(classOf[ProcessingElementSequentialConvTests])

  // We override the memory generation location before each test, so that the MemoryGenerator
  // uses the correct directory to generate hex file into.
  override def beforeEach(testData: TestData): Unit = {
    val genDirStr = (testData.name).replace(' ', '_')
    val genDir = Paths.get(".", "test_run_dir", genDirStr).toAbsolutePath() // TODO: programmatically get test_run_dir?
    MemoryGenerator.setGenDir(genDir)
    super.beforeEach(testData)
  }

  val dtypeUInt4 = lbir.Datatype(quantization = UNIFORM, bitwidth = 4, signed = false, shift = Seq(0), offset = Seq(0))
  val dtypeSInt4 = lbir.Datatype(quantization = UNIFORM, bitwidth = 4, signed = true, shift = Seq(0), offset = Seq(0))

  val testLayer0 = lbir.Conv2DConfig(
    thresh = Seq(0).UQ(4),
    kernel = Seq(1, 0, 0, 0).SQ(4, Seq(1, 1, 2, 2)),
    input = Seq().UQ(4, Seq(1, 1, 3, 3)),
    output = Seq().SQ(4, Seq(1, 1, 2, 2)),
    activation = lbir.Activation.NO_ACTIVATION
  )

  val testOptions0 = services.GenerateCircuitParams.Options(
    isSimple = false,
    pipelineCircuit = false,
    layers = Seq(LayerOptions(busWidthIn = 32, busWidthOut = 32))
  )

  val dtypeUInt6 = lbir.Datatype(quantization = UNIFORM, bitwidth = 6, signed = false, shift = Seq(0), offset = Seq(0))
  val dtypeSInt7 = lbir.Datatype(quantization = UNIFORM, bitwidth = 7, signed = true, shift = Seq(0), offset = Seq(0))
  val testLayer1 = lbir.Conv2DConfig(
    thresh = lbir.QTensor(
      dtype = dtypeSInt7,
      shape = Seq(1),
      values = Seq(-2)
    ),
    kernel = lbir.QTensor(
      dtype = dtypeSInt7,
      shape = Seq(1, 2, 2, 2),
      values = Seq(1, 0, 0, 0, 0, 0, 1, 0)
    ),
    input = lbir.QTensor(
      dtype = dtypeUInt6,
      shape = Seq(1, 2, 3, 3)
    ),
    output = lbir.QTensor(
      dtype = dtypeSInt7,
      shape = Seq(1, 1, 2, 2)
    )
  )

  behavior.of("ProcessingElementSequentialConv module")
  it should "compute the convolution correctly" in { // .withAnnotations(Seq(VerilatorBackendAnnotation))
    test(
      new ProcessingElementSequentialConv[UInt, SInt, SInt, SInt, SInt, SInt](
        layer = testLayer0,
        options = testOptions0.layers(0),
        mul = (x: UInt, w: SInt) => (x * w),
        add = (x: Vec[SInt]) => x.reduceTree(_ +& _),
        actFn = (x: SInt, _: SInt) => x
      )
    ).withAnnotations(Seq(VerilatorBackendAnnotation, NoCircuitDedupAnnotation, WriteFstAnnotation)) { dut =>
      dut.inStream.initSource()
      dut.inStream.setSourceClock(dut.clock)
      dut.outStream.initSink()
      dut.outStream.setSinkClock(dut.clock)
      dut.clock.step(1)
      dut.inStream.enqueueSeq(
        Seq("b1000_0111__0110_0101__0100_0011__0010_0001".U, "b0000_0000__0000_0000__0000_0000__0000_1001".U)
      )
      dut.inStream.last.poke(true.B)
      dut.clock.step(1)
      dut.inStream.last.poke(false.B)
      dut.outStream.expectDequeueSeq(Seq("b0000_0000__0000_0000__0101_0100__0010_0001".U), dut.clock)
    }
  }

  it should "compute a convolution with several channels correctly" in {
    test(
      new ProcessingElementSequentialConv[UInt, SInt, SInt, SInt, SInt, SInt](
        layer = testLayer1,
        options = testOptions0.layers(0),
        mul = (x: UInt, w: SInt) => (x * w),
        add = (x: Vec[SInt]) => x.reduceTree(_ +& _),
        actFn = (x: SInt, y: SInt) => x + y
      )
    ).withAnnotations(Seq(VerilatorBackendAnnotation)) { dut =>
      dut.inStream.initSource()
      dut.inStream.setSourceClock(dut.clock)
      dut.outStream.initSink()
      dut.outStream.setSinkClock(dut.clock)
      dut.clock.step()
      /*  1   2   3   |   1  0  |  1 + 13 - 2 = 12  | 12 14
       *  4   5   6   |   0  0  |  2 + 14 - 2 = 14  | 18 20
       *  7   8   9   |         |  4 + 16 - 2 = 18  |
       *              |         |  5 + 17 - 2 = 20  |
       *  10 11  12   |   0  0  |                   |
       *  13 14  15   |   1  0  |                   |
       *  16 17  18   |         |                   |
       */

      dut.inStream.enqueueSeq(
        Seq(
          "b00_000101_000100_000011_000010_000001".U,
          "b00_001010_001001_001000_000111_000110".U,
          "b00_001111_001110_001101_001100_001011".U,
          "b00_000000_000000_010010_010001_010000".U
        )
      )
      dut.inStream.last.poke(true.B)
      dut.clock.step()
      dut.inStream.last.poke(false.B)
      dut.outStream.expectDequeueSeq(Seq("b0000_0010100_0010010_0001110_0001100".U), dut.clock)
    }
  }
}

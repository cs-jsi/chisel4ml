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
import _root_.chisel4ml.util._
import _root_.lbir.Datatype.QuantizationType.UNIFORM
import _root_.org.slf4j.LoggerFactory
import chisel3._
import chiseltest._
import org.scalatest.flatspec.AnyFlatSpec
import org.scalatest.{BeforeAndAfterEachTestData, TestData}
import java.nio.file.Paths
import memories.MemoryGenerator
import org.nd4j.linalg.api.ndarray.INDArray
import org.nd4j.linalg.factory.Nd4j
import org.nd4j.linalg.indexing.NDArrayIndex
import org.nd4j.linalg.ops.transforms.Transforms

class SlidingWindowUnitTests extends AnyFlatSpec with ChiselScalatestTester with BeforeAndAfterEachTestData {
  val logger = LoggerFactory.getLogger(classOf[SlidingWindowUnitTests])

  // We override the memory generation location before each test, so that the MemoryGenerator
  // uses the correct directory to generate hex file into.
  override def beforeEach(testData: TestData): Unit = {
    val genDirStr = (testData.name).replace(' ', '_')
    val genDir = Paths.get(".", "test_run_dir", genDirStr).toAbsolutePath() // TODO: programmatically get test_run_dir?
    MemoryGenerator.setGenDir(genDir)
    super.beforeEach(testData)
  }

  val dtype = new lbir.Datatype(quantization = UNIFORM, bitwidth = 5, signed = false, shift = Seq(0), offset = Seq(0))
  val testParameters = lbir.QTensor(
    dtype = dtype,
    shape = Seq(1, 2, 3, 3),
    values = Seq(1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18)
  )

  val dtype2 = new lbir.Datatype(quantization = UNIFORM, bitwidth = 6, signed = false, shift = Seq(0), offset = Seq(0))
  val testParameters2 = lbir.QTensor(
    dtype = dtype2,
    shape = Seq(1, 2, 5, 4),
    values = Seq(1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28,
      29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40)
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
        parameters = testParameters
      )
    ) { dut =>
      //                  data,            rowAddr, chAddr, rowWriteMode
      val testVec = Seq(
        ("b00010_00001".U(10.W), 0, 0, true), // (1, 2)
        ("b00101_00100".U(10.W), 1, 0, true), // (4, 5)
        ("b01011_01010".U(10.W), 0, 1, true), // (10, 11)
        ("b01110_01101".U(10.W), 1, 1, true), // (13, 14)
        ("b00110_00011".U(10.W), 1, 0, false), // (3, 6)
        ("b01111_01100".U(10.W), 1, 1, false), // (12, 15)
        ("b00101_00100".U(10.W), 0, 0, true), // (4, 5)
        ("b01000_00111".U(10.W), 1, 0, true), // (7, 8)
        ("b01110_01101".U(10.W), 0, 1, true), // (13, 14)
        ("b10001_10000".U(10.W), 1, 1, true), // (16, 17)
        ("b01001_00110".U(10.W), 1, 0, false), // (6, 9)
        ("b10010_01111".U(10.W), 1, 1, false) // (15, 18)
      )
      dut.clock.step()
      dut.io.start.poke(true.B)
      dut.clock.step()
      dut.io.start.poke(false.B)
      for ((expected, ind) <- testVec.zipWithIndex) {
        while (!dut.io.rrfInValid.peek().litToBoolean) { dut.clock.step() } // wait for valid
        dut.io.rrfInData.expect(expected._1)
        dut.io.rrfRowAddr.expect(expected._2)
        dut.io.rrfChAddr.expect(expected._3)
        dut.io.rrfRowWrMode.expect(expected._4)
        dut.clock.step()
      }
    }
  }
  it should "show appropirate window as it cycles through the input image with a bigger window" in {
    test(
      new SlidingWindowUnitTestBed(
        kernelSize = 3,
        kernelDepth = 2,
        actWidth = 5,
        actHeight = 4,
        actParamSize = 6,
        parameters = testParameters2
      )
    ) { dut =>
      //                  data                      rowAddr, ChAddr, rowWriteMode
      val testVec = Seq(
        ("b000011_000010_000001".U(18.W), 0, 0, true), // (1, 2, 3)
        ("b001000_000111_000110".U(18.W), 1, 0, true), // (6, 7, 8)
        ("b001101_001100_001011".U(18.W), 2, 0, true), // (11, 12, 13)
        ("b010111_010110_010101".U(18.W), 0, 1, true), // (21, 22, 23)
        ("b011100_011011_011010".U(18.W), 1, 1, true), // (26, 27, 28)
        ("b100001_100000_011111".U(18.W), 2, 1, true), // (31, 32, 33)
        ("b001110_001001_000100".U(18.W), 2, 0, false), // (4, 9, 14)
        ("b100010_011101_011000".U(18.W), 2, 1, false), // (24, 29, 34)
        ("b001111_001010_000101".U(18.W), 2, 0, false), // (5, 10, 15)
        ("b100011_011110_011001".U(18.W), 2, 1, false), // (25, 30, 35)
        ("b001000_000111_000110".U(18.W), 0, 0, true), // (6, 7, 8)
        ("b001101_001100_001011".U(18.W), 1, 0, true), // (11, 12, 13)
        ("b010010_010001_010000".U(18.W), 2, 0, true), // (16, 17, 18)
        ("b011100_011011_011010".U(18.W), 0, 1, true), // (26, 27, 28)
        ("b100001_100000_011111".U(18.W), 1, 1, true), // (31, 32, 33)
        ("b100110_100101_100100".U(18.W), 2, 1, true), // (36, 37, 38)
        ("b010011_001110_001001".U(18.W), 2, 0, false), // (9, 14, 19)
        ("b100111_100010_011101".U(18.W), 2, 1, false), // (29, 34, 39)
        ("b010100_001111_001010".U(18.W), 2, 0, false), // (10, 15, 20)
        ("b101000_100011_011110".U(18.W), 2, 1, false)
      ) // (30, 35, 40)

      dut.clock.step()
      dut.io.start.poke(true.B)
      dut.clock.step()
      dut.io.start.poke(false.B)
      for (expected <- testVec) {
        while (!dut.io.rrfInValid.peek().litToBoolean) { dut.clock.step() } // wait for valid
        dut.io.rrfInData.expect(expected._1)
        dut.io.rrfRowAddr.expect(expected._2)
        dut.io.rrfChAddr.expect(expected._3)
        dut.io.rrfRowWrMode.expect(expected._4)
        dut.clock.step()
      }
    }
  }

  val rand = new scala.util.Random(seed = 42)
  Nd4j.getRandom().setSeed(42)
  for (testcaseId <- 0 until 10) {
    val randKernelSize = rand.between(2, 7 + 1) // rand.between(inclusive, exclusive)
    val randKernelDepth = rand.between(1, 16 + 1)
    val randActParamBitwidth = rand.between(1, 8 + 1)
    val randImageWidth = rand.between(randKernelSize + 1, randKernelSize + 7 + 1)
    val randImageHeight = rand.between(randKernelSize + 1, randKernelSize + 7 + 1)
    val randImageNormal = Nd4j.rand(Array(randKernelDepth, randImageHeight, randImageWidth))
    val randImage = Transforms.round(randImageNormal.mul(scala.math.pow(2, randActParamBitwidth) - 1))
    val dtype = new lbir.Datatype(
      quantization = UNIFORM,
      bitwidth = randActParamBitwidth,
      signed = false,
      shift = Seq(0),
      offset = Seq(0)
    )
    val testParameters = lbir.QTensor(
      dtype = dtype,
      shape = Seq(1, randKernelDepth, randImageWidth, randImageHeight),
      values = arrToSeq(randImage)
    )
    it should s"""testcaseid: $testcaseId, random params: kernelSize=$randKernelSize, kernelDepth=$randKernelDepth
                 |actParamBitiwdth=$randActParamBitwidth, imageWidth=$randImageWidth, imageHeight=
                 |$randImageHeight.""".stripMargin.replace("\n", " ") in {
      test(
        new SlidingWindowUnitTestBed(
          kernelSize = randKernelSize,
          kernelDepth = randKernelDepth,
          actWidth = randImageWidth,
          actHeight = randImageHeight,
          actParamSize = randActParamBitwidth,
          parameters = testParameters
        )
      ) { dut =>
        dut.clock.setTimeout(10000)
        dut.clock.step()
        dut.io.start.poke(true.B)
        dut.clock.step()
        dut.io.start.poke(false.B)
        for (i <- 0 until randImageHeight - randKernelSize + 1) {
          for (j <- 0 until randImageWidth - randKernelSize + 1) {
            val window = randImage.get(
              NDArrayIndex.all(), // kernel
              NDArrayIndex.interval(i, i + randKernelSize),
              NDArrayIndex.interval(j, j + randKernelSize)
            )
            while (!dut.io.rrfImageValid.peek().litToBoolean) { dut.clock.step() } // wait for valid
            dut.clock.step()
            dut.io.rrfOutData.expect(ndArrayToBinaryString(window, randActParamBitwidth).U)
          }
        }
      }
    }
  }

  def arrToSeq(arr: INDArray): Seq[Float] = {
    var mySeq: Seq[Float] = Seq()
    val flatArr = Nd4j.toFlattened(arr)
    for (ind <- 0 until flatArr.length()) {
      mySeq = mySeq :+ flatArr.getFloat(ind)
    }
    mySeq
  }
}

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
import _root_.chisel4ml.implicits._
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

class KernelRFLoaderTests extends AnyFlatSpec with ChiselScalatestTester with BeforeAndAfterEachTestData {
  val logger = LoggerFactory.getLogger(classOf[KernelRFLoaderTests])

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
    shape = Seq(2, 1, 4, 4),
    values = Seq(1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28,
      29, 30, 31, 32)
  )

  val testParamsA = lbir.QTensor(
    dtype = dtype2,
    shape = Seq(1, 1, 4, 4),
    values = Seq(1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16)
  )
  val testParamsB = lbir.QTensor(
    dtype = dtype2,
    shape = Seq(1, 1, 4, 4),
    values = Seq(17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32)
  )

  behavior.of("KernelRFLoader module")
  it should "load the single kernel correctly" in {
    test(
      new KernelRFLoaderTestBed(
        kernelSize = 3,
        kernelDepth = 2,
        kernelParamSize = 5,
        numKernels = 1,
        parameters = testParameters
      )
    ) { dut =>
      dut.clock.step()
      dut.io.loadKernel.poke(true.B)
      dut.io.kernelNum.poke(0.U)
      dut.clock.step()
      dut.io.loadKernel.poke(false.B)
      while (!dut.io.kernelReady.peek().litToBoolean) { dut.clock.step() } // wait for ready
      dut.clock.step()
      dut.io.krfOutput.expect(testParameters.toUInt)
      dut.clock.step()
    }
  }
  it should "load the two kernels correctly" in {
    test(
      new KernelRFLoaderTestBed(
        kernelSize = 4,
        kernelDepth = 1,
        kernelParamSize = 6,
        numKernels = 2,
        parameters = testParameters2
      )
    ) { dut =>
      dut.clock.step()
      dut.io.loadKernel.poke(true.B)
      dut.io.kernelNum.poke(0.U)
      dut.clock.step()
      dut.io.loadKernel.poke(false.B)
      while (!dut.io.kernelReady.peek().litToBoolean) { dut.clock.step() } // wait for ready
      dut.clock.step()
      dut.io.krfOutput.expect(testParamsA.toUInt)
      dut.clock.step()
      dut.io.loadKernel.poke(true.B)
      dut.io.kernelNum.poke(1.U)
      dut.clock.step()
      dut.io.loadKernel.poke(false.B)
      while (!dut.io.kernelReady.peek().litToBoolean) { dut.clock.step() } // wait for ready
      dut.clock.step()
      dut.io.krfOutput.expect(testParamsB.toUInt)
      dut.clock.step()
    }
  }
}

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

import _root_.chisel4ml.conv2d.KernelSubsystem
import _root_.lbir.Datatype.QuantizationType.UNIFORM
import _root_.org.slf4j.LoggerFactory
import chiseltest._
import memories.MemoryGenerator
import org.scalatest.flatspec.AnyFlatSpec
import org.scalatest.{BeforeAndAfterEachTestData, TestData}

class KernelRFLoaderTests extends AnyFlatSpec with ChiselScalatestTester with BeforeAndAfterEachTestData {
  val logger = LoggerFactory.getLogger(classOf[KernelRFLoaderTests])

  // We override the memory generation location before each test, so that the MemoryGenerator
  // uses the correct directory to generate hex file into.
  override def beforeEach(testData: TestData): Unit = {
    val genDirStr = (testData.name).replace(' ', '_')
    val genDir = os.pwd / "test_run_dir" / genDirStr
    MemoryGenerator.setGenDir(genDir)
    super.beforeEach(testData)
  }

  val dtype = new lbir.Datatype(quantization = UNIFORM, bitwidth = 5, signed = false, shift = Seq(0), offset = Seq(0))
  val testParameters = lbir.QTensor(
    dtype = dtype,
    shape = Seq(1, 2, 3, 3),
    values = Seq(1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18)
  )
  val threshParams = lbir.QTensor(
    dtype = dtype,
    shape = Seq(2),
    values = Seq(0, 0)
  )

  behavior.of("KernelRFLoader module")
  it should "load the single kernel correctly" in {
    test(
      new KernelSubsystem(lbir.Conv2DConfig(kernel = testParameters, thresh = threshParams))
    ) { dut =>
      dut.clock.step(4)
    }
  }
}

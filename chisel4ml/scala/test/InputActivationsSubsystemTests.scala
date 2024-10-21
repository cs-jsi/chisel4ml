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

import _root_.lbir.Datatype.QuantizationType.UNIFORM
import _root_.org.slf4j.LoggerFactory
import chisel3._
import chisel3.experimental.VecLiterals._
import chisel4ml.conv2d._
import chisel4ml.implicits._
import chisel4ml.quantization.IOContextUU
import chisel4ml.{LBIRNumBeatsIn, LBIRNumBeatsOut, LayerWrapIOField}
import chiseltest._
import lbir.Conv2DConfig
import memories.MemoryGenerator
import org.chipsalliance.cde.config.Config
import org.scalatest.flatspec.AnyFlatSpec
import org.scalatest.{BeforeAndAfterEachTestData, TestData}

class InputActivationsSubsystemTests extends AnyFlatSpec with ChiselScalatestTester with BeforeAndAfterEachTestData {
  val logger = LoggerFactory.getLogger(classOf[InputActivationsSubsystemTests])

  // We override the memory generation location before each test, so that the MemoryGenerator
  // uses the correct directory to generate hex file into.
  override def beforeEach(testData: TestData): Unit = {
    val genDirStr = (testData.name).replace(' ', '_')
    val genDir = os.pwd / "test_run_dir" / genDirStr
    MemoryGenerator.setGenDir(genDir)
    super.beforeEach(testData)
  }

  val dtype = new lbir.Datatype(quantization = UNIFORM, bitwidth = 4, signed = false, shift = Seq(0), offset = Seq(0))
  val inputParams = lbir.QTensor(
    dtype = dtype,
    shape = Seq(1, 1, 3, 3),
    values = Seq(1, 2, 3, 4, 5, 6, 7, 8, 9)
  )
  val kernelParams = lbir.QTensor(
    shape = Seq(1, 1, 2, 2)
  )
  val outParams = lbir.QTensor(
    shape = Seq(1, 1, 2, 2)
  )

  val conv2dLayer = Conv2DConfig(
    input = inputParams,
    kernel = kernelParams,
    output = outParams,
    depthwise = true
  )
  val cfg0 = new Config((_, _, _) => {
    case LayerWrapIOField => (conv2dLayer, IOContextUU)
    case LBIRNumBeatsIn   => 4
    case LBIRNumBeatsOut  => 4
  })
  behavior.of("InputActivationSubsystem module")
  it should "Send a simple input tensor through the input interface and read out the result" in {
    test(new InputActivationsSubsystem[UInt]()(cfg0)) { dut =>
      val goldenResVec = Seq(
        Vec.Lit(1.U(4.W), 2.U(4.W), 4.U(4.W), 5.U(4.W)),
        Vec.Lit(2.U(4.W), 3.U(4.W), 5.U(4.W), 6.U(4.W)),
        Vec.Lit(4.U(4.W), 5.U(4.W), 7.U(4.W), 8.U(4.W)),
        Vec.Lit(5.U(4.W), 6.U(4.W), 8.U(4.W), 9.U(4.W))
      )
      dut.io.inputActivationsWindow.initSink()
      dut.io.inputActivationsWindow.setSinkClock(dut.clock)

      dut.clock.step()
      fork {
        dut.io.inStream.enqueueQTensor(inputParams, dut.clock)
      }.fork {
        dut.io.inputActivationsWindow.expectDequeueSeq(goldenResVec)
      }.join()
    }
  }

  val rand = new scala.util.Random(seed = 42)
  for (testId <- 0 until 20) {
    val p = RandShiftRegConvTestParams(rand, numChannels = rand.between(1, 8))
    val (goldenVec, convLayer) = RandShiftRegConvTestParams.genShiftRegisterConvolverTestCase(p)
    val cfg = new Config((_, _, _) => {
      case LayerWrapIOField => (convLayer, IOContextUU)
      case LBIRNumBeatsIn   => 4
      case LBIRNumBeatsOut  => 4
    })
    it should f"Test $testId window a random input tensor with bw:${p.bitwidth} kernelHeight:${p.kernelHeight} " +
      f"kernelWidth:${p.kernelWidth}, inChannels:${p.inChannels}, inHeight:${p.inHeight}, inWidth:${p.inWidth}" in {
      test(new InputActivationsSubsystem[UInt]()(cfg)) { dut =>
        dut.io.inputActivationsWindow.initSink()
        dut.io.inputActivationsWindow.setSinkClock(dut.clock)

        dut.clock.step()
        fork {
          dut.io.inStream.enqueueQTensor(convLayer.input, dut.clock)
        }.fork {
          dut.io.inputActivationsWindow.expectDequeueSeq(goldenVec)
        }.join()
      }
    }
  }
}

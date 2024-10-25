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
import breeze.linalg.DenseMatrix
import chisel3._
import chisel3.experimental.VecLiterals._
import chisel4ml.sequential.ShiftRegisterConvolver
import chiseltest._
import org.scalatest.flatspec.AnyFlatSpec
import org.slf4j.LoggerFactory

case class RandShiftRegConvTestParams(
  bitwidth:     Int,
  inKernels:    Int,
  inChannels:   Int,
  inHeight:     Int,
  inWidth:      Int,
  kernelHeight: Int,
  kernelWidth:  Int)
object RandShiftRegConvTestParams {
  def apply(rand: scala.util.Random, numChannels: Int = 1) = {
    val bw = rand.between(2, 8)
    val inHeight = rand.between(3, 8)
    val inWidth = rand.between(3, 8)
    val kernelHeight = rand.between(2, inHeight)
    val kernelWidth = rand.between(2, inWidth)
    new RandShiftRegConvTestParams(
      bitwidth = bw,
      inKernels = 1,
      inChannels = numChannels,
      inHeight = inHeight,
      inWidth = inWidth,
      kernelHeight = kernelHeight,
      kernelWidth = kernelWidth
    )
  }

  def genShiftRegisterConvolverTestCase(
    p: RandShiftRegConvTestParams
  ): (Seq[Vec[UInt]], lbir.Conv2DConfig) = {
    def tensorValue(c: Int, h: Int, w: Int): Int =
      ((h * p.inWidth + w + c * (p.inHeight * p.inWidth)) % Math.pow(2, p.bitwidth)).toInt

    val inputTensor = lbir.QTensor(
      dtype =
        lbir.Datatype(quantization = UNIFORM, signed = false, bitwidth = p.bitwidth, shift = Seq(0), offset = Seq(0)),
      shape = Seq(p.inKernels, p.inChannels, p.inHeight, p.inWidth),
      values = Seq
        .tabulate(p.inChannels, p.inHeight, p.inWidth)((c, h, w) => tensorValue(c, h, w))
        .flatten
        .flatten
        .map(_.toFloat)
    )
    val kernelTensor = lbir.QTensor(
      //          kernels, channels, height, width
      shape = Seq(p.inKernels, p.inChannels, p.kernelHeight, p.kernelWidth)
    )
    val outputTensor = lbir.QTensor(
      shape = Seq(p.inChannels, 1, p.inHeight - p.kernelHeight + 1, p.inWidth - p.kernelWidth + 1)
    )
    var expectedValues: Seq[Vec[UInt]] = Seq()
    for (ch <- 0 until p.inChannels) {
      val mtrx = DenseMatrix.tabulate(p.inHeight, p.inWidth) { case (h, w) => tensorValue(ch, h, w) }
      for {
        i <- 0 until (p.inHeight - p.kernelHeight + 1)
        j <- 0 until (p.inWidth - p.kernelWidth + 1)
      } {
        val activeWindow = mtrx(i until (i + p.kernelHeight), j until (j + p.kernelWidth)).valuesIterator.toList
          .map(_.toInt.U(p.bitwidth.W))
        expectedValues = expectedValues :+ Vec.Lit(activeWindow: _*)
      }
    }

    val immutableExpectedValues = Seq.empty ++ expectedValues
    val conv2dLayer = lbir.Conv2DConfig(
      kernel = kernelTensor,
      input = inputTensor,
      output = outputTensor,
      depthwise = true
    )
    (immutableExpectedValues, conv2dLayer)
  }
}

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
  val outputParams = lbir.QTensor(
    shape = Seq(1, 1, 2, 2)
  )
  val goldenVec = Seq(
    Vec.Lit(1.U(5.W), 2.U(5.W), 4.U(5.W), 5.U(5.W)),
    Vec.Lit(2.U(5.W), 3.U(5.W), 5.U(5.W), 6.U(5.W)),
    Vec.Lit(4.U(5.W), 5.U(5.W), 7.U(5.W), 8.U(5.W)),
    Vec.Lit(5.U(5.W), 6.U(5.W), 8.U(5.W), 9.U(5.W))
  )

  val conv2dLayer = lbir.Conv2DConfig(
    kernel = kernelParams,
    input = inputParams,
    output = outputParams,
    depthwise = true
  )

  behavior.of("ShiftRegisterConvolver module")
  it should "show appropirate window as it cycles through the input image" in {
    test(new ShiftRegisterConvolver[UInt](conv2dLayer)) { dut =>
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

  val rand = new scala.util.Random(seed = 42)
  for (testId <- 0 until 20) {
    val p = RandShiftRegConvTestParams(rand)
    val (goldenVector, convLayer) =
      RandShiftRegConvTestParams.genShiftRegisterConvolverTestCase(p)
    it should f"Compute random test $testId correctly. Parameters inHeight:${p.inHeight}, " +
      f"inWidth:${p.inWidth}, kernelHeight:${p.kernelHeight}, kernelWidth:${p.kernelWidth}" in {
      test(new ShiftRegisterConvolver[UInt](convLayer)) { dut =>
        dut.io.nextElement.initSource()
        dut.io.nextElement.setSourceClock(dut.clock)
        dut.io.inputActivationsWindow.initSink()
        dut.io.inputActivationsWindow.setSinkClock(dut.clock)

        dut.reset.poke(true.B)
        dut.clock.step()
        dut.reset.poke(false.B)
        fork {
          dut.io.nextElement.enqueueSeq(convLayer.input.values.map(_.toInt.U))
        }.fork {
          dut.io.inputActivationsWindow.expectDequeueSeq(goldenVector)
        }.join()
      }
    }
  }
}

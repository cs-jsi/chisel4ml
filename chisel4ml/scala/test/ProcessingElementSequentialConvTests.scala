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
import _root_.chisel4ml.sequential._
import _root_.lbir.Datatype.QuantizationType.UNIFORM
import _root_.lbir._
import _root_.services._

import _root_.org.slf4j.LoggerFactory
import chisel3._
import firrtl.transforms.NoCircuitDedupAnnotation
import chiseltest._
import org.scalatest.flatspec.AnyFlatSpec

import org.nd4j.linalg.api.ndarray.INDArray
import org.nd4j.linalg.factory.Nd4j
import org.nd4j.linalg.indexing.NDArrayIndex
import org.nd4j.linalg.ops.transforms.Transforms

class ProcessingElementSequentialConvTests extends AnyFlatSpec with ChiselScalatestTester {
  val logger = LoggerFactory.getLogger(classOf[ProcessingElementSequentialConvTests])

  val dtypeUInt4 = lbir.Datatype(quantization = UNIFORM, bitwidth = 4, signed = false, shift = Seq(0), offset = Seq(0))
  val dtypeSInt4 = lbir.Datatype(quantization = UNIFORM, bitwidth = 4, signed = true, shift = Seq(0), offset = Seq(0))

  val testLayer0 = lbir.Layer(
                    ltype  = lbir.Layer.Type.CONV2D,
                    thresh = Option(lbir.QTensor(
                              dtype = Option(dtypeSInt4),
                              shape = Seq(1),
                              values = Seq(0)
                             )),
                    weights = Option(lbir.QTensor(
                                dtype = Option(dtypeSInt4),
                                shape = Seq(1, 1, 2, 2),
                                values = Seq(1, 0,
                                             0, 0)
                              )),
                    input = Option(lbir.QTensor(
                              dtype = Option(dtypeUInt4),
                              shape = Seq(1, 1, 3, 3)
                            )),
                    output = Option(lbir.QTensor(
                              dtype = Option(dtypeSInt4),
                              shape = Seq(1, 1, 2, 2)
                            )),
                    activation = lbir.Layer.Activation.NO_ACTIVATION
                   )

  val testOptions0 = services.GenerateCircuitParams.Options(isSimple = false)


  val dtypeUInt6 = lbir.Datatype(quantization = UNIFORM, bitwidth = 6, signed = false, shift = Seq(0), offset = Seq(0))
  val dtypeSInt7 = lbir.Datatype(quantization = UNIFORM, bitwidth = 7, signed = true, shift = Seq(0), offset = Seq(0))
  val testLayer1 = lbir.Layer(
                    ltype = lbir.Layer.Type.CONV2D,
                    thresh = Option(lbir.QTensor(
                      dtype = Option(dtypeSInt7),
                      shape = Seq(1),
                      values = Seq(-2)
                    )),
                    weights = Option(lbir.QTensor(
                      dtype = Option(dtypeSInt7),
                      shape = Seq(1, 2, 2, 2),
                      values = Seq(1, 0,
                                   0, 0,
                                   0, 0,
                                   1, 0)
                    )),
                    input = Option(lbir.QTensor(
                      dtype = Option(dtypeUInt6),
                      shape = Seq(1, 2, 3, 3),
                    )),
                    output = Option(lbir.QTensor(
                      dtype = Option(dtypeSInt7),
                      shape = Seq(1, 1, 2, 2)
                    ))
                   )

  val dtypeUInt3 = lbir.Datatype(quantization=UNIFORM, bitwidth=3, signed=false, shift = Seq(0, 0), offset = Seq(0,0))
  val dtypeSInt2 = lbir.Datatype(quantization=UNIFORM, bitwidth=2, signed=false, shift = Seq(0, 0), offset = Seq(0,0))
  val dtypeSInt3 = lbir.Datatype(quantization=UNIFORM, bitwidth=3, signed=false, shift = Seq(0, 0), offset = Seq(0,0))

  val testLayer2 = lbir.Layer(
                    ltype = lbir.Layer.Type.CONV2D,
                    thresh = Option(lbir.QTensor(
                      dtype = Option(dtypeSInt3),
                      shape = Seq(2),
                      values = Seq(1, -1)
                    )),
                    weights = Option(lbir.QTensor(
                      dtype = Option(dtypeSInt3),
                      shape = Seq(2, 1, 2, 2),
                      values = Seq( 1,  2,
                                   -2, -1,
                                    2,  0,
                                    0,  2)
                    )),
                    input = Option(lbir.QTensor(
                      dtype = Option(dtypeUInt3),
                      shape = Seq(1, 1, 5, 6),
                    )),
                    output = Option(lbir.QTensor(
                      dtype = Option(dtypeUInt3),
                      shape = Seq(1, 2, 4, 5)
                    ))
                   )


  behavior.of("ProcessingElementSequentialConv module")
  it should "compute the convolution correctly" in { // .withAnnotations(Seq(VerilatorBackendAnnotation))
    test(new ProcessingElementSequentialConv[UInt, SInt, SInt, SInt, SInt, SInt](layer = testLayer0,
                                                                           options = testOptions0,
                                                                           genIn = UInt(4.W),
                                                                           genWeights = SInt(4.W),
                                                                           genAccu = SInt(8.W),
                                                                           genThresh = SInt(4.W),
                                                                           genOut = SInt(4.W),
                                                                           mul = (x:UInt, w: SInt) => (x * w),
                                                                           add = (x: Vec[SInt]) => x.reduceTree(_ +& _),
                                                                           actFn = (x: SInt, y: SInt) => x)).
                                                                           withAnnotations(
                                                                           Seq(VerilatorBackendAnnotation,
                                                                               NoCircuitDedupAnnotation)) { dut =>
      dut.io.inStream.data.initSource()
      dut.io.inStream.data.setSourceClock(dut.clock)
      dut.io.outStream.data.initSink()
      dut.io.outStream.data.setSinkClock(dut.clock)
      dut.clock.step(1)
      dut.io.inStream.data.enqueueSeq(Seq("b1000_0111__0110_0101__0100_0011__0010_0001".U,
                                          "b0000_0000__0000_0000__0000_0000__0000_1001".U))
      dut.io.inStream.last.poke(true.B)
      dut.clock.step(1)
      dut.io.inStream.last.poke(false.B)
      dut.io.outStream.data.expectDequeueSeq(Seq("b0000_0000__0000_0000__0101_0100__0010_0001".U))
    }
  }

  it should "compute a convolution with several channels correctly" in {
    test(new ProcessingElementSequentialConv[UInt, SInt, SInt, SInt, SInt, SInt](layer = testLayer1,
                                                                           options = testOptions0,
                                                                           genIn = UInt(6.W),
                                                                           genWeights = SInt(7.W),
                                                                           genAccu = SInt(8.W),
                                                                           genThresh = SInt(7.W),
                                                                           genOut = SInt(7.W),
                                                                           mul = (x:UInt, w: SInt) => (x * w),
                                                                           add = (x: Vec[SInt]) => x.reduceTree(_ +& _),
                                                                           actFn = (x: SInt, y:SInt) => x + y)).
                                                                           withAnnotations(
                                                                           Seq(VerilatorBackendAnnotation)){ dut =>
      dut.io.inStream.data.initSource()
      dut.io.inStream.data.setSourceClock(dut.clock)
      dut.io.outStream.data.initSink()
      dut.io.outStream.data.setSinkClock(dut.clock)
      dut.clock.step()
      /*  1   2   3   |   1  0  |  1 + 13 - 2 = 12  | 12 14
       *  4   5   6   |   0  0  |  2 + 14 - 2 = 14  | 18 20
       *  7   8   9   |         |  4 + 16 - 2 = 18  |
       *              |         |  5 + 17 - 2 = 20  |
       *  10 11  12   |   0  0  |                   |
       *  13 14  15   |   1  0  |                   |
       *  16 17  18   |         |                   |
       */

      dut.io.inStream.data.enqueueSeq(Seq("b00_000101_000100_000011_000010_000001".U,
                                          "b00_001010_001001_001000_000111_000110".U,
                                          "b00_001111_001110_001101_001100_001011".U,
                                          "b00_000000_000000_010010_010001_010000".U))
      dut.io.inStream.last.poke(true.B)
      dut.clock.step()
      dut.io.inStream.last.poke(false.B)
      dut.io.outStream.data.expectDequeueSeq(Seq("b0000_0010100_0010010_0001110_0001100".U))
    }
  }

  it should "compute a convolution with several kernels correctly" in {
    test(new ProcessingElementSequentialConv[UInt, SInt, SInt, SInt, SInt, UInt](layer = testLayer2,
                                                                           options = testOptions0,
                                                                           genIn = UInt(3.W),
                                                                           genWeights = SInt(3.W),
                                                                           genAccu = SInt(6.W),
                                                                           genThresh = SInt(3.W),
                                                                           genOut = UInt(3.W),
                                                                           mul = (x: UInt, y: SInt) => (x * y),
                                                                           add = (x: Vec[SInt]) => x.reduceTree(_ +& _),
                                                                           actFn = reluFnS)){ dut =>
    /*                         | (bias = -thresh) |
     *  1   2   3   4  5  6    |   1   2   b = -1 | 0 0 0 2 7   7 7 7 7 7
     *  7   6   5   4  3  2    |  -2  -1          | 7 7 7 2 0   7 7 7 7 7
     *  1   0   1   2  3  4    |                  | 0 0 0 0 0   7 7 7 7 7
     *  5   6   7   6  5  4    |   2   0   b = +1 | 7 7 7 7 7   7 7 7 7 7
     *  3   2   1   0  1  2    |   0   2          |
     *                         |                  |
     *
     */

      dut.io.inStream.data.initSource()
      dut.io.inStream.data.setSourceClock(dut.clock)
      dut.io.outStream.data.initSink()
      dut.io.outStream.data.setSinkClock(dut.clock)
      dut.clock.step()
      dut.io.inStream.data.enqueueSeq(Seq("b00_100_101_110_111_110_101_100_011_010_001".U,
                                          "b00_110_101_100_011_010_001_000_001_010_011".U,
                                          "b00_010_001_000_001_010_011_100_101_110_111".U))
      dut.io.inStream.last.poke(true.B)
      dut.clock.step()
      dut.io.inStream.last.poke(false.B)
      dut.io.outStream.data.expectDequeueSeq(Seq("b00_000_010_111_111_111_111_010_000_000_000".U,  // 0x02FFF400
                                                 "b00_111_111_111_111_111_000_000_000_000_000".U,  // 0x3FFF8000
                                                 "b00_111_111_111_111_111_111_111_111_111_111".U,  // 0x3FFFFFFF
                                                 "b00_111_111_111_111_111_111_111_111_111_111".U)) // 0x3FFFFFFF
      dut.clock.step(10)
    }
  }

  val rand = new scala.util.Random(seed = 42)
  Nd4j.getRandom().setSeed(42)

  val imageWidth=28
  val imageHeight=28
  val imageDepth=3
  val numKernels=16
  val kernelSize=3
  for (kernelBW   <- 2 to 8;
       actInpBW   <- 2 to 8) {
    val kernelNormal = Nd4j.rand(Array(numKernels, imageDepth, kernelSize, kernelSize))
    val kernelUnsign = Transforms.round(kernelNormal.mul(scala.math.pow(2, kernelBW)-1))
    val kernel       = Transforms.round(kernelUnsign.sub(scala.math.pow(2, kernelBW-1)))
    val dtypeKernel = lbir.Datatype(quantization = UNIFORM,
                                    bitwidth     = kernelBW,
                                    signed       = true,
                                    shift        = Seq.fill(numKernels)(0),
                                    offset       = Seq.fill(numKernels)(0))
    val dtypeInOut  = lbir.Datatype(quantization = UNIFORM,
                                    bitwidth     = actInpBW,
                                    signed       = false,
                                    shift        = Seq.fill(numKernels)(0),
                                    offset       = Seq.fill(numKernels)(0))
    val dtypeThresh = lbir.Datatype(quantization = UNIFORM,
                                    bitwidth     = 8,
                                    signed       = true,
                                    shift        = Seq.fill(numKernels)(0),
                                    offset       = Seq.fill(numKernels)(0))

    val testLayerAuto = lbir.Layer(ltype = lbir.Layer.Type.CONV2D,
                                   thresh = Option(lbir.QTensor(dtype = Option(dtypeThresh),
                                                                shape = Seq(numKernels),
                                                                values = Seq.tabulate(numKernels)(_.toFloat) // 0, 1...
                                   )),
                                   weights = Option(lbir.QTensor(dtype = Option(dtypeKernel),
                                                                 shape = Seq(numKernels,
                                                                             imageDepth,
                                                                             kernelSize,
                                                                             kernelSize),
                                                                 values = arrToSeq(kernel)
                                   )),
                                   input = Option(lbir.QTensor(dtype = Option(dtypeInOut),
                                                               shape = Seq(1,
                                                                           imageDepth,
                                                                           imageHeight,
                                                                           imageWidth),
                                   )),
                                   output = Option(lbir.QTensor(dtype = Option(dtypeInOut),
                                                                shape = Seq(1,
                                                                            numKernels,
                                                                            imageHeight,
                                                                            imageWidth)
                                   ))
                   )
    it should s"""compute convolution for automatic test parameters: imageWidth=$imageWidth, imageHeight=$imageHeight,
                 |imageDepth=$imageDepth, kernelSize=$kernelSize, numKernels=$numKernels, kernelBW=$kernelBW, actInpBw=
                 |$actInpBW""".stripMargin.replaceAll("\n","") in {
    test(new ProcessingElementSequentialConv[UInt, SInt, SInt, SInt, SInt, UInt](layer = testLayerAuto,
                                                                           options = testOptions0,
                                                                           genIn = UInt(actInpBW.W),
                                                                           genWeights = SInt(kernelBW.W),
                                                                           genAccu = SInt((kernelBW + actInpBW).W),
                                                                           genThresh = SInt(8.W),
                                                                           genOut = UInt(actInpBW.W),
                                                                           mul = (x: UInt, y: SInt) => (x * y),
                                                                           add = (x: Vec[SInt]) => x.reduceTree(_ +& _),
                                                                           actFn = reluFnS)).withAnnotations(
                                                                           Seq(VerilatorBackendAnnotation)){ dut =>
      dut.io.inStream.data.initSource()
      dut.io.inStream.data.setSourceClock(dut.clock)
      dut.io.outStream.data.initSink()
      dut.io.outStream.data.setSinkClock(dut.clock)
      dut.clock.step(50)
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

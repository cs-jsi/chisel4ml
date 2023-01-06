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

import org.scalatest.flatspec.AnyFlatSpec
import chisel3._
import chiseltest._

import _root_.chisel4ml.sequential._
import _root_.chisel4ml.implicits._

import org.nd4j.linalg.factory.Nd4j
import org.nd4j.linalg.api.ndarray.INDArray
import org.nd4j.linalg.indexing.NDArrayIndex
import org.nd4j.linalg.ops.transforms.Transforms

import _root_.org.slf4j.Logger
import _root_.org.slf4j.LoggerFactory

class RollingRegisterFileTests extends AnyFlatSpec with ChiselScalatestTester {
    val logger = LoggerFactory.getLogger(classOf[RollingRegisterFileTests])

    behavior.of("RollingRegisterFile module")
    it should "show appropirate window as it cycles through the input image" in {
        test(new RollingRegisterFile(2, 1, 4)) { dut =>
            dut.io.shiftRegs.poke(false.B)
            dut.io.rowWriteMode.poke(true.B)
            dut.io.chAddr.poke(0.U)
            dut.io.rowAddr.poke(0.U)
            dut.io.inData.poke(0.U)
            dut.io.inValid.poke(false.B)
            dut.clock.step()

            dut.io.rowAddr.poke(0.U)
            dut.io.inValid.poke(true.B)
            dut.io.inData.poke("b0001_0000".U)
            dut.clock.step()

            /* 0  1
             *
             * 0  0
             */
            dut.io.outData.expect("b0000_0000_0001_0000".U)
            dut.io.rowAddr.poke(1.U)
            dut.io.inValid.poke(true.B)
            dut.io.inData.poke("b0011_0010".U)
            dut.clock.step()

            /* 0  1
             *
             * 2  3
             */
            dut.io.outData.expect("b0011_0010_0001_0000".U)
            dut.io.shiftRegs.poke(true.B)
            dut.io.rowWriteMode.poke(false.B)
            dut.io.inData.poke("b0101_0100".U)
            dut.io.inValid.poke(true.B)
            dut.clock.step()

            /* 1  4
             *
             * 3  5
             */
            dut.io.outData.expect("b0101_0011_0100_0001".U)
            dut.io.inValid.poke(false.B)
        }
    }

    val rand = new scala.util.Random(seed = 42)
    Nd4j.getRandom().setSeed(42)
    def ndArrayToBinaryString(arr: INDArray, bits: Int): String = {
        val flatArr      = Nd4j.toFlattened(arr)
        var binaryString = ""
        for (i <- 0 until arr.length) {
            binaryString = toBinary(flatArr.getDouble(i).toInt, bits) + binaryString
        }
        "b" + binaryString
    }
    for (testCaseId <- 0 until 10) { // Set this number to a bigger one for more exhaustive tests
        val randKernSize          = rand.between(2, 7 + 1) // rand.between(inclusive, exclusive)
        val randKernDepth         = rand.between(1, 16 + 1)
        val randKernParamBitwidth = rand.between(1, 8 + 1)
        val randImageSize         = rand.between(randKernSize + 1, randKernSize + 7)
        val randImageNormal       = Nd4j.rand(Array(randKernDepth, randImageSize, randImageSize))
        val randImage             = Transforms.round(randImageNormal.mul(scala.math.pow(2, randKernParamBitwidth) - 1))
        it should s"""work with random params: kernelSize: $randKernSize, kernelDepth: $randKernDepth, kernelBitwidth:
                     |$randKernParamBitwidth, imageSize: $randImageSize.""".stripMargin.replaceAll("\n", "") in {
            test(new RollingRegisterFile(randKernSize, randKernDepth, randKernParamBitwidth)) { dut =>
                logger.debug(s"Simulating test case for random image:\n $randImage.")
                for (i <- 0 until randImageSize - randKernSize + 1) {
                    for (j <- 0 until randImageSize - randKernSize + 1) {
                        var window = randImage.get(
                          NDArrayIndex.all(),                         // kernel
                          NDArrayIndex.interval(i, i + randKernSize), // row
                          NDArrayIndex.interval(j, j + randKernSize)  // col
                        )
                        if (j == 0) {
                            fillWindow(window)
                        } else {
                            fillAdded(
                              window.get(NDArrayIndex.all(), NDArrayIndex.all(), NDArrayIndex.point(randKernSize - 1))
                            )
                        }
                        dut.io.outData.expect(ndArrayToBinaryString(window, randKernParamBitwidth).U)
                    }
                }

                def fillAdded(added: INDArray): Unit = {
                    logger.debug(s"Filling only added registers:\n $added.")
                    dut.io.inValid.poke(false.B)
                    dut.clock.step(rand.between(0, 3)) // random delay to see that there is no timing dependency
                    dut.io.inValid.poke(true.B)
                    dut.io.shiftRegs.poke(true.B)
                    dut.io.rowWriteMode.poke(false.B)
                    for (i <- 0 until added.shape()(0)) { // kernelDepth
                        dut.io.chAddr.poke(i.U)
                        logger.debug(
                          s"row: ${added.getRow(i)} -> ${ndArrayToBinaryString(added.getRow(i), randKernParamBitwidth)}"
                        )
                        dut.io.inData.poke(ndArrayToBinaryString(added.getRow(i), randKernParamBitwidth).U)
                        dut.clock.step()
                        dut.io.shiftRegs.poke(false.B)
                    }
                }

                def fillWindow(window: INDArray): Unit = {
                    logger.debug(s"Refilling entire register file with window:\n $window.")
                    dut.io.inValid.poke(false.B)
                    dut.clock.step(rand.between(0, 3)) // random delay to see there is no timing dependency
                    dut.io.rowWriteMode.poke(true.B)
                    dut.io.shiftRegs.poke(false.B)
                    dut.io.inValid.poke(true.B)
                    for (i <- 0 until window.shape()(0)) { // kernelDepth
                        dut.io.chAddr.poke(i.U)
                        for (j <- 0 until window.shape()(2)) { // kernelSize
                            dut.io.rowAddr.poke(j.U)
                            dut.io.inData.poke(
                              ndArrayToBinaryString(
                                window.get(NDArrayIndex.point(i), NDArrayIndex.point(j)),
                                randKernParamBitwidth
                              ).U
                            )
                            dut.clock.step()
                        }
                    }
                }
            } // end test
        } // end it should "work..."
    } // end test cases for loop
}

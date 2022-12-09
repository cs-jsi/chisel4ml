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
package chisel4ml.sequential

import chisel3._
import chisel3.util._

import _root_.chisel4ml.util.bus.AXIStream
import _root_.chisel4ml.util.{SRAM, ROM}
import _root_.chisel4ml.util.LbirUtil.log2
import _root_.chisel4ml.util.LbirUtil
import _root_.chisel4ml.implicits._
import _root_.lbir.{Layer}
import _root_.services.GenerateCircuitParams.Options
import _root_.scala.math

/** A sequential processing element for convolutions.
 *
 *  This hardware module can handle two-dimensional convolutions of various types, and also can adjust
 *  the aritmetic units depending on the quantization type. It does not take advantage of sparsity.
 *  It uses the filter stationary approach and streams in the activations for each filter sequentialy.
 *  The compute unit computes one whole neuron at once. The reason for this is that it simplifies the
 *  design, which would otherwise require complex control logic / program code. This design, of course,
 *  comes at a price of utilization of the arithmetic units, which is low. But thanks to the low
 *  bitwidths of parameters this should be an acceptable trade-off.
 */
class ProcessingElementSequentialConv(layer: Layer, options: Options)
extends ProcessingElementSequential(layer, options) {
    /****************************/
    /* KERNEL MEMORY            */
    /****************************/
    val kernelParamSize: Int = layer.weights.get.dtype.get.bitwidth
    val kernelParamsPerWord: Int = memWordWidth / kernelParamSize
    val kernelNumParams: Int = layer.weights.get.shape.reduce(_ * _)
    val kernelMemDepth: Int = math.ceil(kernelNumParams.toFloat / kernelParamsPerWord.toFloat).toInt
    val kernelMem = Module(new ROM(depth=kernelMemDepth,
                                   width=memWordWidth,
                                   memFile=LbirUtil.createHexMemoryFile(layer.weights.get)
                               )
                    )

    /****************************/
    /* ACTIVATION MEMORY        */
    /****************************/
    val actParamSize: Int = layer.input.get.dtype.get.bitwidth
    val actParamsPerWord: Int = memWordWidth / actParamSize
    val actNumParams: Int = layer.input.get.shape.reduce(_ * _)
    val actMemDepth: Int = math.ceil(actNumParams.toFloat / actParamsPerWord.toFloat).toInt
    val actMem = Module(new SRAM(depth=actMemDepth, width=memWordWidth))

    /****************************/
    /* KERNEL REGISTERS         */
    /****************************/
    val numOfKernels: Int = layer.weights.get.shape(0)
    val bitsPerKernel: Int = layer.weights.get.totalBitwidth / numOfKernels
    //val kernelRegFile = RegInit(Vec(Seq.fill(kernelNumParams)(0.U(kernelParamSize.W))))

    /****************************/
    /* ACTIVATION REGISTERS     */
    /****************************/
    val actRegs = RegInit(VecInit(Seq.fill(kernelNumParams)(0.U(actParamSize.W))))

}

/** A register file for storing the kernel of a convotional layer.
 *
 *  kernelSize: Int - Signifies one dimension of a square kernel (If its a 3x3 kernel then kernelSize=3)
 *  kernelDepth: Int - The depth of the kernel (number of input channels)
 *  actParamSize: Int - The activation parameterSize in bits.
 */
class KernelRegisterFile(kernelSize: Int, kernelDepth:Int, kernelParamSize: Int) extends Module {
    val totalNumOfElements: Int = kernelSize * kernelSize * kernelDepth
    val kernelNumOfElements: Int = kernelSize * kernelSize
    val outDataSize: Int = kernelSize*kernelSize*kernelDepth*kernelParamSize
    val wrDataWidth: Int = kernelSize * kernelParamSize
    val kernelAddrWidth: Int = math.floor(log2(kernelDepth.toFloat)).toInt + 1
    val rowAddrWidth: Int = math.ceil(log2(kernelSize.toFloat)).toInt

    val io = IO(new Bundle {
        val flushRegs = Input(Bool())
        val shiftRegs = Input(Bool())
        val rowWriteMode = Input(Bool())
        val rowAddr = Input(UInt(rowAddrWidth.W))
        val kernelAddr = Input(UInt(kernelAddrWidth.W))
        val inData = Input(UInt(wrDataWidth.W))
        val inValid = Input(Bool())
        val outData = Output(UInt(outDataSize.W))
    })

    val kernelRegFile = RegInit(VecInit.fill(kernelDepth, kernelSize, kernelSize)(0.U(kernelParamSize.W)))
    //val kernelRegFile = RegInit(VecInit(Array.fill[UInt](kernelDepth, kernelSize, kernelSize)(0.U)))
    //val rowGroupedRegFile: Vec[UInt] = kernelRegFile.grouped(kernelSize).toVec
    io.outData := kernelRegFile.asUInt

    when (io.inValid) {
        when (io.rowWriteMode === true.B) {
            kernelRegFile(io.kernelAddr)(io.rowAddr) := io.inData.asTypeOf(kernelRegFile(0)(0))
        }.otherwise {
            for (i <- 0 until kernelSize) {
                kernelRegFile(io.kernelAddr)(i)(kernelSize-1) := io.inData.asTypeOf(kernelRegFile(0)(0))(i)
            }

            when (io.shiftRegs === true.B) {
                for (i <- 0 until kernelDepth) {
                    for (j <- 0 until kernelSize) {
                        for (k <- 1 until kernelSize) {
                            kernelRegFile(i)(j)(k-1) := kernelRegFile(i)(j)(k)
                        }
                    }
                }
            }
        }
    }
}

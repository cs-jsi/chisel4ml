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
package chisel4ml

import _root_.chisel3._
import _root_.chisel3.util._
import _root_.chisel3.experimental._
import _root_.lbir.{Layer}
import _root_.chisel4ml.{LBIRStream}
import interfaces.amba.axis._
import _root_.services.LayerOptions
import fft._
import dsptools._
import afe._

class AudioFeaturesExtractWrapper(layer: Layer, options: LayerOptions) extends Module with LBIRStream {
  	val fftSize = 512
  	val radix = "2"

	val fftParams = FFTParams.fixed(
    	dataWidth = 12 + 4,
    	binPoint = 4,
    	numPoints = fftSize,
    	decimType = DITDecimType,
    	trimType = RoundHalfUp,
    	useBitReverse = true,
    	windowFunc = WindowFunctionTypes.Hamming(),
    	overflowReg = true,
    	numAddPipes = 1,
    	numMulPipes = 1,
    	sdfRadix = "2",
    	runTime = false,
    	expandLogic =  Array.fill(log2Up(fftSize))(0),
    	keepMSBorLSB = Array.fill(log2Up(fftSize))(true),
  	)

    require(options.busWidthIn == 12, s"${options.busWidthIn}")
    require(options.busWidthOut == layer.output.get.dtype.get.bitwidth, 
            s"${options.busWidthOut} != ${layer.output.get.dtype.get.bitwidth}")
    val inStream = IO(Flipped(AXIStream(UInt(options.busWidthIn.W))))
    val outStream = IO(AXIStream(UInt(options.busWidthOut.W)))

    val afe = Module(new AudioFeaturesExtract(fftParams))


	inStream.ready := afe.io.inStream.ready
    afe.io.inStream.valid := inStream.valid
    afe.io.inStream.bits.real := (inStream.bits ## 0.U(4.W)).asTypeOf(afe.io.inStream.bits.real)
    afe.io.inStream.bits.imag := 0.U.asTypeOf(afe.io.inStream.bits.imag)
    afe.io.inStream.last := inStream.last 

    afe.io.outStream.ready := outStream.ready
	outStream.valid := afe.io.outStream.valid
    outStream.bits := afe.io.outStream.bits.asTypeOf(outStream.bits)
    outStream.last := afe.io.outStream.last
}

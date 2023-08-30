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

class FFTWrapper(layer: Layer, options: LayerOptions) extends Module with LBIRStream {
  	val fftSize = 512

	val fftParams = FFTParams.fixed(
    	dataWidth = 32,
    	binPoint = 16,
    	numPoints = fftSize,
    	decimType = DITDecimType,
    	trimType = RoundHalfToEven,
        twiddleWidth = 32,
    	useBitReverse = true,
    	//windowFunc = WindowFunctionTypes.None(), //WindowFunctionTypes.Hamming(32),
    	overflowReg = true,
    	numAddPipes = 1,
    	numMulPipes = 1,
    	sdfRadix = "2",
    	runTime = false,
    	expandLogic =  Array.fill(log2Up(fftSize))(1),
    	keepMSBorLSB = Array.fill(log2Up(fftSize))(true),
  	)

    //require(options.busWidthIn == 12, s"${options.busWidthIn}")
    require(options.busWidthOut == layer.output.get.dtype.get.bitwidth, 
            s"${options.busWidthOut} != ${layer.output.get.dtype.get.bitwidth}")
    val inStream = IO(Flipped(AXIStream(UInt(options.busWidthIn.W))))
    val outStream = IO(AXIStream(UInt(options.busWidthOut.W)))

    val sdffft = Module(new SDFFFT(fftParams))

    // Fix discrepancy between last signal semantics of LBIRDriver and FFT.
    // FFT has per frame last signals, LBIRDriver per tensor last signal.
    val (_, fftCounterWrap) = Counter(inStream.fire, fftParams.numPoints)

    object fftState extends ChiselEnum {
        val sWAIT = Value(0.U)
        val sREADY = Value(1.U)
    }
    val state = RegInit(fftState.sREADY)

    when (state === fftState.sREADY && fftCounterWrap) {
        state := fftState.sWAIT
    }.elsewhen(state === fftState.sWAIT && outStream.last) {
        state := fftState.sREADY
    }


	inStream.ready := state === fftState.sREADY
    sdffft.io.in.valid := inStream.valid
    sdffft.io.in.bits.real := (inStream.bits(32-1-16, 0) ## 0.U(16.W)).asTypeOf(sdffft.io.in.bits.real)
    sdffft.io.in.bits.imag := 0.U.asTypeOf(sdffft.io.in.bits.imag)
    sdffft.io.lastIn := inStream.last 

    sdffft.io.out.ready := outStream.ready
	outStream.valid := sdffft.io.out.valid
    outStream.bits := sdffft.io.out.bits.real.asTypeOf(outStream.bits)
    outStream.last := sdffft.io.lastOut
}

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
	val wordSize = 13
  	val fftSize = 512
  	val isBitReverse = true
  	val radix = "2"
  	val separateVerilog = true

	val fftParams = FFTParams.fixed(
    	dataWidth = wordSize,
    	binPoint = 0,
    	trimEnable= false,
    	//dataWidthOut = 16, // only appied when trimEnable=True
    	//binPointOut = 0,
    	twiddleWidth = 16,
    	numPoints = fftSize,
    	decimType = DIFDecimType,
    	trimType = RoundHalfUp,
    	useBitReverse = isBitReverse,
    	windowFunc = WindowFunctionTypes.Hamming(),
    	overflowReg = true,
    	numAddPipes = 1,
    	numMulPipes = 1,
    	sdfRadix = radix,
    	runTime = false,
    	expandLogic =  Array.fill(log2Up(fftSize))(1),
    	keepMSBorLSB = Array.fill(log2Up(fftSize))(true),
    	minSRAMdepth = 8
  	)

    require(options.busWidthIn == wordSize)
    require(options.busWidthOut == layer.output.get.dtype.get.bitwidth)
    val inStream = IO(Flipped(AXIStream(UInt(options.busWidthIn.W))))
    val outStream = IO(AXIStream(UInt(options.busWidthOut.W)))

    val afe = Module(new AudioFeaturesExtract(fftParams))

    // This counter fixes the discrepancy between the last signal semantics of LBIRDriver and fft.
    // The fft wants per frame last signals, while LBIRDriver provides per tensor last signal.
    val (_, fftCounterWrap) = Counter(inStream.fire, fftSize)

	object afeState extends ChiselEnum {
    	val sWAIT  = Value(0.U)
    	val sREADY = Value(1.U)
  	}
  	val state  = RegInit(afeState.sREADY)

	// STATE MACHINE
	when (state === afeState.sREADY && fftCounterWrap) {
		state := afeState.sWAIT
	}.elsewhen(state === afeState.sWAIT && !afe.io.busy) {
		state := afeState.sREADY
	}


	inStream.ready := state === afeState.sREADY
    afe.io.inStream.valid := inStream.valid
    afe.io.inStream.bits.real := inStream.bits.asTypeOf(afe.io.inStream.bits.real)
    afe.io.inStream.bits.imag := 0.U.asTypeOf(afe.io.inStream.bits.imag)
    afe.io.inStream.last := inStream.last || fftCounterWrap

    afe.io.outStream.ready := outStream.ready
	outStream.valid := afe.io.outStream.valid
    outStream.bits := afe.io.outStream.bits.asTypeOf(outStream.bits)
    outStream.last := afe.io.outStream.last
}

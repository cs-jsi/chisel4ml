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

import chisel3._
import chisel3.util._
import chisel4ml.HasLBIRStream
import lbir.FFTConfig
import dsptools._
import fft._
import interfaces.amba.axis._
import org.chipsalliance.cde.config.{Field, Parameters}
import chisel4ml.logging.HasParameterLogging
import fixedpoint._

case object FFTConfigField extends Field[FFTConfig]

trait HasFFTParameters extends HasLBIRStreamParameters[FFTConfig] {
  type T = FFTConfig
  val p: Parameters
  val cfg = p(FFTConfigField)
  require(numBeatsIn == 1)
  require(numBeatsOut == 1)
  val fftParams = FFTParams.fixed(
    dataWidth = 24,
    binPoint = 12,
    trimEnable = false,
    numPoints = cfg.fftSize,
    decimType = DITDecimType,
    trimType = RoundHalfToEven,
    twiddleWidth = 16,
    useBitReverse = true,
    overflowReg = true,
    numAddPipes = 1,
    numMulPipes = 1,
    sdfRadix = "2",
    runTime = false,
    expandLogic = Array.fill(log2Up(cfg.fftSize))(1),
    keepMSBorLSB = Array.fill(log2Up(cfg.fftSize))(true)
  )
}

class FFTWrapper(implicit val p: Parameters)
    extends Module
    with HasLBIRStream
    with HasFFTParameters
    with HasParameterLogging {
  logParameters
  val inStream = IO(Flipped(AXIStream(SInt(cfg.input.dtype.bitwidth.W), numBeatsIn)))
  val outStream = IO(AXIStream(SInt(cfg.output.dtype.bitwidth.W), numBeatsOut))

  val window = VecInit(cfg.winFn.map(_.F(16.W, 16.BP)))
  val sdffft = Module(new SDFFFT(fftParams))

  // Fix discrepancy between last signal semantics of LBIRDriver and FFT.
  // FFT has per frame last signals, LBIRDriver per tensor last signal.
  val (fftCounter, fftCounterWrap) = Counter(0 until fftParams.numPoints, inStream.fire)
  val (_, outCounterWrap) = Counter(0 until cfg.numFrames, sdffft.io.lastOut)

  object fftState extends ChiselEnum {
    val sWAIT = Value(0.U)
    val sREADY = Value(1.U)
  }
  val state = RegInit(fftState.sREADY)

  when(state === fftState.sREADY && fftCounterWrap) {
    state := fftState.sWAIT
  }.elsewhen(state === fftState.sWAIT && sdffft.io.lastOut) {
    state := fftState.sREADY
  }

  inStream.ready := state === fftState.sREADY
  sdffft.io.in.valid := inStream.valid && state === fftState.sREADY
  val currWindow = window(fftCounter).asUInt.zext
  dontTouch(currWindow)
  // U(12, 0) x S(0, 16) => S(12, 16) >> 4 => S(12,12)
  val windowedSignal = (inStream.bits.head.asUInt.asSInt * currWindow) >> 4
  sdffft.io.in.bits.real := windowedSignal.asTypeOf(sdffft.io.in.bits.real)
  sdffft.io.in.bits.imag := 0.U.asTypeOf(sdffft.io.in.bits.imag)
  sdffft.io.lastIn := inStream.last || fftCounterWrap

  sdffft.io.out.ready := outStream.ready
  outStream.valid := sdffft.io.out.valid
  outStream.bits := sdffft.io.out.bits.real.asTypeOf(outStream.bits)
  outStream.last := outCounterWrap
}

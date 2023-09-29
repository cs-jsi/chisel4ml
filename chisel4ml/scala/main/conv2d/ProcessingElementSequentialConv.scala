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

import chisel4ml.LBIRStream
import chisel4ml.implicits._
import chisel4ml.util.reluFn
import lbir.Activation._
import lbir.Conv2DConfig
import lbir.Datatype.QuantizationType._
import org.slf4j.LoggerFactory
import services.LayerOptions
import chisel3._
import chisel4ml.MemWordSize
import interfaces.amba.axis._
import memories.MemoryGenerator

import scala.reflect.runtime.universe._
import memories.SRAMWrite

/** A sequential processing element for convolutions.
  *
  * This hardware module can handle two-dimensional convolutions of various types, and also can adjust the aritmetic
  * units depending on the quantization type. It does not take advantage of sparsity. It uses the filter stationary
  * approach and streams in the activations for each filter sequentialy. The compute unit computes one whole neuron at
  * once. The reason for this is that it simplifies the design, which would otherwise require complex control logic /
  * program code. This design, of course, comes at a price of utilization of the arithmetic units, which is low. But
  * thanks to the low bitwidths of parameters this should be an acceptable trade-off.
  */

class ProcessingElementSequentialConv[
  I <: Bits with Num[I]: TypeTag,
  W <: Bits with Num[W]: TypeTag,
  M <: Bits,
  S <: Bits: TypeTag,
  A <: Bits: TypeTag,
  O <: Bits: TypeTag
](layer:   Conv2DConfig,
  options: LayerOptions,
  mul:     (I, W) => M,
  add:     Vec[M] => S,
  actFn:   (S, A) => O)
    extends Module
    with LBIRStream {
  val logger = LoggerFactory.getLogger("ProcessingElementSequentialConv")

  def genType[T <: Bits: TypeTag](bitwidth: Int): T = {
    val tpe = implicitly[TypeTag[T]].tpe
    val hwType =
      if (tpe =:= typeOf[UInt]) UInt(bitwidth.W)
      else if (tpe =:= typeOf[SInt]) SInt(bitwidth.W)
      else throw new NotImplementedError
    hwType.asInstanceOf[T]
  }

  val genIn = genType[I](layer.input.dtype.bitwidth)
  val genWeights = genType[W](layer.kernel.dtype.bitwidth)
  val genAccu = genType[S](layer.input.dtype.bitwidth + layer.kernel.dtype.bitwidth)
  val genThresh = genType[A](layer.thresh.dtype.bitwidth)
  val genOut = genType[O](layer.output.dtype.bitwidth)

  logger.info(s"""Generated new ProcessingElementSequentialConv with input shape:${layer.input.shape}, input dtype:
          | ${layer.input.dtype}. Number of kernel parameters is ${layer.kernel.numKernelParams}.""")

  val inStream = IO(Flipped(AXIStream(UInt(options.busWidthIn.W))))
  val outStream = IO(AXIStream(UInt(options.busWidthOut.W)))

  val kernelMem = Module(MemoryGenerator.SRAMInitFromString(hexStr = layer.kernel.toHexStr, width = MemWordSize.bits))
  val actMem = Module(MemoryGenerator.SRAM(depth = layer.input.memDepth, width = MemWordSize.bits))
  val resMem = Module(MemoryGenerator.SRAM(depth = layer.output.memDepth, width = MemWordSize.bits))

  val krf = Module(new KernelRegisterFile(layer.kernel))

  val actRegFile = Module(new RollingRegisterFile(layer.input, layer.kernel))

  val dynamicNeuron = Module(
    new DynamicNeuron[I, W, M, S, A, O](
      genIn = genIn,
      numSynaps = layer.kernel.numKernelParams,
      genWeights = genWeights,
      genAccu = genAccu,
      genThresh = genThresh,
      genOut = genOut,
      mul = mul,
      add = add,
      actFn = actFn
    )
  )

  val swu = Module(new SlidingWindowUnit(input = layer.input, kernel = layer.kernel))

  val kRFLoader = Module(new KernelRFLoader(layer.kernel))

  val tas = Module(new ThreshAndShiftUnit[A](genThresh = genThresh, thresh = layer.thresh, kernel = layer.kernel))

  val rmb = Module(new ResultMemoryBuffer[O](genOut = genOut, output = layer.output))

  val ctrl = Module(new PeSeqConvController(layer))

  kernelMem.io.write.enable := false.B // io.kernelMemWrEna
  kernelMem.io.write.address := 0.U // io.kernelMemWrAddr
  kernelMem.io.write.data := 0.U // io.kernelMemWrData
  kernelMem.io.read <> kRFLoader.io.rom

  actMem.io.read <> swu.io.actMem
  resMem.io.write <> rmb.io.resRam
  krf.io.write <> kRFLoader.io.krf

  actRegFile.io.shiftRegs := swu.io.shiftRegs
  actRegFile.io.rowWriteMode := swu.io.rowWriteMode
  actRegFile.io.rowAddr := swu.io.rowAddr
  actRegFile.io.chAddr := swu.io.chAddr
  actRegFile.io.inData := swu.io.data
  actRegFile.io.inValid := swu.io.valid

  rmb.io.resultValid := RegNext(swu.io.imageValid)
  rmb.io.result := dynamicNeuron.io.out
  rmb.io.start := ctrl.io.rmbStart

  dynamicNeuron.io.in := actRegFile.io.outData
  dynamicNeuron.io.weights := krf.io.kernel
  dynamicNeuron.io.thresh := tas.io.thresh
  dontTouch(dynamicNeuron.io.thresh)
  dontTouch(dynamicNeuron.io.shift)
  dontTouch(dynamicNeuron.io.shiftLeft)
  dontTouch(tas.io)
  dynamicNeuron.io.shift := tas.io.shift
  dynamicNeuron.io.shiftLeft := tas.io.shiftLeft

  tas.io.start := ctrl.io.swuStart
  tas.io.nextKernel := ctrl.io.krfLoadKernel

  swu.io.start := ctrl.io.swuStart
  ctrl.io.swuEnd := swu.io.end

  ctrl.io.krfReady := kRFLoader.io.kernelReady
  kRFLoader.io.loadKernel := ctrl.io.krfLoadKernel
  kRFLoader.io.kernelNum := ctrl.io.krfKernelNum

  inStream.ready := ctrl.io.inStreamReady
  actMem.io.write.enable := inStream.ready && inStream.valid
  actMem.io.write.address := ctrl.io.actMemAddr
  actMem.io.write.data := inStream.bits
  ctrl.io.inStreamLast := inStream.last
  ctrl.io.inStreamValid := inStream.valid

  outStream.valid := ctrl.io.outStreamValid
  outStream.bits := resMem.io.read.data
  outStream.last := ctrl.io.outStreamLast
  ctrl.io.outStreamReady := outStream.ready

  resMem.io.read.enable := ctrl.io.resMemEna
  resMem.io.read.address := ctrl.io.resMemAddr
}

object ProcessingElementSequentialConv {
  def apply(layer: Conv2DConfig, options: LayerOptions) = (
    layer.input.dtype.quantization,
    layer.input.dtype.signed,
    layer.kernel.dtype.quantization,
    layer.activation
  ) match {
    case (UNIFORM, true, UNIFORM, RELU) =>
      new ProcessingElementSequentialConv[SInt, SInt, SInt, SInt, SInt, UInt](
        layer,
        options,
        mul = (x: SInt, y: SInt) => x * y,
        add = (x: Vec[SInt]) => x.reduceTree(_ +& _),
        actFn = reluFn
      )
    case (UNIFORM, false, UNIFORM, RELU) =>
      new ProcessingElementSequentialConv[UInt, SInt, SInt, SInt, SInt, UInt](
        layer,
        options,
        mul = (x: UInt, y: SInt) => x * y,
        add = (x: Vec[SInt]) => x.reduceTree(_ +& _),
        actFn = reluFn
      )
    case _ => throw new RuntimeException()
  }
}

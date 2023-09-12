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

import scala.reflect.runtime.universe._
import interfaces.amba.axis._
import _root_.chisel4ml.lbir._
import _root_.chisel4ml.LBIRStream
import chisel4ml.MemWordSize
import memories.MemoryGenerator
import _root_.chisel4ml.util._
import _root_.chisel4ml.implicits._
import _root_.lbir.{Conv2DConfig, QTensor}
import _root_.lbir.Datatype.QuantizationType._
import _root_.lbir.Activation._
import _root_.services.LayerOptions
import chisel3._



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
    O <: Bits: TypeTag,
  ](
    layer:      Conv2DConfig,
    options:    LayerOptions,
    mul:        (I, W) => M,
    add:        Vec[M] => S,
    actFn:      (S, A) => S,
  ) extends Module with LBIRStream {
  def gen[T <: Bits: TypeTag](bitwidth: Int): T = {
    val tpe = implicitly[TypeTag[T]].tpe
    val hwType = if (tpe =:= typeOf[UInt]) UInt(bitwidth.W)
    else if (tpe =:= typeOf[SInt]) SInt(bitwidth.W)
    else throw new NotImplementedError
    hwType.asInstanceOf[T]
  }


  val genIn = gen[I](layer.input.dtype.bitwidth)
  val genWeights = gen[W](layer.kernel.dtype.bitwidth)
  val genAccu = gen[S](layer.input.dtype.bitwidth + layer.kernel.dtype.bitwidth)
  val genThresh = gen[A](layer.thresh.dtype.bitwidth)
  val genOut = gen[O](layer.output.dtype.bitwidth)

  
  val inStream = IO(Flipped(AXIStream(UInt(options.busWidthIn.W))))
  val outStream = IO(AXIStream(UInt(options.busWidthOut.W)))
  val kernelMem = Module(MemoryGenerator.SRAMInitFromString(hexStr=layer.kernel.toHexStr,
                                                            width=MemWordSize.bits))

  val actMem = Module(MemoryGenerator.SRAM(depth = layer.input.memDepth,
										   width = MemWordSize.bits))

  val krf = Module(new KernelRegisterFile(kernelSize = layer.kernel.width,
                                          kernelDepth = layer.kernel.numChannels,
                                          kernelParamSize = layer.kernel.dtype.bitwidth))

  val actRegFile = Module(new RollingRegisterFile(kernelSize = layer.kernel.width,
                                                  kernelDepth = layer.kernel.numChannels,
                                                  paramSize = layer.input.dtype.bitwidth))

  val resMem = Module(MemoryGenerator.SRAM(depth = layer.output.memDepth,
                                           width = MemWordSize.bits))

  val dynamicNeuron = Module(new DynamicNeuron[I, W, M, S, A, O](genIn = genIn,
                                                                 numSynaps = layer.kernel.numKernelParams,
                                                                 genWeights = genWeights,
                                                                 genAccu = genAccu,
                                                                 genThresh = genThresh,
                                                                 genOut = genOut,
                                                                 mul = mul,
                                                                 add = add,
                                                                 actFn = actFn))

  val swu = Module(new SlidingWindowUnit(kernelSize = layer.kernel.width,
                                         kernelDepth = layer.kernel.numChannels,
                                         actWidth = layer.input.width,
                                         actHeight = layer.input.height,
                                         actParamSize = layer.input.dtype.bitwidth))

  val kRFLoader = Module(new KernelRFLoader(kernelSize = layer.kernel.width,
                                            kernelDepth = layer.kernel.numChannels,
                                            kernelParamSize = layer.kernel.dtype.bitwidth,
                                            numKernels = layer.kernel.numKernels))

  val tas = Module(new ThreshAndShiftUnit[A](numKernels = layer.kernel.numKernels,
                                             genThresh = genThresh,
                                             layer = layer))

  val rmb = Module(new ResultMemoryBuffer[O](genOut = genOut,
                                             resultsPerKernel = layer.output.height * layer.output.width,
                                             resMemDepth = layer.output.memDepth,
                                             numKernels = layer.kernel.numKernels))

  val paramsPerTrans = math.floor(options.busWidthIn.toFloat / layer.input.dtype.bitwidth.toFloat).toInt
  val actMemTransfers = math.ceil(layer.input.numParams.toFloat / paramsPerTrans.toFloat).toInt
  val ctrl = Module(new PeSeqConvController(numKernels = layer.kernel.numKernels,
                                            resMemDepth = layer.output.memDepth,
                                            actMemDepth = layer.input.memDepth,
                                            actMemTransfers = actMemTransfers))

  kernelMem.io.rdEna  := kRFLoader.io.romRdEna
  kernelMem.io.rdAddr := kRFLoader.io.romRdAddr
  kRFLoader.io.romRdData := kernelMem.io.rdData

  kernelMem.io.wrEna  := false.B // io.kernelMemWrEna
  kernelMem.io.wrAddr := 0.U // io.kernelMemWrAddr
  kernelMem.io.wrData := 0.U // io.kernelMemWrData

  actMem.io.rdEna  := swu.io.actRdEna
  actMem.io.rdAddr := swu.io.actRdAddr
  swu.io.actRdData := actMem.io.rdData

  resMem.io.wrEna  := rmb.io.resRamEn
  resMem.io.wrAddr := rmb.io.resRamAddr
  resMem.io.wrData := rmb.io.resRamData

  krf.io.chAddr  := kRFLoader.io.chAddr
  krf.io.rowAddr := kRFLoader.io.rowAddr
  krf.io.colAddr := kRFLoader.io.colAddr
  krf.io.inData  := kRFLoader.io.data
  krf.io.inValid := kRFLoader.io.valid

  actRegFile.io.shiftRegs    := swu.io.shiftRegs
  actRegFile.io.rowWriteMode := swu.io.rowWriteMode
  actRegFile.io.rowAddr      := swu.io.rowAddr
  actRegFile.io.chAddr       := swu.io.chAddr
  actRegFile.io.inData       := swu.io.data
  actRegFile.io.inValid      := swu.io.valid

  rmb.io.resultValid := RegNext(swu.io.imageValid)
  rmb.io.result      := dynamicNeuron.io.out
  rmb.io.start       := ctrl.io.rmbStart

  dynamicNeuron.io.in        := actRegFile.io.outData
  dynamicNeuron.io.weights   := krf.io.outData
  dynamicNeuron.io.thresh    := tas.io.thresh
  dynamicNeuron.io.shift     := tas.io.shift
  dynamicNeuron.io.shiftLeft := tas.io.shiftLeft

  tas.io.start  := ctrl.io.swuStart
  tas.io.nextKernel := ctrl.io.krfLoadKernel

  swu.io.start   := ctrl.io.swuStart
  ctrl.io.swuEnd := swu.io.end

  ctrl.io.krfReady        := kRFLoader.io.kernelReady
  kRFLoader.io.loadKernel := ctrl.io.krfLoadKernel
  kRFLoader.io.kernelNum  := ctrl.io.krfKernelNum

  inStream.ready         := ctrl.io.inStreamReady
  actMem.io.wrEna        := inStream.ready && inStream.valid
  actMem.io.wrAddr       := ctrl.io.actMemAddr
  actMem.io.wrData       := inStream.bits
  ctrl.io.inStreamLast   := inStream.last
  ctrl.io.inStreamValid  := inStream.valid

  outStream.valid := ctrl.io.outStreamValid
  outStream.bits  := resMem.io.rdData
  outStream.last  := ctrl.io.outStreamLast
  ctrl.io.outStreamReady  := outStream.ready

  resMem.io.rdEna  := ctrl.io.resMemEna
  resMem.io.rdAddr := ctrl.io.resMemAddr
}

object ProcessingElementSequentialConv {
  def reluFn(act: SInt, thresh: SInt): SInt = Mux((act - thresh) > 0.S, (act - thresh), 0.S)
  def apply(layer: Conv2DConfig, options: LayerOptions) = (layer.input.dtype.quantization,
                                                           layer.input.dtype.signed,
                                                           layer.kernel.dtype.quantization,
                                                           layer.activation) match {
    case (UNIFORM, true, UNIFORM, RELU) => new ProcessingElementSequentialConv[SInt, SInt, SInt, SInt, SInt, SInt](
                                                            layer,
                                                            options,
                                                            mul = (x: SInt, y: SInt) => x * y,
                                                            add = (x: Vec[SInt]) => x.reduceTree(_ +& _),
                                                            actFn = reluFn
                                                        )
    case (UNIFORM, false, UNIFORM, RELU) => new ProcessingElementSequentialConv[UInt, SInt, SInt, SInt, SInt, SInt](
                                                            layer,
                                                            options,
                                                            mul = (x: UInt, y: SInt) => x * y,
                                                            add = (x: Vec[SInt]) => x.reduceTree(_ +& _),
                                                            actFn = reluFn
                                                        )
  }
}

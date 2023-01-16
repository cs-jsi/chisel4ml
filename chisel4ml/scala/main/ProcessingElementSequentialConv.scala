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

import _root_.chisel4ml.lbir._
import _root_.chisel4ml.memory.{ROM, SRAM}
import _root_.chisel4ml.util._
import _root_.lbir.Layer
import _root_.scala.math
import _root_.services.GenerateCircuitParams.Options
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
    I <: Bits,
    W <: Bits: WeightsProvider,
    M <: Bits,
    S <: Bits: ThreshProvider,
    A <: Bits: ThreshProvider,
    O <: Bits,
  ](
    layer:      Layer,
    options:    Options,
    genIn:      I,
    genWeights: W,
    genAccu:    S,
    genThresh:  A,
    genOut:     O,
    mul:        (I, W) => M,
    add:        Vec[M] => S,
    actFn:      (S, A) => O,
  ) extends ProcessingElementSequential(layer, options) {

  val cfg = ProcessingElementSequentialConvConfig(layer)

  val kernelMem = Module(new ROM(depth = cfg.kernel.mem.depth, //kernelMemDepth,
                                 width = MemWordSize.bits, //memWordWidth,
                                 memFile = genHexMemoryFile(layer.weights.get, layout = "CDHW")))

  println(s"actMem depth: ${cfg.input.mem.depth}")
  val actMem = Module(new SRAM(depth = cfg.input.mem.depth, //actMemDepth,
                               width = MemWordSize.bits)) //memWordWidth))

  val krf = Module(new KernelRegisterFile(kernelSize = cfg.kernel.width,
                                          kernelDepth = cfg.kernel.numChannels,
                                          kernelParamSize = cfg.kernel.paramBitwidth))

  val actRegFile = Module(new RollingRegisterFile(kernelSize = cfg.kernel.width,
                                                  kernelDepth = cfg.kernel.numChannels,
                                                  paramSize = cfg.input.paramBitwidth))

  val resMem = Module(new SRAM(depth = cfg.result.mem.depth, //resMemDepth,
                               width = MemWordSize.bits))

  val dynamicNeuron = Module(new DynamicNeuron[I, W, M, S, A, O](genIn = genIn,
                                                              numSynaps = cfg.kernel.numKernelParams,
                                                              genWeights = genWeights,
                                                              genAccu = genAccu,
                                                              genThresh = genThresh,
                                                              genOut = genOut,
                                                              mul = mul,
                                                              add = add,
                                                              actFn = actFn))

  val swu = Module(new SlidingWindowUnit(kernelSize = cfg.kernel.width,
                                         kernelDepth = cfg.kernel.numChannels,
                                         actWidth = cfg.input.width,
                                         actHeight = cfg.input.height,
                                         actParamSize = cfg.input.paramBitwidth))

  val kRFLoader = Module(new KernelRFLoader(kernelSize = cfg.kernel.width,
                                            kernelDepth = cfg.kernel.numChannels,
                                            kernelParamSize = cfg.kernel.paramBitwidth,
                                            numKernels = cfg.kernel.numKernels))

  val tas = Module(new ThreshAndShiftUnit[A](numKernels = cfg.kernel.numKernels,
                                             genThresh = genThresh,
                                             layer = layer))

  val rmb = Module(new ResultMemoryBuffer[O](genOut = genOut,
                                             resultsPerKernel = cfg.result.height * cfg.result.width,
                                             resMemDepth = cfg.result.mem.depth,
                                             numKernels = cfg.kernel.numKernels))

  val ctrl = Module(new PeSeqConvController(numKernels = cfg.kernel.numKernels,
                                            resMemDepth = cfg.result.mem.depth,
                                            actMemDepth = cfg.input.mem.depth))

  kernelMem.io.rdEna  := kRFLoader.io.romRdEna
  kernelMem.io.rdAddr := kRFLoader.io.romRdAddr
  kRFLoader.io.romRdData := kernelMem.io.rdData

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
  rmb.io.start       := ctrl.io.swuStart

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

  io.inStream.data.ready := ctrl.io.inStreamReady
  actMem.io.wrEna        := io.inStream.data.ready && io.inStream.data.valid
  actMem.io.wrAddr       := ctrl.io.actMemAddr
  actMem.io.wrData       := io.inStream.data.bits
  ctrl.io.inStreamLast   := io.inStream.last
  ctrl.io.inStreamValid  := io.inStream.data.valid

  io.outStream.data.valid := ctrl.io.outStreamValid
  io.outStream.data.bits  := resMem.io.rdData
  io.outStream.last       := ctrl.io.outStreamLast
  ctrl.io.outStreamReady  := io.outStream.data.ready

  resMem.io.rdEna  := ctrl.io.resMemEna
  resMem.io.rdAddr := ctrl.io.resMemAddr
}

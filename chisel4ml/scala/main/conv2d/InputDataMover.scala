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
package chisel4ml.conv2d

import chisel3._
import chisel3.util._
import memories.SRAMRead
import chisel4ml.implicits._
import chisel4ml.MemWordSize

class InputDataMover(input: lbir.QTensor, kernel: lbir.QTensor) extends Module {
  val io = IO(new Bundle {
    // interface to the RollingRegisterFile module.
    val shiftRegs = Output(Bool())
    val rowWriteMode = Output(Bool())
    val rowAddr = Output(UInt(log2Up(kernel.width).W))
    val chAddr = Output(UInt(log2Up(kernel.numChannels).W))
    val data = Output(UInt((kernel.width * input.dtype.bitwidth).W))
    val valid = Output(Bool())
    val imageValid = Output(Bool())

    // interface to the activation memory
    val actMem = Flipped(new SRAMRead(input.memDepth, MemWordSize.bits))

    // control interface
    val start = Input(Bool())
    val end = Output(Bool())
  })

}

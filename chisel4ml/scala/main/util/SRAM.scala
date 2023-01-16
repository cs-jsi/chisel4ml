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
/*
 * SYNTHESIZES TO BLOCK RAM (or several block rams, depending on the depth)
 * (tested in vivado 2021.1)
 */
package chisel4ml.memory
import _root_.chisel4ml.util.reqWidth
import chisel3._

class SRAM(depth: Int, width: Int = 32) extends Module {
  val io = IO(new Bundle {
    val rdEna  = Input(Bool())
    val rdAddr = Input(UInt(reqWidth(depth).W))
    val rdData = Output(UInt(width.W))
    val wrEna  = Input(Bool())
    val wrAddr = Input(UInt(reqWidth(depth).W))
    val wrData = Input(UInt(width.W))
  })
  val mem = SyncReadMem(depth, UInt(width.W))

  // Create one write port and one read port
  when(io.wrEna) {
    mem.write(io.wrAddr, io.wrData)
  }
  io.rdData := mem.read(io.rdAddr, io.rdEna)
}

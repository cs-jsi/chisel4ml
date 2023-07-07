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
package chisel4ml.tests

import _root_.chisel4ml.sequential._
import _root_.chisel4ml.util._
import _root_.org.slf4j.LoggerFactory
import chisel3._
import chisel3.experimental.VecLiterals._
import chiseltest._
import org.scalatest.flatspec.AnyFlatSpec

class DynamicNeuronTests extends AnyFlatSpec with ChiselScalatestTester {
  val logger = LoggerFactory.getLogger(classOf[DynamicNeuronTests])

  behavior.of("DynamicNeuron module")
  it should "Compute the right value" in {
    test(
      new DynamicNeuron[UInt, SInt, SInt, SInt, SInt, UInt](
        genIn = UInt(4.W),
        numSynaps = 4,
        genWeights = SInt(4.W),
        genAccu = SInt(4.W),
        genThresh = SInt(4.W),
        genOut = UInt(4.W),
        mul = (i: UInt, w: SInt) => i * w,
        add = (x: Vec[SInt]) => x.reduceTree(_ +& _),
        actFn = reluFnS,
      ),
    ) { dut =>

      dut.io.in.poke("b0011_0010_0001_0000".U)
      dut.io.weights.poke("b0001_0001_0001_0001".U)
      dut.io.thresh.poke(0.S)
      dut.io.shift.poke(0.U)
      dut.io.shiftLeft.poke(true.B)
      dut.clock.step()
      dut.io.out.expect(6.U)
      dut.clock.step()
    }
  }
}

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
import chisel4ml.implicits._
import freechips.rocketchip.diplomacy._
import lbir.DenseConfig
import org.chipsalliance.cde.config._
import chisel4ml.bitstream._
import lbir._
import scala.language.existentials

case class PEWrapsSimpleParameters(
  layers:   Seq[LayerWrap with HasInputOutputQTensor],
  numPipes: Int)

/**
  * ProcessingElementWrapSimpleDense
  *
  * Wraps a series of simple (combinatorial) dense layers with a BitStream interface.
  *
  * @param layers - The dense layers to wrap with sequential logic
  * @param numPipes - number of pipeline stages to add between the combinational logic
  * @param p - implicit parameters
  */
class ProcessingElementWrapSimpleDense[I <: Bits, O <: Bits](
  layers:                  Seq[DenseConfig],
  numPipes:                Int
)(override implicit val p: Parameters)
    extends ProcessingElement[I, O] {
  val sNode = BSSlaveNode[I](
    BSSlaveParameters[I](
      layers.head.input,
      layers.head.input.getType[I],
      None
    )
  )
  val mNode = BSMasterNode[O](
    BSMasterParameters[O](
      layers.last.output,
      layers.last.output.getType[O],
      Some(4)
    )
  )

  val module = new LazyModuleImp(this) {
    val (in, inEdge) = sNode.in.head
    val (out, outEdge) = mNode.out.head
    val numInTrans = layers.head.input.numBSTransactions(inEdge.numBeats)
    val numOutTrans = layers.last.output.numBSTransactions(outEdge.numBeats)
    val peList = layers.map(l => Module(new ProcessingElementDenseSimple(l)(LayerGenerator.layerToQC(l))))
    val inputBuffer = RegInit(VecInit.fill(numInTrans, inEdge.numBeats)(inEdge.genT.cloneType))
    val outputBuffer = RegInit(VecInit.fill(numOutTrans, outEdge.numBeats)(outEdge.genT.cloneType))
    val (inputCounterValue, inputCounterWrap) = Counter(in.fire, numInTrans)
    val (outputCounterValue, outputCounterWrap) = Counter(out.fire, numOutTrans)
    val stall = Wire(Bool())

    // Store input transactions to a buffer and connect the buffer to the input of the first dense layer
    when(in.fire) {
      inputBuffer(inputCounterValue) <> in.bits
      assert(inputCounterWrap == in.last)
    }
    peList.head.in.bits <> inputBuffer
    peList.head.in.valid := RegNext(in.last)
    in.ready := !stall

    //connect all the dense layers together
    for (i <- 1 until peList.length) {
      peList(i).in <> peList(i - 1).out
    }
    val regs = ShiftRegisters(peList.last.out, numPipes, !stall)
    stall := regs.map(_.valid).reduce(_ && _) && outputBufferHasValue && !outputCounterWrap

    val outputBufferHasValue = RegInit(false.B)
    when(regs.last.valid && (!outputBufferHasValue || outputCounterWrap)) {
      outputBuffer <> regs.last.bits
      outputBufferHasValue := true.B
      assert(outputCounterValue == 0.U)
    }
    out.bits := outputBuffer(outputCounterValue)
    out.valid := outputBufferHasValue
    out.last := outputCounterWrap
    when(outputCounterWrap) {
      outputBufferHasValue := false.B
    }
  }
}

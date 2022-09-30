/*
 * Contains refrences to the annotated model
 *
 */

package chisel4ml

import _root_.chisel3._
import _root_.chisel3.util._
import _root_.lbir.QTensor
import _root_.chiseltest.simulator.SimulatorContext

case class Circuit (
    val model: firrtl.AnnotationSeq,
    val output: QTensor,
    val tester: SimulatorContext,
    val isSimple: Boolean
)

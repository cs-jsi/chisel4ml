/*
 * Contains refrences to the annotated model
 *
 */

package chisel4ml

import _root_.chisel3._
import _root_.lbir.QTensor
import _root_.chisel4ml.Simulation

case class Circuit (
    val model: firrtl.AnnotationSeq,
    val tester: Simulation,
)

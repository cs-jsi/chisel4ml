/*
 * HEADER: TODO
 *
 *
 * This file contains the definition of the abstract class ProcessingElement.
 */

package chisel4ml

import chisel3._
import chisel3.util._
import chisel3.experimental._
import _root_.lbir.{Activation, Datatype, Layer}
import _root_.chisel4ml.util._

import _root_.org.slf4j.Logger
import _root_.org.slf4j.LoggerFactory

object Neuron {
    def apply[I <: Bits, W <: Bits, M <: Bits, A <: Bits, O <: Bits](in: Seq[I], 
                                                                     weights: Seq[W],
                                                                     thresh: A,
                                                                     mul: (I, W) => M, 
                                                                     add: Vec[M] => A,
                                                                     actFn: (A, A) => O): O = {
        val muls = VecInit((in zip weights).map{case (a,b) => mul(a,b)})
        val act = add(muls)
        actFn(act, thresh)
    }
}


abstract class ProcessingElementSimple(layer: Layer) extends Module {
    val io = IO(new Bundle {
        val in  = Input(UInt(LbirUtil.qtensorTotalBitwidth(layer.input.get).W))
        val out = Output(UInt(LbirUtil.qtensorTotalBitwidth(layer.output.get).W))
    })
}

object ProcessingElementSimple {
    def signFn(act: UInt, thresh: UInt): Bool = act >= thresh
    def signFn(act: SInt, thresh: SInt): Bool = act >= thresh
    def mul(i: Bool, w: Bool): Bool = ~(i ^ w)
    def mul(i: UInt, w: Bool): SInt = Mux(w, i.zext, -(i.zext)) 

    def apply(layer: Layer) = layer.input.get.dtype.get.quantization match {
        case Datatype.QuantizationType.UNIFORM => new ProcessingElementSimpleDense[UInt, Bool, SInt, SInt, Bool](layer,     
                                                                        UInt(layer.input.get.dtype.get.bitwidth.W),
                                                                        Bool(),
                                                                        mul,
                                                                        (x: Vec[SInt]) => x.reduceTree(_ +& _),
                                                                        signFn
                                                                        )
        case Datatype.QuantizationType.BINARY  => new ProcessingElementSimpleDense[Bool, Bool, Bool, UInt, Bool](layer,
                                                                                                          Bool(),
                                                                                                          Bool(),
                                                                                                           mul,
                                                                                   (x: Vec[Bool]) => PopCount(x),
                                                                                                          signFn)
    }
}

class ProcessingElementSimpleDense[I <: Bits, 
                                   W <: Bits : WeightsProvider, 
                                   M <: Bits, 
                                   A <: Bits : ThreshProvider, 
                                   O <: Bits](layer: Layer, 
                                              genI: I, 
                                              genO: O,
                                              mul: (I,W) => M,
                                              add: Vec[M] => A,
                                              actFn: (A, A) => O) 

extends ProcessingElementSimple(layer) {
    val weights: Seq[Seq[W]] = LbirUtil.transformWeights[W](layer.weights.get)
    val thresh: Seq[A] = LbirUtil.transformThresh[A](layer.biases.get) // A ali kaj drugo?

    val in_int  = Wire(Vec(layer.input.get.shape(0), genI))
    val out_int = Wire(Vec(layer.output.get.shape(0), genO))

    for (i <- 0 until layer.output.get.shape(0)) { 
        out_int(i) := Neuron[I, W, M, A, O](in_int, weights(i), thresh(i), mul, add, actFn) 
    }
}

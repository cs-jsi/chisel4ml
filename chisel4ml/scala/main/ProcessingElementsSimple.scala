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
import _root_.lbir.Datatype.QuantizationType._
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
        val pAct = add(muls)
        actFn(pAct, thresh)
    }
}


abstract class ProcessingElementSimple(layer: Layer) extends Module {
    val logger = LoggerFactory.getLogger(classOf[ProcessingElementSimple])
    val io = IO(new Bundle {
        val in  = Input(UInt(LbirUtil.qtensorTotalBitwidth(layer.input.get).W))
        val out = Output(UInt(LbirUtil.qtensorTotalBitwidth(layer.output.get).W))
    })
}

object ProcessingElementSimple {
    def signFn(act: UInt, thresh: UInt): Bool = act >= thresh
    def signFn(act: SInt, thresh: SInt): Bool = act >= thresh
    def reluFn(act: SInt, thresh: SInt): UInt = Mux((act - thresh) > 0.S, (act-thresh).asUInt, 0.U)

    def mul(i: Bool, w: Bool): Bool = ~(i ^ w)
    def mul(i: UInt, w: Bool): SInt = Mux(w, i.zext, -(i.zext)) 
    def mul(i: UInt, w: SInt): SInt = i * w

    def apply(layer: Layer) = (layer.input.get.dtype.get.quantization,
                               layer.weights.get.dtype.get.quantization) match {
        case (UNIFORM, UNIFORM) => new ProcessingElementSimpleDense[UInt, SInt, SInt, SInt, UInt](layer,
                                                                        UInt(layer.input.get.dtype.get.bitwidth.W),
                                                                        UInt(layer.output.get.dtype.get.bitwidth.W),
                                                                        mul,
                                                                        (x: Vec[SInt]) => x.reduceTree(_ +& _),
                                                                        layer.weights.get.dtype.get.scale,
                                                                        reluFn
                                                                        )
        case (UNIFORM, BINARY) => new ProcessingElementSimpleDense[UInt, Bool, SInt, SInt, Bool](layer,     
                                                                        UInt(layer.input.get.dtype.get.bitwidth.W),
                                                                        Bool(),
                                                                        mul,
                                                                        (x: Vec[SInt]) => x.reduceTree(_ +& _),
                                                                        layer.weights.get.dtype.get.scale,
                                                                        signFn
                                                                        )
        case (BINARY, BINARY)  => new ProcessingElementSimpleDense[Bool, Bool, Bool, UInt, Bool](layer,
                                                                                                 Bool(),
                                                                                                 Bool(),
                                                                                                    mul,
                                                                          (x: Vec[Bool]) => PopCount(x),
                                                                      layer.weights.get.dtype.get.scale,
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
                                              scales: Seq[Int],
                                              actFn: (A, A) => O) 

extends ProcessingElementSimple(layer) {
    val weights: Seq[Seq[W]] = LbirUtil.transformWeights[W](layer.weights.get)
    val thresh: Seq[A] = LbirUtil.transformThresh[A](layer.biases.get, layer.input.get.shape(0)) // A ali kaj drugo?

    val in_int  = Wire(Vec(layer.input.get.shape(0), genI))
    val out_int = Wire(Vec(layer.output.get.shape(0), genO))
    val out_scaled = Wire(Vec(layer.output.get.shape(0), genO))

    in_int := io.in.asTypeOf(in_int)
    for (i <- 0 until layer.output.get.shape(0)) { 
        out_int(i) := Neuron[I, W, M, A, O](in_int, weights(i), thresh(i), mul, add, actFn) 
    }

    if (scales.length == out_scaled.length) {
        out_scaled := (out_int zip scales).map{ case (a, s) => (a >> LbirUtil.log2(s)).asTypeOf(genO) }
    } else if (scales.length == 1) {
        out_scaled := out_int.map{ case(a) => (a >> LbirUtil.log2(scales(0))).asTypeOf(genO) }
    } else {
        out_scaled := out_int // error? TODO
        logger.error(s"ProcessingElementSimple has no scaling factors. Something is'nt right here.")
    }

    // The CAT operator reverses the order of bits, so we reverse them
    // to evenout the reversing (its not pretty but it works).
    io.out := Cat(out_scaled.reverse)

    logger.info(s"""Created new ProcessingElementSimpleDense processing element. It has an input shape: 
                    | ${layer.input.get.shape} and output shape: ${layer.output.get.shape}. The input bitwidth
                    | is ${layer.input.get.dtype.get.bitwidth}, the output bitwidth 
                    | ${layer.output.get.dtype.get.bitwidth}. Thus the total size of the input vector is 
                    | ${LbirUtil.qtensorTotalBitwidth(layer.input.get)} bits, and the total size of the output vector 
                    | is ${LbirUtil.qtensorTotalBitwidth(layer.output.get)} bits.""".stripMargin.replaceAll("\n", ""))
}

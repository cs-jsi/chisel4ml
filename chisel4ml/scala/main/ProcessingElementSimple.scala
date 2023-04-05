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
import chisel3.experimental._
import _root_.lbir.{Datatype, Layer}
import _root_.lbir.Datatype.QuantizationType._
import _root_.lbir.Layer.Activation._
import _root_.services.GenerateCircuitParams.Options
import _root_.chisel4ml.util._
import _root_.chisel4ml.implicits._

import scala.math.pow
import _root_.org.slf4j.Logger
import _root_.org.slf4j.LoggerFactory

object Neuron {
    val logger = LoggerFactory.getLogger(classOf[ProcessingElementSimple])
    def apply[I <: Bits,
              W <: Bits : WeightsProvider,
              M <: Bits,
              A <: Bits : ThreshProvider,
              O <: Bits](in: Seq[I],
                         weights: Seq[W],
                         thresh: A,
                         mul: (I, W) => M,
                         add: Vec[M] => A,
                         actFn: (A, A) => O,
                         shift: Int): O = {
        val muls = VecInit((in zip weights).map{case (a,b) => mul(a,b)})
        val pAct = add(muls)

        val sAct = shift compare 0 match {
            case 0 => pAct
            case -1 => {
                        // Handles the case when the scale factor (shift) basically sets the output to zero always.
                        if (-shift >= pAct.getWidth) {
                            0.U.asTypeOf(pAct)
                        }
                        else {
                            // We add the "cutt-off" bit to round the same way a convential rounding is done (1 >= 0.5, 0 < 0.5)
                            ((pAct >> shift.abs).asSInt + Cat(0.S((pAct.getWidth-1).W), pAct(shift.abs-1)).asSInt).asTypeOf(pAct)
                        }
            }
            case 1 => (pAct << shift.abs).asTypeOf(pAct)
            }

        actFn(sAct, thresh)
    }
}


abstract class ProcessingElementSimple(layer: Layer) extends Module {
    val io = IO(new Bundle {
        val in  = Input(UInt(layer.input.get.totalBitwidth.W))
        val out = Output(UInt(layer.output.get.totalBitwidth.W))
    })
}

object ProcessingElementSimple {
    val logger = LoggerFactory.getLogger(classOf[ProcessingElementSimple])
    def signFn(act: UInt, thresh: UInt): Bool = act >= thresh
    def signFn(act: SInt, thresh: SInt): Bool = act >= thresh
    def reluFn(act: SInt, thresh: SInt): UInt = Mux((act - thresh) > 0.S, (act - thresh).asUInt, 0.U)
    def linFn(act: SInt, thresh: SInt): SInt = act - thresh
    def noSaturate(x: Bool, bitwidth: Int): Bool = x
    def noSaturate(x: SInt, bitwidth: Int): SInt = Mux(x > (pow(2, bitwidth-1)-1).toInt.S, (pow(2, bitwidth-1)-1).toInt.S,
                                                       Mux(x < -pow(2, bitwidth-1).toInt.S, -pow(2, bitwidth-1).toInt.S, x))

    def saturate(x: UInt, bitwidth: Int): UInt = Mux(x > (pow(2, bitwidth)-1).toInt.U, (pow(2, bitwidth)-1).toInt.U, x) // TODO

    def mul(i: Bool, w: Bool): Bool = ~(i ^ w)
    def mul(i: UInt, w: Bool): SInt = Mux(w, i.zext, -(i.zext))
    def mul(i: UInt, w: SInt): SInt = {
        if (w.litValue == 1.S.litValue) {
            i.zext
        } else if(w.litValue == -1.S.litValue) {
            -(i.zext)
        } else if(w.litValue == 0.S.litValue) {
            0.S
        } else {
            i * w
        }
    }
    def mul(i: SInt, w: SInt): SInt = {
        if(w.litValue == 0.S.litValue) {
            0.S
        } else {
            i * w
        }
    }

    def apply(layer: Layer) = (layer.input.get.dtype.get.quantization,
                               layer.input.get.dtype.get.signed,
                               layer.weights.get.dtype.get.quantization,
                               layer.activation) match {
        case (UNIFORM, true,UNIFORM, RELU) => new ProcessingElementSimpleDense[SInt, SInt, SInt, SInt, UInt](layer,
                                                                        SInt(layer.input.get.dtype.get.bitwidth.W),
                                                                        UInt(layer.output.get.dtype.get.bitwidth.W),
                                                                        mul,
                                                                        (x: Vec[SInt]) => x.reduceTree(_ +& _),
                                                                        layer.weights.get.dtype.get.shift,
                                                                        reluFn,
                                                                        saturate
                                                                        )
        case (UNIFORM, false,UNIFORM, RELU) => new ProcessingElementSimpleDense[UInt, SInt, SInt, SInt, UInt](layer,
                                                                        UInt(layer.input.get.dtype.get.bitwidth.W),
                                                                        UInt(layer.output.get.dtype.get.bitwidth.W),
                                                                        mul,
                                                                        (x: Vec[SInt]) => x.reduceTree(_ +& _),
                                                                        layer.weights.get.dtype.get.shift,
                                                                        reluFn,
                                                                        saturate
                                                                        )
        case (UNIFORM, false, UNIFORM, NO_ACTIVATION) => new ProcessingElementSimpleDense[UInt, SInt, SInt, SInt, SInt](layer,
                                                                        UInt(layer.input.get.dtype.get.bitwidth.W),
                                                                        SInt(layer.output.get.dtype.get.bitwidth.W),
                                                                        mul,
                                                                        (x: Vec[SInt]) => x.reduceTree(_ +& _),
                                                                        layer.weights.get.dtype.get.shift,
                                                                        linFn,
                                                                        noSaturate
                                                                        )
        case (UNIFORM, false, BINARY, BINARY_SIGN) => new ProcessingElementSimpleDense[UInt, Bool, SInt, SInt, Bool](layer,
                                                                        UInt(layer.input.get.dtype.get.bitwidth.W),
                                                                        Bool(),
                                                                        mul,
                                                                        (x: Vec[SInt]) => x.reduceTree(_ +& _),
                                                                        layer.weights.get.dtype.get.shift,
                                                                        signFn,
                                                                        noSaturate
                                                                        )
        case (BINARY, false, BINARY, BINARY_SIGN)  => new ProcessingElementSimpleDense[Bool, Bool, Bool, UInt, Bool](layer,
                                                                                                 Bool(),
                                                                                                 Bool(),
                                                                                                    mul,
                                                                          (x: Vec[Bool]) => PopCount(x),
                                                                      layer.weights.get.dtype.get.shift,
                                                                                                 signFn,
                                                                                                 noSaturate)
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
                                              shifts: Seq[Int],
                                              actFn: (A, A) => O,
                                              saturateFn: (O, Int) => O)

extends ProcessingElementSimple(layer) {
    import ProcessingElementSimple.logger
    val weights: Seq[Seq[W]] = LbirUtil.transformWeights[W](layer.weights.get)
    val thresh: Seq[A] = LbirUtil.transformThresh[A](layer.thresh.get, layer.input.get.shape(0))
    val shift: Seq[Int] = layer.weights.get.dtype.get.shift

    val in_int  = Wire(Vec(layer.input.get.shape(0), genI))
    val out_int = Wire(Vec(layer.output.get.shape(0), genO))

    in_int := io.in.asTypeOf(in_int)
    for (i <- 0 until layer.output.get.shape(0)) {
        //
        out_int(i) := saturateFn(Neuron[I, W, M, A, O](in_int, weights(i), thresh(i), mul, add, actFn, shift(i)),
                                 layer.output.get.dtype.get.bitwidth
                                 )
    }


    // The CAT operator reverses the order of bits, so we reverse them
    // to evenout the reversing (its not pretty but it works).
    io.out := Cat(out_int.reverse)

    logger.info(s"""Created new ProcessingElementSimpleDense processing element. It has an input shape:
                    | ${layer.input.get.shape} and output shape: ${layer.output.get.shape}. The input bitwidth
                    | is ${layer.input.get.dtype.get.bitwidth}, the output bitwidth
                    | ${layer.output.get.dtype.get.bitwidth}. Thus the total size of the input vector is
                    | ${layer.input.get.totalBitwidth} bits, and the total size of the output vector
                    | is ${layer.output.get.totalBitwidth} bits.""".stripMargin.replaceAll("\n", ""))
}

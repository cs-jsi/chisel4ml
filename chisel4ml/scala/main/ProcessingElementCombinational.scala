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
package chisel4ml.combinational

import _root_.chisel4ml.implicits._
import _root_.chisel4ml.lbir._
import _root_.chisel4ml.util._
import _root_.lbir.Datatype.QuantizationType._
import _root_.lbir.Layer
import _root_.lbir.Layer.Activation._
import _root_.org.slf4j.LoggerFactory
import chisel3._
import chisel3.util._

abstract class ProcessingElementCombinational(layer: Layer) extends Module {
  val io = IO(new Bundle {
    val in  = Input(UInt(layer.input.get.totalBitwidth.W))
    val out = Output(UInt(layer.output.get.totalBitwidth.W))
  })
}

object ProcessingElementCombinational {
  def apply(layer: Layer) =
    (layer.input.get.dtype.get.quantization, layer.weights.get.dtype.get.quantization, layer.activation) match {
      case (UNIFORM, UNIFORM, RELU) =>
        new ProcessingElementCombinationalDense[UInt, SInt, SInt, SInt, UInt](
          layer,
          UInt(layer.input.get.dtype.get.bitwidth.W),
          UInt(layer.output.get.dtype.get.bitwidth.W),
          mul,
          (x: Vec[SInt]) => x.reduceTree(_ +& _),
          layer.weights.get.dtype.get.shift,
          reluFn,
          saturate,
        )
      case (UNIFORM, UNIFORM, NO_ACTIVATION) =>
        new ProcessingElementCombinationalDense[UInt, SInt, SInt, SInt, SInt](
          layer,
          UInt(layer.input.get.dtype.get.bitwidth.W),
          SInt(layer.output.get.dtype.get.bitwidth.W),
          mul,
          (x: Vec[SInt]) => x.reduceTree(_ +& _),
          layer.weights.get.dtype.get.shift,
          linFn,
          noSaturate,
        )
      case (UNIFORM, BINARY, BINARY_SIGN) =>
        new ProcessingElementCombinationalDense[UInt, Bool, SInt, SInt, Bool](
          layer,
          UInt(layer.input.get.dtype.get.bitwidth.W),
          Bool(),
          mul,
          (x: Vec[SInt]) => x.reduceTree(_ +& _),
          layer.weights.get.dtype.get.shift,
          signFn,
          noSaturate,
        )
      case (BINARY, BINARY, BINARY_SIGN) =>
        new ProcessingElementCombinationalDense[Bool, Bool, Bool, UInt, Bool](
          layer,
          Bool(),
          Bool(),
          mul,
          (x: Vec[Bool]) => PopCount(x),
          layer.weights.get.dtype.get.shift,
          signFn,
          noSaturate,
        )
    }
}

class ProcessingElementCombinationalDense[
    I <: Bits,
    W <: Bits: WeightsProvider,
    M <: Bits,
    A <: Bits: ThreshProvider,
    O <: Bits,
  ](
    layer:      Layer,
    genI:       I,
    genO:       O,
    mul:        (I, W) => M,
    add:        Vec[M] => A,
    shifts:     Seq[Int],
    actFn:      (A, A) => O,
    saturateFn: (O, Int) => O,
  ) extends ProcessingElementCombinational(layer) {
  val logger = LoggerFactory.getLogger(classOf[ProcessingElementCombinational])
  val weights: Seq[Seq[W]] = LbirDataTransforms.transformWeights[W](layer.weights.get)
  val thresh:  Seq[A]      = LbirDataTransforms.transformThresh[A](layer.thresh.get, layer.input.get.shape(0))
  val shift:   Seq[Int]    = layer.weights.get.dtype.get.shift

  val in_int  = Wire(Vec(layer.input.get.shape(0), genI))
  val out_int = Wire(Vec(layer.output.get.shape(0), genO))

  in_int := io.in.asTypeOf(in_int)
  for (i <- 0 until layer.output.get.shape(0)) {
    out_int(i) := saturateFn(
      StaticNeuron[I, W, M, A, O](in_int, weights(i), thresh(i), mul, add, actFn, shift(i)),
      layer.output.get.dtype.get.bitwidth,
    )
  }

  // The CAT operator reverses the order of bits, so we reverse them
  // to evenout the reversing (its not pretty but it works).
  io.out := Cat(out_int.reverse)

  logger.info(s"""Created new ProcessingElementCombinationalDense processing element. It has an input shape:
                 | ${layer.input.get.shape} and output shape: ${layer.output.get.shape}. The input bitwidth
                 | is ${layer.input.get.dtype.get.bitwidth}, the output bitwidth
                 | ${layer.output.get.dtype.get.bitwidth}. Thus the total size of the input vector is
                 | ${layer.input.get.totalBitwidth} bits, and the total size of the output vector
                 | is ${layer.output.get.totalBitwidth} bits.""".stripMargin.replaceAll("\n", ""))
}

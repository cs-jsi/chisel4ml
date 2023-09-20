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

import _root_.chisel4ml.combinational.StaticNeuron
import _root_.chisel4ml.implicits._
import _root_.chisel4ml.util._
import _root_.lbir.Activation.{BINARY_SIGN, NO_ACTIVATION, RELU}
import _root_.lbir.Datatype.QuantizationType._
import _root_.lbir.DenseConfig
import _root_.org.slf4j.LoggerFactory
import chisel3._
import chisel3.util._

abstract class ProcessingElementSimple(layer: DenseConfig) extends Module {
  val io = IO(new Bundle {
    val in = Input(UInt(layer.input.totalBitwidth.W))
    val out = Output(UInt(layer.output.totalBitwidth.W))
  })
}

object ProcessingElementSimple {
  val logger = LoggerFactory.getLogger(classOf[ProcessingElementSimple])
  def apply(layer: DenseConfig) = (
    layer.input.dtype.quantization,
    layer.input.dtype.signed,
    layer.weights.dtype.quantization,
    layer.activation
  ) match {
    case (UNIFORM, true, UNIFORM, RELU) =>
      new ProcessingElementSimpleDense[SInt, SInt, SInt, SInt, UInt](
        layer,
        SInt(layer.input.dtype.bitwidth.W),
        UInt(layer.output.dtype.bitwidth.W),
        mul,
        (x: Vec[SInt]) => x.reduceTree(_ +& _),
        reluFn,
        saturate
      )
    case (UNIFORM, false, UNIFORM, RELU) =>
      new ProcessingElementSimpleDense[UInt, SInt, SInt, SInt, UInt](
        layer,
        UInt(layer.input.dtype.bitwidth.W),
        UInt(layer.output.dtype.bitwidth.W),
        mul,
        (x: Vec[SInt]) => x.reduceTree(_ +& _),
        reluFn,
        saturate
      )
    case (UNIFORM, false, UNIFORM, NO_ACTIVATION) =>
      new ProcessingElementSimpleDense[UInt, SInt, SInt, SInt, SInt](
        layer,
        UInt(layer.input.dtype.bitwidth.W),
        SInt(layer.output.dtype.bitwidth.W),
        mul,
        (x: Vec[SInt]) => x.reduceTree(_ +& _),
        linFn,
        noSaturate
      )
    case (UNIFORM, _, BINARY, BINARY_SIGN) =>
      new ProcessingElementSimpleDense[UInt, Bool, SInt, SInt, Bool](
        layer,
        UInt(layer.input.dtype.bitwidth.W),
        Bool(),
        mul,
        (x: Vec[SInt]) => x.reduceTree(_ +& _),
        signFn,
        noSaturate
      )
    case (BINARY, _, BINARY, BINARY_SIGN) =>
      new ProcessingElementSimpleDense[Bool, Bool, Bool, UInt, Bool](
        layer,
        Bool(),
        Bool(),
        mul,
        (x: Vec[Bool]) => PopCount(x),
        signFn,
        noSaturate
      )
    case _ => throw new RuntimeException()
  }
}

class ProcessingElementSimpleDense[I <: Bits, W <: Bits, M <: Bits, A <: Bits, O <: Bits](
  layer:      DenseConfig,
  genI:       I,
  genO:       O,
  mul:        (I, W) => M,
  add:        Vec[M] => A,
  actFn:      (A, A) => O,
  saturateFn: (O, Int) => O)
    extends ProcessingElementSimple(layer) {
  import ProcessingElementSimple.logger
  val weights: Seq[Seq[W]] = layer.getWeights[W]
  val thresh:  Seq[A] = layer.getThresh[A]
  val shift:   Seq[Int] = layer.weights.dtype.shift

  val in_int = Wire(Vec(layer.input.shape(0), genI))
  val out_int = Wire(Vec(layer.output.shape(0), genO))

  in_int := io.in.asTypeOf(in_int)
  for (i <- 0 until layer.output.shape(0)) {
    out_int(i) := saturateFn(
      StaticNeuron[I, W, M, A, O](in_int, weights(i), thresh(i), mul, add, actFn, shift(i)),
      layer.output.dtype.bitwidth
    )
  }

  // The CAT operator reverses the order of bits, so we reverse them
  // to evenout the reversing (its not pretty but it works).
  io.out := Cat(out_int.reverse)

  logger.info(
    s"""Created new ProcessingElementSimpleDense processing element. It has an input shape:
       | ${layer.input.shape} and output shape: ${layer.output.shape}. The input bitwidth
       | is ${layer.input.dtype.bitwidth}, the output bitwidth
       | ${layer.output.dtype.bitwidth}. Thus the total size of the input vector is
       | ${layer.input.totalBitwidth} bits, and the total size of the output vector
       | is ${layer.output.totalBitwidth} bits.
       | The input quantization is ${genI}, output quantization is ${genO}.""".stripMargin
      .replaceAll("\n", "")
  )
}

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

import chisel4ml.combinational.StaticNeuron
import chisel4ml.implicits._
import chisel4ml.util._
import lbir.Activation.{BINARY_SIGN, NO_ACTIVATION, RELU}
import lbir.Datatype.QuantizationType._
import lbir.DenseConfig
import org.slf4j.LoggerFactory
import chisel3._
import chisel3.util._
import dsptools.DspException

trait QuantizationCompute[I <: Bits, W <: Bits, M <: Bits, A <: Bits, O <: Bits] extends Any {
  def mul:   (I, W) => M
  def add:   Vec[M] => A
  def actFn: (A, A) => O
}
/*
trait BinarizedQuantizationCompute extends Any with QuantizationCompute[Bool, Bool, Bool, UInt, Bool] {
  def mul = (i: Bool, w: Bool) => ~(i ^ w)
  def add = (x: Vec[Bool]) => PopCount(x.asUInt)
}*/

class BinarizedQuantizationCompute(act: (UInt, UInt) => Bool)
    extends QuantizationCompute[Bool, Bool, Bool, UInt, Bool] {
  def mul = (i: Bool, w: Bool) => ~(i ^ w)
  def add = (x: Vec[Bool]) => PopCount(x.asUInt)
  def actFn: (UInt, UInt) => Bool = act
}

// implementiraj s dsptools?
//trait UniformQuantizationCompute[I <: Bits, W <: Bits,] extends Any with QuantizationCompute[]

object Neuron {
  def apply[I <: Bits, W <: Bits, M <: Bits, A <: Bits, O <: Bits](
    in:      Seq[I],
    weights: Seq[W],
    thresh:  A,
    shift:   Int
  )(qc:      QuantizationCompute[I, W, M, A, O]
  ): O = {
    val muls = VecInit((in.zip(weights)).map { case (a, b) => qc.mul(a, b) })
    val pAct = qc.add(muls)
    val sAct = shiftAndRound(pAct, shift)
    qc.actFn(sAct, thresh)
  }
}

object ProcessingElementSimple {
  def apply(layer: DenseConfig) = (
    layer.input.dtype.quantization,
    layer.input.dtype.signed,
    layer.weights.dtype.quantization
  ) match {
    //case (UNIFORM, true, UNIFORM) => new ProcessingElementSimple[SInt, SInt, SInt, SInt, UInt](new DenseConfigTyped(layer))
    //case (UNIFORM, false, UNIFORM) => new ProcessingElementSimple[UInt, SInt, SInt, SInt, UInt](new DenseConfigTyped(layer))
    //case (UNIFORM, _, BINARY) => new ProcessingElementSimple[UInt, Bool, SInt, SInt, Bool](new DenseConfigTyped(layer))
    case (BINARY, _, BINARY) =>
      new ProcessingElementSimple(new DenseConfigTyped[Bool, Bool, Bool, UInt, Bool](layer))(
        new BinarizedQuantizationCompute(signFn)
      )
    case _ => throw new RuntimeException()
  }
}

class DenseConfigTyped[I <: Bits, W <: Bits, M <: Bits, A <: Bits, O <: Bits](layer: DenseConfig) {
  val p = layer
  val genI = (layer.input.dtype.quantization, layer.input.dtype.signed) match {
    case (UNIFORM, true)  => SInt(layer.input.dtype.bitwidth.W).asInstanceOf[I]
    case (UNIFORM, false) => UInt(layer.input.dtype.bitwidth.W).asInstanceOf[I]
    case (BINARY, _)      => Bool().asInstanceOf[I]
    case _                => throw new DspException("Unknown quantization.")
  }
  require(layer.weights.dtype.signed == true)
  val genW = (layer.weights.dtype.quantization) match {
    case UNIFORM => SInt(layer.weights.dtype.bitwidth.W).asInstanceOf[W]
    case BINARY  => Bool().asInstanceOf[W]
    case _       => throw new DspException("Weight type not supported")
  }

  val genO = (layer.output.dtype.quantization, layer.output.dtype.signed) match {
    case (UNIFORM, true)  => SInt(layer.output.dtype.bitwidth.W).asInstanceOf[O]
    case (UNIFORM, false) => UInt(layer.output.dtype.bitwidth.W).asInstanceOf[O]
    case (BINARY, _)      => Bool().asInstanceOf[O]
    case _                => throw new DspException("Output type not supported.")
  }

  def mul: (I, W) => M = (layer.input.dtype.quantization, layer.weights.dtype.quantization) match {
    case (BINARY, BINARY) => (i: I, w: W) => (~(i.asBool ^ w.asBool)).asInstanceOf[M]
    case (UNIFORM, BINARY) =>
      (i: I, w: W) => Mux(w.asBool, i.asUInt.zext.asInstanceOf[M], (-i.asUInt.zext).asInstanceOf[M])
    case (UNIFORM, UNIFORM) => (i: I, w: W) => (i.asUInt * w.asUInt).asInstanceOf[M]
    case _                  => throw new DspException("Unsuported multiplicaton operaiton.")
  }

  def add: Vec[M] => A = (layer.input.dtype.quantization, layer.weights.dtype.quantization) match {
    case (BINARY, BINARY) => (x: Vec[M]) => PopCount(x.asUInt).asInstanceOf[A]
    case _                => (x: Vec[M]) => VecInit(x.map(_.asSInt)).reduceTree(_ +& _).asInstanceOf[A]
  }

  def actFn: (A, A) => O = layer.activation match {
    case BINARY_SIGN => (act: A, thresh: A) => (act.asSInt >= thresh.asSInt).asInstanceOf[O]
    case RELU =>
      (act: A, thresh: A) =>
        saturate(reluFn(act.asSInt, thresh.asSInt), layer.output.dtype.bitwidth, layer.output.dtype.signed)
          .asInstanceOf[O]
    case NO_ACTIVATION => (act: A, thresh: A) => act.asInstanceOf[O]
    case _             => throw new DspException("Unsopported activation")
  }

}

class ProcessingElementSimple[I <: Bits, W <: Bits, M <: Bits, A <: Bits, O <: Bits](
  layer: DenseConfigTyped[I, W, M, A, O]
)(qc:    QuantizationCompute[I, W, M, A, O])
    extends Module
    with LBIRStreamSimple {
  val logger = LoggerFactory.getLogger("ProcessingElementSimple")
  val in = IO(Input(Vec(layer.p.input.width, layer.genI)))
  val out = IO(Output(Vec(layer.p.output.width, layer.genO)))
  val weights: Seq[Seq[W]] = layer.p.getWeights[W]
  val thresh:  Seq[A] = layer.p.getThresh[A]
  val shift:   Seq[Int] = layer.p.weights.dtype.shift

  for (i <- 0 until layer.p.output.shape(0)) {
    out(i) := Neuron[I, W, M, A, O](in.map(_.asInstanceOf[I]), weights(i), thresh(i), shift(i))(qc)
  }

  logger.info(
    s"""Created new ProcessingElementSimpleDense processing element. It has an input shape:
       | ${layer.p.input.shape} and output shape: ${layer.p.output.shape}. The input bitwidth
       | is ${layer.p.input.dtype.bitwidth}, the output bitwidth
       | ${layer.p.output.dtype.bitwidth}. Thus the total size of the input vector is
       | ${layer.p.input.totalBitwidth} bits, and the total size of the output vector
       | is ${layer.p.output.totalBitwidth} bits.
       | The input quantization is ${layer.genI}, output quantization is ${layer.genO}.""".stripMargin
      .replaceAll("\n", "")
  )
}

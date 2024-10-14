package chisel4ml

import chisel3._
import chisel3.experimental.VecLiterals._
import chisel4ml.implicits._
import lbir.Datatype.QuantizationType.{BINARY, UNIFORM}
import lbir.{DenseConfig, HasInputOutputQTensor, IsActiveLayer, LayerWrap}

object LayerMapping {
  def layerToInputMap(layer: LayerWrap): Seq[Seq[Int]] = layer match {
    case l: DenseConfig => Seq.fill(l.output.width)((0 until l.input.width).toSeq)
    case _ => throw new RuntimeException
  }

  def layerToKernelMap(layer: LayerWrap): Seq[Seq[Int]] = layer match {
    case l: DenseConfig => (0 until l.kernel.numParams).toSeq.grouped(l.input.width).toSeq
    case _ => throw new RuntimeException
  }

  def getReceptiveField[T <: Data](input: Seq[T], receptiveField: Seq[Int]): Seq[T] = {
    Seq(receptiveField.map(input(_)): _*)
  }

  def getReceptiveField[T <: Data](input: Vec[T], receptiveField: Seq[Int]): Vec[T] = {
    Vec.Lit(receptiveField.map(input(_)): _*)
  }

  def getKernel[T <: Data](layer: LayerWrap with HasInputOutputQTensor with IsActiveLayer): Seq[T] =
    layer.kernel.dtype.quantization match {
      case UNIFORM =>
        layer.kernel.values.map(_.toInt.S.asInstanceOf[T]).grouped(layer.kernel.width).toSeq.transpose.flatten
      case BINARY =>
        layer.kernel.values.map(_ > 0).map(_.B.asInstanceOf[T]).grouped(layer.kernel.width).toSeq.transpose.flatten
      case _ => throw new RuntimeException
    }

  def getThresh[T <: Data](layer: LayerWrap with HasInputOutputQTensor with IsActiveLayer): Seq[T] =
    (layer.input.dtype.quantization, layer.kernel.dtype.quantization, layer.activation) match {
      case (BINARY, BINARY, lbir.Activation.BINARY_SIGN) =>
        layer.thresh.values.map(x => (layer.input.shape(0) + x) / 2).map(_.ceil).map(_.toInt.U).map(_.asInstanceOf[T])
      case _ => layer.thresh.values.map(_.toInt.S(layer.thresh.dtype.bitwidth.W)).map(_.asInstanceOf[T])
    }
}

package chisel4ml

import chisel3._
import chisel3.experimental.VecLiterals._
import chisel4ml.implicits._
import lbir.Datatype.QuantizationType.{BINARY, UNIFORM}
import lbir.{Conv2DConfig, DenseConfig, HasInputOutputQTensor, IsActiveLayer, LayerWrap, MaxPool2DConfig, QTensor}

import scala.collection.mutable.ArraySeq

object LayerMapping {
  def checkParamsSlidingWindow(
    tensor:     QTensor,
    kernelSize: Seq[Int],
    stride:     Seq[Int],
    padding:    Seq[Int],
    dilation:   Seq[Int],
    groups:     Int
  ): Unit = {
    require(kernelSize.length == 2)
    require(kernelSize(0) > 0 && kernelSize(1) > 0)

    require(stride.length == 2)
    require(stride(0) > 0 && stride(1) > 0)

    require(padding.length == 2)
    require(padding(0) >= 0 && padding(1) >= 0)

    require(groups == tensor.numChannels || groups == 1, "Only depthwise or normal conv supported.")
    require(dilation.length == 0, "Dillation not yet supported")
  }
  def inBoundary(h: Int, w: Int, padding: Seq[Int], tensor: QTensor): Boolean = {
    h >= padding(0) &&
    h < (padding(0) + tensor.height) &&
    w >= padding(1) &&
    w < (padding(1) + tensor.width)
  }
  def slidingWindowMap(
    tensor:     QTensor,
    kernelSize: Seq[Int],
    stride:     Seq[Int],
    padding:    Seq[Int],
    dilation:   Seq[Int],
    groups:     Int
  ): Seq[Seq[Int]] = {
    checkParamsSlidingWindow(tensor, kernelSize, stride, padding, dilation, groups)
    // NCHW layout
    /*  (padding = (1,1))    -1 -1 -1 -1 -1
      A B C                  -1  0  1  2 -1
      D E F         =>       -1  3  4  5 -1
      G I H                  -1  6  7  8 -1
                             -1 -1 -1 -1 -1
     */
    val paddedInputHeight: Int = tensor.height + 2 * padding(0)
    val paddedInputWidth:  Int = tensor.width + 2 * padding(1)
    // -1 signifies padded values (later gets converted to zeros)
    val inputIndecies = Seq
      .tabulate(tensor.numChannels, paddedInputHeight, paddedInputWidth)((c, h, w) => {
        if (inBoundary(h, w, padding, tensor))
          c * (tensor.width * tensor.height) + (h - padding(0)) * tensor.width + (w - padding(1))
        else -1
      })
      .flatten
      .flatten
    //val inputIndecies = (0 until tensor.numParams).toList
    val outWidth = ((tensor.width - kernelSize(1) + 2 * padding(1)) / stride(0)) + 1
    val outHeight = ((tensor.height - kernelSize(0) + 2 * padding(0)) / stride(1)) + 1
    val outChannels = tensor.numChannels / groups
    val out: ArraySeq[Seq[Int]] = ArraySeq.fill(outHeight * outWidth)(Seq())

    for {
      ch <- 0 until outChannels
      h <- 0 until outHeight
      w <- 0 until outWidth
    } {
      var map: Seq[Int] = Seq()
      val baseIndex = ch * (paddedInputWidth * paddedInputHeight) + h * paddedInputWidth + w
      for {
        kh <- 0 until kernelSize(0)
        kw <- 0 until kernelSize(1)
      } {
        map = map :+ inputIndecies(baseIndex + kh * paddedInputWidth + kw)
      }
      val outIndex = ch * (outWidth * outHeight) + h * outWidth + w
      out(outIndex) = map
    }
    out.map(_.toSeq).toSeq
  }
  def layerToInputMap(layer: LayerWrap): Seq[Seq[Int]] = layer match {
    case l: DenseConfig => Seq.fill(l.output.width)((0 until l.input.width).toSeq)
    case l: Conv2DConfig =>
      slidingWindowMap(
        l.input,
        kernelSize = Seq(l.kernel.height, l.kernel.width),
        stride = Seq(1, 1),
        padding = Seq(0, 0),
        dilation = Seq(),
        groups = { if (l.depthwise) l.input.numChannels else 1 }
      )
    case l: MaxPool2DConfig =>
      slidingWindowMap(
        l.input,
        kernelSize = Seq(l.input.shape(0) / l.output.shape(0), l.input.shape(1) / l.output.shape(1)),
        stride = Seq(1, 1),
        padding = Seq(0, 0),
        dilation = Seq(),
        groups = l.input.numChannels
      )
    case _ => throw new RuntimeException
  }

  def layerToKernelMap(layer: LayerWrap): Seq[Seq[Int]] = layer match {
    case l: DenseConfig  => (0 until l.kernel.numParams).toSeq.grouped(l.input.width).toSeq
    case l: Conv2DConfig => (0 until l.kernel.numParams).toSeq.grouped(l.input.width).toSeq
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

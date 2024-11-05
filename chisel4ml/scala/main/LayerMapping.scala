package chisel4ml

import chisel3._
import chisel3.experimental.VecLiterals._
import chisel4ml.implicits._
import lbir.Datatype.QuantizationType.{BINARY, UNIFORM}
import lbir.{Conv2DConfig, DenseConfig, HasInputOutputQTensor, IsActiveLayer, LayerWrap, MaxPool2DConfig, QTensor}

import scala.collection.mutable.ArraySeq

object LayerMapping {
  def checkParamsSlidingWindow(
    tensor:      QTensor,
    kernelSize:  Seq[Int],
    stride:      Seq[Int],
    padding:     Seq[Int],
    dilation:    Seq[Int],
    groups:      Int,
    outChannels: Int
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
    inputTensor: QTensor,
    kernelSize:  Seq[Int],
    stride:      Seq[Int],
    padding:     Seq[Int],
    dilation:    Seq[Int],
    groups:      Int,
    outChannels: Int
  ): Seq[Seq[Int]] = {
    checkParamsSlidingWindow(inputTensor, kernelSize, stride, padding, dilation, groups, outChannels)
    // NCHW layout
    /*  (padding = (1,1))    -1 -1 -1 -1 -1
      A B C                  -1  0  1  2 -1
      D E F         =>       -1  3  4  5 -1
      G I H                  -1  6  7  8 -1
                             -1 -1 -1 -1 -1
     */
    val paddedInputHeight: Int = inputTensor.height + 2 * padding(0)
    val paddedInputWidth:  Int = inputTensor.width + 2 * padding(1)
    // -1 signifies padded values (later gets converted to zeros)
    val inputIndecies = Seq
      .tabulate(inputTensor.numChannels, paddedInputHeight, paddedInputWidth)((c, h, w) => {
        if (inBoundary(h, w, padding, inputTensor))
          c * (inputTensor.width * inputTensor.height) + (h - padding(0)) * inputTensor.width + (w - padding(1))
        else -1
      })
      .flatten
      .flatten

    val outWidth = ((inputTensor.width - kernelSize(1) + 2 * padding(1)) / stride(0)) + 1
    val outHeight = ((inputTensor.height - kernelSize(0) + 2 * padding(0)) / stride(1)) + 1
    val out: ArraySeq[Seq[Int]] = ArraySeq.fill(outChannels * outHeight * outWidth)(Seq())
    for {
      och <- 0 until outChannels
      h <- 0 until outHeight
      w <- 0 until outWidth
    } {
      var map: Seq[Int] = Seq()
      for {
        ich <- 0 until (inputTensor.numChannels / groups)
        kh <- 0 until kernelSize(0)
        kw <- 0 until kernelSize(1)
      } {
        val channelsPerGroup = inputTensor.numChannels / groups
        val groupsOffset = och * channelsPerGroup * (paddedInputWidth * paddedInputHeight)
        val baseIndex = ich * (paddedInputWidth * paddedInputHeight) + h * paddedInputWidth + w
        map = map :+ inputIndecies(baseIndex + groupsOffset + kh * paddedInputWidth + kw)
      }
      val outIndex = och * (outWidth * outHeight) + h * outWidth + w
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
        groups = { if (l.depthwise) l.input.numChannels else 1 },
        outChannels = l.output.numChannels
      )
    case l: MaxPool2DConfig =>
      slidingWindowMap(
        l.input,
        kernelSize = Seq(l.input.shape(0) / l.output.shape(0), l.input.shape(1) / l.output.shape(1)),
        stride = Seq(1, 1),
        padding = Seq(0, 0),
        dilation = Seq(),
        groups = l.input.numChannels,
        outChannels = l.output.numChannels
      )
    case _ => throw new RuntimeException
  }

  def layerToKernelMap(layer: LayerWrap): Seq[Seq[Int]] = layer match {
    case l: DenseConfig => (0 until l.kernel.numParams).toSeq.grouped(l.input.width).toSeq
    case l: Conv2DConfig =>
      Seq
        .tabulate(l.output.numChannels, l.output.width * l.output.height)((c, _) => {
          c * (l.kernel.numKernelParams) until ((c + 1) * (l.kernel.numKernelParams))
        })
        .flatten
    case _ => throw new RuntimeException
  }

  def layerToThreshMap(layer: LayerWrap with IsActiveLayer): Seq[Int] = layer match {
    case l: DenseConfig => {
      require(l.thresh.numParams == l.output.numParams)
      0 until l.output.numParams
    }
    case l: Conv2DConfig => {
      l.thresh.numParams match {
        case 1 => Seq.fill(l.output.numParams)(0)
        case _ if (l.output.numChannels == l.thresh.numParams) =>
          Seq.tabulate(l.output.numChannels, l.output.width * l.output.height)((c, _) => c).flatten
        case _ => throw new RuntimeException
      }
    }
    case _ => throw new RuntimeException
  }

  def layerToShiftMap(layer: LayerWrap with IsActiveLayer): Seq[Int] = layer match {
    case l: DenseConfig => {
      require(l.thresh.numParams == l.output.numParams)
      0 until l.output.numParams
    }
    case l: Conv2DConfig => {
      l.kernel.dtype.shift.length match {
        case 1 => Seq.fill(l.output.numParams)(0)
        case x if (x == l.kernel.numKernels) =>
          Seq.tabulate(l.output.numChannels, l.output.width * l.output.height)((c, _) => c).flatten
        case x if (x == l.kernel.numChannels) =>
          throw new RuntimeException("Per-channel scaling factors not supported.")
        case _ => throw new RuntimeException(f"${l.kernel.dtype.shift.length} != ${l.kernel.numKernels}")
      }
    }
    case _ => throw new RuntimeException
  }

  def getReceptiveField[T <: Data](input: Seq[T], receptiveField: Seq[Int]): Seq[T] = {
    Seq(receptiveField.map(input(_)): _*)
  }

  def getReceptiveField[T <: Data](input: Vec[T], receptiveField: Seq[Int]): Vec[T] = {
    Vec.Lit(receptiveField.map(input(_)): _*)
  }

  def getKernel[T <: Data](layer: LayerWrap with HasInputOutputQTensor with IsActiveLayer): Seq[T] =
    layer match {
      case l: DenseConfig => {
        layer.kernel.dtype.quantization match {
          case UNIFORM =>
            l.kernel.values.map(_.toInt.S.asInstanceOf[T]).grouped(l.kernel.width).toSeq.transpose.flatten
          case BINARY =>
            l.kernel.values.map(_ > 0).map(_.B.asInstanceOf[T]).grouped(l.kernel.width).toSeq.transpose.flatten
          case _ => throw new RuntimeException
        }
      }
      case l: Conv2DConfig => {
        l.kernel.dtype.quantization match {
          case UNIFORM =>
            l.kernel.values.map(_.toInt.S.asInstanceOf[T]).grouped(l.kernel.width).toSeq.flatten
          case BINARY =>
            l.kernel.values.map(_ > 0).map(_.B.asInstanceOf[T]).grouped(l.kernel.width).toSeq.flatten
          case _ => throw new RuntimeException
        }
      }
      case _ => throw new RuntimeException
    }

  def saturateToZero(x: Float): Float = {
    if (x > 0.0) x else 0.0f
  }

  def getThresh[T <: Data](layer: LayerWrap with HasInputOutputQTensor with IsActiveLayer): Seq[T] =
    (layer.input.dtype.quantization, layer.kernel.dtype.quantization, layer.activation) match {
      case (BINARY, BINARY, lbir.Activation.BINARY_SIGN) =>
        layer.thresh.values
          .map(x => (layer.numActiveParams + x) / 2)
          .map(_.ceil)
          .map((x: Float) => saturateToZero(x))
          .map(_.toInt.U)
          .map(_.asInstanceOf[T])
      case _ => layer.thresh.values.map(_.toInt.S(layer.thresh.dtype.bitwidth.W)).map(_.asInstanceOf[T])
    }
}

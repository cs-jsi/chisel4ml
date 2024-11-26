package chisel4ml

import chisel3._
import chisel4ml.implicits._
import lbir.Datatype.QuantizationType.{BINARY, UNIFORM}
import lbir.{Conv2DConfig, DenseConfig, HasInputOutputQTensor, IsActiveLayer, LayerWrap, MaxPool2DConfig, QTensor}

import scala.collection.mutable.ArraySeq

/** Functions related to tensor linearization in memory/vector
  */
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

    require(padding.length == 4)
    // padding = List(pad_top, pad_left, pad_bottom, pad_right)
    require(padding(0) >= 0 && padding(1) >= 0 && padding(2) >= 0 && padding(3) >= 0)

    require(groups == tensor.numChannels || groups == 1, "Only depthwise or normal conv supported.")
    require(dilation.length == 0, "Dillation not yet supported")
  }

  /** Checks that a given index is inside the active area, and not the padding
    *
    * @param h
    *   Height index
    * @param w
    *   Width index
    * @param padding
    *   A length of four integer sequence of padding information (top, left, bottom, right)
    * @param tensor
    *   The tensor for which this is being computed
    */
  def inBoundary(h: Int, w: Int, padding: Seq[Int], tensor: QTensor): Boolean = {
    h >= padding(0) &&
    h < (padding(0) + tensor.height) &&
    w >= padding(1) &&
    w < (padding(1) + tensor.width)
  }

  /** Given an input tensor and configuration it returns the sliding window map of input used by succesive output
    * neurons. This is used for two-dimensional MaxPool and Conv operations.
    *
    * @param inputTensor
    *   The input tensor for which we seek the map (e.g. conv.input)
    * @param kernelSize
    *   A length of two sequence of the kernel shape (kernelHeight, kernelWidth)
    * @param stride
    *   A length of two sequence of the stride (stride in height direction, stride in width direction)
    * @param padding
    *   A length of four integer sequence of padding information (top, left, bottom, right)
    * @param dilation
    *   NOT CURRENTLY SUPPORTED
    * @param groups
    *   Number of groups to use (see ONNX defintion for Conv op)
    * @param outChannels
    *   The number of output channels
    *
    * @return
    *   Returns a sequence the length of the number of outputs. For each output it returns a sequence of inputs that
    *   relate to this input. In other words it returns the input activation map/window.
    */
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
    // Padding assumes the same "layout" as in onnx
    val paddedInputHeight: Int = padding(0) + inputTensor.height + padding(2)
    val paddedInputWidth:  Int = padding(1) + inputTensor.width + padding(3)

    // -1 signifies padded values (later gets converted to zeros)
    val inputIndecies = Seq
      .tabulate(inputTensor.numChannels, paddedInputHeight, paddedInputWidth)((c, h, w) => {
        if (inBoundary(h, w, padding, inputTensor))
          c * (inputTensor.width * inputTensor.height) + (h - padding(0)) * inputTensor.width + (w - padding(1))
        else -1
      })
      .flatten
      .flatten

    val channelsPerGroup = inputTensor.numChannels / groups
    val kernelsPerGroups = outChannels / groups
    val outWidth = ((inputTensor.width - kernelSize(1) + padding(1) + padding(3)) / stride(1)) + 1
    val outHeight = ((inputTensor.height - kernelSize(0) + padding(0) + padding(2)) / stride(0)) + 1
    val out: ArraySeq[Seq[Int]] = ArraySeq.fill(outChannels * outHeight * outWidth)(Seq())
    for {
      och <- 0 until outChannels
      h <- 0 until outHeight
      w <- 0 until outWidth
    } {
      var map: Seq[Int] = Seq()
      for {
        ich <- 0 until channelsPerGroup
        kh <- 0 until kernelSize(0)
        kw <- 0 until kernelSize(1)
      } {
        val heightOffset = (h * stride(0)) * paddedInputWidth
        val widthOffset = w * stride(1)
        val channelOffset = ich * (paddedInputWidth * paddedInputHeight)
        val group = och / kernelsPerGroups
        val groupOffset = group * (channelsPerGroup * paddedInputWidth * paddedInputHeight)
        map =
          map :+ inputIndecies(groupOffset + channelOffset + heightOffset + widthOffset + kh * paddedInputWidth + kw)
      }
      val outIndex = och * (outWidth * outHeight) + h * outWidth + w
      out(outIndex) = map
    }
    out.map(_.toSeq).toSeq
  }

  /** Converts a LBIR LayerWrap object to an input activation map. (e.g. see use in
    * combinational/NeuronProcessingUnit.scala)
    *
    * @param layer
    *   The LBIR layer object to convert to an input map
    * @return
    *   The input activation map, i.e. for each output what inputs are relevant?
    */
  def layerToInputMap(layer: LayerWrap): Seq[Seq[Int]] = layer match {
    case l: DenseConfig => Seq.fill(l.output.width)((0 until l.input.width).toSeq)
    case l: Conv2DConfig =>
      slidingWindowMap(
        l.input,
        kernelSize = Seq(l.kernel.height, l.kernel.width),
        stride = l.stride,
        padding = l.padding,
        dilation = Seq(),
        groups = { if (l.depthwise) l.input.numChannels else 1 },
        outChannels = l.output.numChannels
      )
    case l: MaxPool2DConfig =>
      slidingWindowMap(
        l.input,
        kernelSize = l.kernelShape,
        stride = l.stride,
        padding = l.padding,
        dilation = Seq(),
        groups = l.input.numChannels,
        outChannels = l.output.numChannels
      )
    case _ => throw new RuntimeException
  }

  /** Coverts a LBIR LayerWrap object to a kernel map. (e.g. see use in combinational/NeuronProcessingUnit.scala)
    *
    * @param layer
    *   The LBIR layer object to convert to a kernel map
    * @return
    *   The kernel map, i.e. for each output which kernel values are relevant?
    */
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

  /** Coverts a LBIR LayerWrap object to a threshold map. (e.g. see use in combinational/NeuronProcessingUnit.scala)
    *
    * @param layer
    *   The LBIR layer object to convert to a threshold map
    * @return
    *   The threshold map, i.e. for each output which threshold value is relevant?
    */
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

  /** Coverts a LBIR LayerWrap object to a shift map. (e.g. see use in combinational/NeuronProcessingUnit.scala)
    *
    * @param layer
    *   The LBIR layer object to convert to a shift map
    * @return
    *   The shift map, i.e. for each output which shift value is relevant?
    */
  def layerToShiftMap(layer: LayerWrap with IsActiveLayer): Seq[Int] = layer match {
    case l: DenseConfig => {
      l.kernel.dtype.shift.length match {
        case 1                              => Seq.fill(l.output.numParams)(0)
        case x if (x == l.output.numParams) => 0 until l.output.numParams
        case _                              => throw new RuntimeException
      }
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

  /** Given an input map for a particular output return the chisel values of the input.
    *
    * @param input
    *   The entire input values sequence.
    * @param receptiveField
    *   The input map for a particular output (i.e. which inputs to take).
    * @param zeroGen
    *   In place of the -1 map value place this value (signifies padding)
    * @param skipPadding
    *   Skip values that contain -1. Used in maxpool because padding values should not be compared (see
    *   OrderProcessingUnit).
    *
    * @return
    *   Chisel values of the receptive field of the input.
    */
  def getReceptiveField[T <: Data](
    input:          Seq[T],
    receptiveField: Seq[Int],
    zeroGen:        Option[T],
    skipPadding:    Boolean = false
  ): Seq[T] = {
    val filiteredField = if (skipPadding) {
      receptiveField.filter(_ != -1) // -1 signals a padding value
    } else {
      receptiveField
    }
    Seq(filiteredField.map {
      case -1 => zeroGen.get // -1 signals a padding value
      case x  => input(x)
    }: _*)
  }

  /** Transforms the LBIR LayerWrap values of kernel into properly shaped Chisel sequence.
    *
    * @param layer
    *   The LBIR layer in question.
    *
    * @return
    *   A sequence of chisel values representing the kernel of a LBIR active layer.
    */
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

  /** Transforms the LBIR LayerWrap values of threshold into properly shaped Chisel sequence.
    *
    * @param layer
    *   The LBIR layer in question.
    *
    * @return
    *   A sequence of chisel values representing the thresholds of a LBIR active layer.
    */
  def getThresh[T <: Data](layer: LayerWrap with HasInputOutputQTensor with IsActiveLayer): Seq[T] =
    (layer.input.dtype.quantization, layer.kernel.dtype.quantization, layer.activation) match {
      case (BINARY, BINARY, lbir.Activation.BINARY_SIGN) =>
        layer.thresh.values
          .map(x => (layer.numActiveParams + x) / 2)
          .map(_.ceil)
          .map((x: Float) => saturateToZero(x))
          .map(_.toInt.U)
          .map(_.asInstanceOf[T])
      case _ => layer.thresh.values.map(_.toInt.S).map(_.asInstanceOf[T])
    }
}

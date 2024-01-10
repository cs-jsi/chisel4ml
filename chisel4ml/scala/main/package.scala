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
import lbir.Activation.{BINARY_SIGN, RELU, NO_ACTIVATION}
import lbir.Datatype.QuantizationType.{BINARY, UNIFORM}
import lbir.{AXIStreamLBIRDriver, Datatype, DenseConfig, QTensor, IsActiveLayer}
import chisel4ml.Quantization._
import org.slf4j.LoggerFactory
import chisel4ml.util._
import interfaces.amba.axis._

import scala.language.implicitConversions

package object implicits {
  val logger = LoggerFactory.getLogger("chisel4ml")

  implicit def axiStreamToDriver(x: AXIStreamIO[UInt]): AXIStreamDriver[UInt] = new AXIStreamDriver(x)

  implicit def axiStreamToLBIRDriver(x: AXIStreamIO[UInt]): AXIStreamLBIRDriver = {
    new AXIStreamLBIRDriver(new AXIStreamDriver(x))
  }

  implicit class ActiveConfigExtensions(layer: IsActiveLayer) 
  {
    def getQuantizationContext[I <: Bits, W <: Bits, M <: Bits, A <: Bits, O <: Bits] = (
      layer.input.dtype.quantization,
      layer.input.dtype.signed,
      layer.kernel.dtype.quantization,
      layer.activation
    ) match {
      case (UNIFORM, true, UNIFORM, RELU) => new UniformQuantizationContextSSUReLU(layer.roundingMode)
      case (UNIFORM, false, UNIFORM, RELU) => new UniformQuantizationContextUSUReLU(layer.roundingMode)
      case (UNIFORM, true, UNIFORM, NO_ACTIVATION) => new UniformQuantizationContextSSSNoAct(layer.roundingMode)
      case (UNIFORM, false, UNIFORM, NO_ACTIVATION) => new UniformQuantizationContextUSSNoAct(layer.roundingMode)
      case _ => throw new RuntimeException()
  }
  }

  implicit class DenseConfigExtensions(layer: DenseConfig) {
    def getThresh[T <: Bits]: Seq[T] =
      (layer.input.dtype.quantization, layer.kernel.dtype.quantization, layer.activation) match {
        case (BINARY, BINARY, BINARY_SIGN) =>
          layer.thresh.values.map(x => (layer.input.shape(0) + x) / 2).map(_.ceil).map(_.toInt.U).map(_.asInstanceOf[T])
        case _ => layer.thresh.values.map(_.toInt.S(layer.thresh.dtype.bitwidth.W)).map(_.asInstanceOf[T])
      }
    def getWeights[T <: Bits]: Seq[Seq[T]] = layer.kernel.dtype.quantization match {
      case UNIFORM =>
        layer.kernel.values.map(_.toInt.S.asInstanceOf[T]).grouped(layer.kernel.shape(0)).toSeq.transpose
      case BINARY =>
        layer.kernel.values.map(_ > 0).map(_.B.asInstanceOf[T]).grouped(layer.kernel.shape(0)).toSeq.transpose
      case _ => throw new RuntimeException
    }
  }

  implicit class QTensorExtensions(qt: QTensor) {
    /* LBIR Transactions contain all parameters bit packed, with no parameter being
     * separated into two transactions. Thus, depending on the bitwidth of parameters and
     * the bus width there could be some stuffing bits.
     *
     * i.e. say we have 5-bit parameters and the following values in qt: (1, 2, 3, 4, 3, 2, 1).
     * For a 32-bit bus we need 2 transactions, since we have 7 parameters. One transaction can have
     * at most 6 parameters, and the 2-left over bits are empty with dont care (zeros in code).
     * The following lines show the correct transaction data for the above case:
     *  xx_00010_00011_00100_00011_00010_00001
     *  xx_xxxxx_xxxxx_xxxxx_xxxxx_xxxxx_00001
     */
    def getType[T <: Bits]: T = (qt.dtype.quantization, qt.dtype.signed) match {
      case (BINARY, _)      => Bool().asInstanceOf[T]
      case (UNIFORM, true)  => SInt(qt.dtype.bitwidth.W).asInstanceOf[T]
      case (UNIFORM, false) => UInt(qt.dtype.bitwidth.W).asInstanceOf[T]
      case _                => throw new Exception("Datatype not supported.")
    }
    
    def toLBIRTransactions(busWidth: Int): Seq[UInt] = {
      require(busWidth % qt.dtype.bitwidth == 0)
      val binaryStr = qt.toBinaryString
      val paramWidth = qt.dtype.bitwidth
      val paramsPerTransaction: Int = busWidth / paramWidth
      val transactions: Seq[String] = binaryStr
        .drop(1) // drop the "b" symbol
        .grouped(paramWidth)
        .toSeq
        .reverse
        .grouped(paramsPerTransaction)
        .map(
          _.reverse.reduce(_ + _)
        )
        .toSeq
      transactions.map(s"b".concat(_).U(busWidth.W))
    }

    def toUInt: UInt = {
      qt.toBinaryString.U
    }

    def toBinaryString: String = {
      var values = qt.values.reverse
      if (qt.dtype.quantization == BINARY) {
        values = values.map(x => (x + 1) / 2) // 1 -> 1, -1 -> 0
      }
      "b".concat(values.map(_.toInt).map(toBinary(_, qt.dtype.bitwidth)).mkString)
    }

    def toHexString(memWordWidth: Int = 32): String = {
      require(memWordWidth >= qt.dtype.bitwidth)
      val bitwidth:       Int = qt.dtype.bitwidth
      val paramsPerWord:  Int = memWordWidth / bitwidth
      val memValidBits:   Int = paramsPerWord * bitwidth
      val memInvalidBits: Int = memWordWidth - memValidBits
      val numKernels:     Int = if (qt.shape.length == 4) qt.shape(0) else 1
      val totalElements:  Int = qt.shape.reduce(_ * _)
      val elemPerKernel:  Int = totalElements / numKernels
      require(elemPerKernel * numKernels == totalElements, "All tensor must be of the same size.")

      var values:            Seq[Float] = Seq()
      var realElemPerKernel: Int = 0
      // We insert zeros where necesessary for kernel alignment (each kernel goes to new word)
      val finalElemOffset = elemPerKernel % paramsPerWord
      if (finalElemOffset != 0) {
        realElemPerKernel = elemPerKernel + (paramsPerWord - finalElemOffset)
        for (i <- 0 until numKernels) {
          values = values :++ qt.values.grouped(elemPerKernel).toSeq(i)
          values = values :++ Seq.fill(paramsPerWord - finalElemOffset)(0.0f)
        }
      } else {
        values = qt.values
        realElemPerKernel = elemPerKernel
      }

      var hex: String = ""
      var bin: Seq[String] = Seq()
      var tmp: String = ""
      for (kernel <- values.grouped(realElemPerKernel)) {
        for (paramGroup <- kernel.grouped(paramsPerWord)) {
          tmp = ""
          for (param <- paramGroup) {
            tmp = toBinary(param.toInt, bitwidth) + " " + tmp
          }
          if (memInvalidBits > 0) {
            tmp = toBinary(0, memInvalidBits) + " " + tmp + "\n"
          } else {
            tmp = tmp + "\n"
          }
          bin = bin :+ tmp
        }
      }
      for (binStr <- bin) {
        tmp = BigInt(binStr.trim.replaceAll(" ", ""), 2).toString(16)
        tmp = tmp.reverse.padTo(8, "0").reverse.mkString
        hex = hex + tmp + s" // " + binStr
      }
      hex
    }

    // KCHW layout
    def width:  Int = qt.shape.reverse(0)
    def height: Int = qt.shape.reverse(1)
    def numChannels: Int = qt.shape.length match {
      case 4 => qt.shape(1)
      case 3 => qt.shape(0)
      case _ => throw new RuntimeException("Tensor shape not appropriate.")
    }
    def numKernels: Int = qt.shape.length match {
      case 4 => qt.shape(0)
      case _ => throw new RuntimeException("Shape to small.")
    }
    def numParams:       Int = qt.shape.reduce(_ * _)
    def numKernelParams: Int = numParams / numKernels
    def numActiveParams(depthwise: Boolean): Int = if (depthwise) width * height else numKernelParams
    def paramsPerWord(wordSize: Int = 32): Int = {
      require(wordSize >= qt.dtype.bitwidth)
      wordSize / qt.dtype.bitwidth
    }
    def totalBitwidth: Int = qt.dtype.bitwidth * numParams
    def memDepth(memWordSize: Int): Int = {
      qt.shape.length match {
        // Each kernel goes to a new word!
        case 4 => math.ceil(numKernelParams.toFloat / paramsPerWord(memWordSize).toFloat).toInt * numKernels
        case _ => math.ceil(numParams.toFloat / paramsPerWord(memWordSize).toFloat).toInt
      }
    }
    def memDepthOneKernel(memWordSize: Int): Int = memDepth(memWordSize) / numKernels
    def numTransactions(busWidth: Int): Int = {
      require(
        busWidth >= qt.dtype.bitwidth,
        s"Buswidth must be at least the size of input data bitwidth. $busWidth !>= ${qt.dtype.bitwidth}"
      )
      val paramsPerTrans = paramsPerWord(busWidth)
      math.ceil(numParams.toFloat / paramsPerTrans.toFloat).toInt
    }
  }

  implicit class SeqBigIntExtensions(x: Seq[BigInt]) {
    /* LBIR Transactions -> QTensor
     *
     * Converts LBIR Transactions back to a QTensor. An appropriate stencil is needed as Seq[BigInt]
     * does not contain enough information to know how to reconstruct the QTensor.
     */
    def toQTensor(stencil: QTensor, busWidth: Int) = {
      require(
        busWidth >= stencil.dtype.bitwidth,
        s"""Invalid LBIR transaction. Buswidth must be greater than or equals
           | to the bitwidth of a single qtensor element. Buswidth is $busWidth,
           | bitwidth:${stencil.dtype.bitwidth}.""".stripMargin.replaceAll("\n", "")
      )
      val bitsPerTransaction:   Int = stencil.paramsPerWord(busWidth) * stencil.dtype.bitwidth
      logger.debug(
        s"$stencil, busWidth:$busWidth, bitsPerTranscation:$bitsPerTransaction"
      )

      val binaryVals = x
        .map((a: BigInt) => toBinaryB(a, bitsPerTransaction))
        .map(
          _.grouped(stencil.dtype.bitwidth).toList.reverse
        )
      val flatVals = binaryVals.flatten.map(BigInt(_, 2).toFloat)
      val values = flatVals.toList.dropRight(flatVals.length - stencil.shape.reduce(_ * _))
      val valuesMod = if (stencil.dtype.quantization == BINARY) {
        values.map(x => (x * 2) - 1)
      } else {
        values.map(signedCorrect(_, stencil.dtype))
      }
      logger.debug(
        s"""Converted Seq[BigInt] to QTensor. Values: $values, valuesMod: $valuesMod.
           | Original Seq: $x, bitsPerTransaction: $bitsPerTransaction, 
           | flatVals: $flatVals. binaryVals: $binaryVals""".stripMargin
          .replaceAll("\n", "")
      )
      QTensor(dtype = stencil.dtype, shape = stencil.shape, values = valuesMod)

    }
  }

  implicit class SeqIntHelperExtensions(x: Seq[Int]) {
    def BQ(shape: Seq[Int] = Seq()): QTensor = {
      QTensor(
        dtype = Datatype(quantization = BINARY, bitwidth = 1, signed = true, shift = Seq(0), offset = Seq(0)),
        shape = if (shape.isEmpty) Seq(x.length) else shape,
        values = x.map(_.toFloat)
      )
    }

    def UQ(bw: Int, shape: Seq[Int] = Seq()): QTensor = {
      QTensor(
        dtype = Datatype(quantization = UNIFORM, bitwidth = bw, signed = false, shift = Seq(0), offset = Seq(0)),
        shape = if (shape.isEmpty) Seq(x.length) else shape,
        values = x.map(_.toFloat)
      )
    }

    def SQ(bw: Int, shape: Seq[Int] = Seq()): QTensor = {
      QTensor(
        dtype = Datatype(quantization = UNIFORM, bitwidth = bw, signed = true, shift = Seq(0), offset = Seq(0)),
        shape = if (shape.isEmpty) Seq(x.length) else shape,
        values = x.map(_.toFloat)
      )
    }
  }

}

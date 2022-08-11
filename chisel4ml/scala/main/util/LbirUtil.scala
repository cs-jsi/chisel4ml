/*
 * Contains functions
 */

package chisel4ml.util

import chisel3._
import _root_.lbir._

import _root_.org.slf4j.Logger
import _root_.org.slf4j.LoggerFactory

import _root_.scala.math.log

trait ThreshProvider[T <: Bits] {
    def instance(tensor: QTensor, fanIn: Int): Seq[T]
}

object ThreshProvider {
    def transformThresh[T <: Bits : ThreshProvider](tensor: QTensor, fanIn: Int): Seq[T] = {
        implicitly[ThreshProvider[T]].instance(tensor, fanIn)
    }
    // Binarized neurons
    implicit object ThreshProviderUInt extends ThreshProvider[UInt] {
        def instance(tensor: QTensor, fanIn: Int): Seq[UInt] = {
            LbirUtil.logger.info(s"""Transformed input tensor of thresholds to a Seq[UInt]. The input fan-in is
                                     | fanIn""".stripMargin.replaceAll("\n", ""))
            tensor.values.map(x => (fanIn + x) / 2).map(_.ceil).map(_.toInt.U)
        }
    }
    implicit object ThreshProviderSInt extends ThreshProvider[SInt] {
        def instance(tensor: QTensor, fanIn: Int): Seq[SInt] = {
            LbirUtil.logger.info(s"""Transformed input tensor of thresholds to a Seq[SInt].""")
            tensor.values.map(_.toInt.S)
        }
    }
}

trait WeightsProvider[T <: Bits] {
    def instance(tensor: QTensor): Seq[Seq[T]]
}

object WeightsProvider {
    def transformWeights[T <: Bits : WeightsProvider](tensor: QTensor): Seq[Seq[T]] = implicitly[WeightsProvider[T]].instance(tensor)
    implicit object WeightsProviderBool extends WeightsProvider[Bool] {
        def instance(tensor: QTensor): Seq[Seq[Bool]] = {
            LbirUtil.logger.info(s"""Transformed input tensor of weights to a Seq[Seq[Bool]].""")
            tensor.values.map(_ > 0).map(_.B).grouped(tensor.shape(1)).toSeq.transpose
        }
    }
    implicit object WeightsProviderSInt extends WeightsProvider[SInt] {
        def instance(tensor: QTensor): Seq[Seq[SInt]] = {
            LbirUtil.logger.info(s"""Transformed input tensor of weights to a Seq[Seq[SInt]].""")
            tensor.values.map(_.toInt.S).grouped(tensor.shape(1)).toSeq.transpose
        }
    }
}

final class LbirUtil
object LbirUtil {
    val logger = LoggerFactory.getLogger(classOf[LbirUtil])
    def transformWeights[T <: Bits : WeightsProvider](tensor: QTensor): Seq[Seq[T]] = WeightsProvider.transformWeights[T](tensor)

    def transformThresh[T <: Bits : ThreshProvider](tensor: QTensor, fanIn: Int): Seq[T] = ThreshProvider.transformThresh[T](tensor, fanIn)

    def qtensorTotalBitwidth(tensor: QTensor): Int = { tensor.dtype.get.bitwidth * tensor.shape.reduce(_ * _) }
    
    private def toBinary(i: Int, digits: Int = 8) = String.format("%" + digits + "s", i.toBinaryString).replace(' ', '0')

    private def toBinaryB(i: BigInt, digits: Int = 8) = String.format("%" + digits + "s", i.toString(2)).replace(' ', '0')

    def qtensorToBigInt(qtensor: QTensor): BigInt = {
        var values = qtensor.values.reverse
        if (qtensor.dtype.get.quantization == Datatype.QuantizationType.BINARY) {
            values = values.map(x => (x + 1) / 2) // 1 -> 1, -1 -> 0
        }

        val string_int = values.map(x => toBinary(x.toInt, qtensor.dtype.get.bitwidth)).mkString
        val big_int    = BigInt(string_int, radix = 2)
        logger.info(s"""Converted lbir.QTensor: ${qtensor.values} to BigInt: ${string_int}. 
              		    | The number of bits is: ${qtensor.dtype.get.bitwidth}."""
        )
        big_int
    }

    def bigIntToQtensor(value: BigInt, outSize: Int): QTensor = {
        val dataType    = Datatype(quantization = Datatype.QuantizationType.BINARY, 
                                   bitwidth = 1, 
                                   scale = Seq(1), 
                                   offset = Seq(0))
        // We substract the 48 because x is an ASCII encoded symbol
        val lbir_values = toBinaryB(value, outSize).toList.map(x => x.toFloat - 48).reverse.map(x => (x * 2) - 1)
        val qtensor     = QTensor(dtype = Option(dataType), shape = List(outSize), values = lbir_values)
        logger.info(
          "Converted BigInt: " + value + " to lbir.QTensor: " + qtensor + ". The number of bits is " + outSize + "."
        )
        qtensor
    }

    def log2(x: Int): Int = (log(x) / log(2)).toInt
}

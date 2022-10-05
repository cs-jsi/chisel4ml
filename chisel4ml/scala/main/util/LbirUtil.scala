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
            tensor.values.map(_.toInt.S(tensor.dtype.get.bitwidth.W))
        }
    }
}

trait WeightsProvider[T <: Bits] {
    def instance(tensor: QTensor): Seq[Seq[T]]
}

object WeightsProvider {
    def transformWeights[T <: Bits : WeightsProvider](tensor: QTensor): Seq[Seq[T]] = {
        implicitly[WeightsProvider[T]].instance(tensor)
    }
    
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
    
    def transformWeights[T <: Bits : WeightsProvider](tensor: QTensor): Seq[Seq[T]] = {
        WeightsProvider.transformWeights[T](tensor)
    }
    
    def transformThresh[T <: Bits : ThreshProvider](tensor: QTensor, fanIn: Int): Seq[T] = {
        ThreshProvider.transformThresh[T](tensor, fanIn)
    }

    // Creates a sequence of UInts of size busWidth
    def toUIntSeq(x: UInt, busWidth: Int): Seq[UInt] = {
        val numOfBusTrans = math.ceil(x.getWidth.toFloat / busWidth.toFloat).toInt
        x.asTypeOf(Vec(numOfBusTrans, UInt(busWidth.W)))
    }

    def mergeUIntSeq(x: Seq[UInt]): UInt = {
        VecInit(x).asUInt
    }

    def log2(x: Int): Int = (log(x) / log(2)).toInt
}

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
package chisel4ml.util

import chisel3._
import _root_.chisel4ml._
import _root_.chisel4ml.implicits._
import _root_.lbir._

import _root_.java.nio.file.{Path, Paths}
import _root_.java.io.{BufferedWriter, FileWriter}
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
            LbirUtil.logger.debug(s"""Transformed input tensor of thresholds to a Seq[UInt]. The input fan-in is
                                     | fanIn""".stripMargin.replaceAll("\n", ""))
            tensor.values.map(x => (fanIn + x) / 2).map(_.ceil).map(_.toInt.U)
        }
    }

    implicit object ThreshProviderSInt extends ThreshProvider[SInt] {
        def instance(tensor: QTensor, fanIn: Int): Seq[SInt] = {
            LbirUtil.logger.debug(s"""Transformed input tensor of thresholds to a Seq[SInt].""")
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
            LbirUtil.logger.debug(s"""Transformed input tensor of weights to a Seq[Seq[Bool]].""")
            tensor.values.map(_ > 0).map(_.B).grouped(tensor.shape(1)).toSeq.transpose
        }
    }

    implicit object WeightsProviderSInt extends WeightsProvider[SInt] {
        def instance(tensor: QTensor): Seq[Seq[SInt]] = {
            LbirUtil.logger.debug(s"""Transformed input tensor of weights to a Seq[Seq[SInt]].""")
            tensor.values.map(_.toInt.S).grouped(tensor.shape(1)).toSeq.transpose
        }
    }
}

final class LbirUtil
object LbirUtil {
    var cnt: Int = 0
    var directory: Path = Paths.get("")

    def setDirectory(dir: Path) = {
        directory = dir
        cnt = 0
    }

    val logger = LoggerFactory.getLogger(classOf[LbirUtil])

    def transformWeights[T <: Bits : WeightsProvider](tensor: QTensor): Seq[Seq[T]] = {
        WeightsProvider.transformWeights[T](tensor)
    }

    def transformThresh[T <: Bits : ThreshProvider](tensor: QTensor, fanIn: Int): Seq[T] = {
        ThreshProvider.transformThresh[T](tensor, fanIn)
    }

    def log2(x: Int): Int = (log(x) / log(2)).toInt
    def log2(x: Float): Float = (log(x) / log(2.0)).toFloat

    def createHexMemoryFile(tensor: QTensor): String = {
        val fPath = Paths.get(directory.toString, s"mem$cnt.hex").toAbsolutePath()
        val relPath = Paths.get("").toAbsolutePath().relativize(fPath)
        val writer = new BufferedWriter(new FileWriter(fPath.toString))
        writer.write(tensor.toHexStr)
        writer.close()
        logger.debug(s"Created new memory file: ${fPath.toString}.")
        cnt = cnt + 1
        relPath.toString
    }
}

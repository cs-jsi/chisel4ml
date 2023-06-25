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

import _root_.chisel3._
import _root_.lbir.{QTensor, Datatype, AXIStreamLBIRDriver}
import _root_.lbir.Datatype.QuantizationType.{BINARY, UNIFORM}
import interfaces.amba.axis._

import _root_.org.slf4j.Logger
import _root_.org.slf4j.LoggerFactory
import _root_.scala.math.pow
import scala.language.implicitConversions

package object implicits {
    val logger = LoggerFactory.getLogger("chisel4ml")

    implicit def axiStreamToDriver[T <: Data](x: AXIStreamIO[T]): AXIStreamDriver[T] = new AXIStreamDriver(x)

    implicit def axiStreamToLBIRDriver(x: AXIStreamIO[UInt]): AXIStreamLBIRDriver = {
        new AXIStreamLBIRDriver(new AXIStreamDriver(x))
    }

    def toBinary(i: Int, digits: Int = 8): String = String.format(s"%${digits}s",
                                                            i.toBinaryString.takeRight(digits)).replace(' ', '0')
    def toBinaryB(i: BigInt, digits: Int = 8): String = String.format("%" + digits + "s", i.toString(2)).replace(' ', '0')
	def signedCorrect(x: Float, dtype: Datatype): Float = {
        if (dtype.signed && x > (pow(2,dtype.bitwidth-1) - 1))
            x - pow(2, dtype.bitwidth).toFloat
        else
            x
    }

    // Allows QTensor to be converted to chisel UInt
    implicit class QTensorAddOns(qt: QTensor) {
        def toBinaryString: String = {
            var values = qt.values.reverse
            if (qt.dtype.get.quantization == BINARY) {
                values = values.map(x => (x + 1) / 2) // 1 -> 1, -1 -> 0
            }
            "b".concat(values.map(_.toInt).map(toBinary(_, qt.dtype.get.bitwidth)).mkString)
        }
        def toUInt: UInt = {
            logger.debug(s"Converting QTensor to an UInt.")
            val binaryString = qt.toBinaryString
            binaryString.U(qt.totalBitwidth.W)
        }

        def totalBitwidth: Int = qt.dtype.get.bitwidth * qt.shape.reduce(_ * _)
    }

    implicit class BigIntSeqToUInt(x: Seq[BigInt]) {
        def toUInt(busWidth:Int): UInt = {
            logger.debug(s"Converting Seq[BigInt]=$x to an UInt.")
            val totalWidth = busWidth * x.length
            "b".concat(x.map( (a:BigInt) => toBinaryB(a, busWidth) ).reverse.mkString).U(totalWidth.W)
        }
    }

    implicit class IntSeqToUInt(x: Seq[Int]) {
        def BQ: QTensor = {
            QTensor(dtype=Some(Datatype(quantization=BINARY,
                                        bitwidth=1,
                                        signed=true,
                                        shift=Seq(0),
                                        offset=Seq(0))),
                    shape=Seq(x.length),  // TODO: add multu-dim support
                    values=x.map(_.toFloat)
            )
        }

        def UQ(bw: Int): QTensor = {
            QTensor(dtype=Some(Datatype(quantization=UNIFORM,
                                        bitwidth=bw,
                                        signed=false,
                                        shift=Seq(0),
                                        offset=Seq(0))),
                    shape=Seq(x.length),
                    values=x.map(_.toFloat)
            )
        }
    }

    // And vice versa
    implicit class UIntToQTensor(x: UInt) {
        def toQTensor(stencil: QTensor) = {
            val valuesString = toBinaryB(x.litValue, stencil.totalBitwidth).grouped(stencil.dtype.get.bitwidth).toList
            val values = valuesString.reverse.map(Integer.parseInt(_, 2).toFloat)
            val valuesMod = if (stencil.dtype.get.quantization == BINARY) {
                values.map(x => (x * 2) - 1)
            } else {
                values.map(signedCorrect(_, stencil.dtype.get))
            }
            logger.debug(s"""Converted UInt to QTensor. ValuesString: $valuesString, values: $values,
							 | valuesMod: $valuesMod. Uint val: $x, LitValue: ${x.litValue}, Binary string:
                             | ${toBinaryB(x.litValue, stencil.totalBitwidth)}""".stripMargin.replaceAll("\n", ""))
            QTensor(dtype = stencil.dtype,
                    shape = stencil.shape,
                    values = valuesMod)
        }

        def toUIntSeq(busWidth: Int): Seq[UInt] = {
            val numOfBusTrans = math.ceil(x.getWidth.toFloat / busWidth.toFloat).toInt
            val temp0 = toBinaryB(x.litValue, x.getWidth)
            val temp1 = temp0.grouped(busWidth).toList
            temp1.map("b".concat(_).U(busWidth.W))
        }
    }


}

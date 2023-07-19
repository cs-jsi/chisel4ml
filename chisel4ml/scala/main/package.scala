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
import chisel4ml.util._
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

		def toHexStr: String = {
      		logger.debug("Convertin QTensor to a hex file string.")
      		val bitwidth:       Int = qt.dtype.get.bitwidth
      		val memWordWidth:   Int = 32
      		val paramsPerWord:  Int = memWordWidth / bitwidth
      		val memValidBits:   Int = paramsPerWord * bitwidth
      		val memInvalidBits: Int = memWordWidth - memValidBits
      		val numKernels:     Int = if (qt.shape.length == 4) qt.shape(0) else 1
      		val totalElements:  Int = qt.shape.reduce(_ * _)
      		val elemPerKernel:  Int = totalElements / numKernels
      		require(elemPerKernel * numKernels == totalElements, "All tensor must be of the same size.")


      		var values: Seq[Float]     = Seq()
      		var realElemPerKernel: Int = 0
      		// We insert zeros where necesessary for kernel alignment (each kernel goes to new word)
      		val finalElemOffset = elemPerKernel % paramsPerWord
      		if (finalElemOffset != 0) {
      		  realElemPerKernel = elemPerKernel + (paramsPerWord - finalElemOffset)
      		  for (i <- 0 until numKernels) {
      		    values = values :++ qt.values.grouped(elemPerKernel).toSeq(i)
      		    values = values :++ Seq.fill(paramsPerWord - finalElemOffset)(0.0F)
      		  }
      		} else {
      		  values = qt.values
      		  realElemPerKernel = elemPerKernel
      		}

      		var hex: String      = ""
      		var bin: Seq[String] = Seq()
      		var tmp: String      = ""
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
            val values = valuesString.reverse.map(BigInt(_, 2).toFloat)
            val valuesMod = if (stencil.dtype.get.quantization == BINARY) {
                values.map(x => (x * 2) - 1)
            } else {
                values.map(signedCorrect(_, stencil.dtype.get))
            }
            logger.info(s"""Converted UInt to QTensor. ValuesString: $valuesString, values: $values,
							 | valuesMod: $valuesMod. Uint val: $x, LitValue: ${x.litValue}, Binary string:
                             | ${toBinaryB(x.litValue, stencil.totalBitwidth)}""".stripMargin.replaceAll("\n", ""))
            QTensor(dtype = stencil.dtype,
                    shape = stencil.shape,
                    values = valuesMod)
        }

        def toUIntSeq(busWidth: Int, paramWidth: Int): Seq[UInt] = {
            val binaryStr = toBinaryB(x.litValue, x.getWidth)
            val emptyBits = busWidth % paramWidth
            val dataBits = busWidth - emptyBits
            val paramsPerTransaction: Int = busWidth / paramWidth
            val transactions = binaryStr.grouped(paramWidth).toList.reverse.grouped(paramsPerTransaction).map(
                _.reverse.reduce(_ + _)
            ).toList
            transactions.map(s"b${"0"*emptyBits}".concat(_).U(busWidth.W))
        }
    }


}

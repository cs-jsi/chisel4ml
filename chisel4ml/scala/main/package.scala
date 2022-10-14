/*
 *
 */

import _root_.chisel3._
import _root_.lbir.{QTensor, Datatype}
import _root_.lbir.Datatype.QuantizationType.BINARY

import _root_.org.slf4j.Logger
import _root_.org.slf4j.LoggerFactory
import _root_.scala.math.pow

package object chisel4ml {
    val logger = LoggerFactory.getLogger("chisel4ml")
    
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
        def toUInt: UInt = {
            logger.debug(s"Converting QTensor to an UInt.")
            var values = qt.values.reverse 
            if (qt.dtype.get.quantization == BINARY) {
                values = values.map(x => (x + 1) / 2) // 1 -> 1, -1 -> 0
            }
            "b".concat(values.map(_.toInt).map(toBinary(_, qt.dtype.get.bitwidth)).mkString).U
        }

        def totalBitwidth: Int = qt.dtype.get.bitwidth * qt.shape.reduce(_ * _)

        def toHexStr: String = {
            """abcd
0123
4444
ffff"""
        }
    }

    implicit class BigIntSeqToUInt(x: Seq[BigInt]) {
        def toUInt(busWidth:Int): UInt = {
            logger.debug(s"Converting Seq[BigInt] to an UInt.")
            val totalWidth = busWidth * x.length
            "b".concat(x.map( (a:BigInt) => toBinaryB(a, busWidth) ).mkString).U(totalWidth.W)
        }
    }

    // And vice versa
    implicit class UIntToQTensor(x: UInt) {
        def toQTensor(stencil: QTensor) = {

            val valuesString = toBinaryB(x.litValue, stencil.totalBitwidth).grouped(stencil.dtype.get.bitwidth).toList
            val values = valuesString.map(Integer.parseInt(_, 2).toFloat).reverse
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

package chisel4ml.tests

import org.scalatest.funsuite.AnyFunSuite
import _root_.lbir.{QTensor, Datatype}
import _root_.lbir.Datatype.QuantizationType.{BINARY, UNIFORM}
import _root_.chisel4ml._
import _root_.chisel4ml.implicits._
import _root_.chisel3._

class LbirChiselConversionTests extends AnyFunSuite {
    val binaryDatatype = Some(new Datatype(quantization=BINARY,
                                           bitwidth=1,
                                           signed=true,
                                           shift=Seq(0),
                                           offset=Seq(0)))

    // TEST QTensor -> UInt WITH TEST VECTORS
    val testVectors = List(Seq(-1, -1, -1, 1).BQ -> "b1000".U.litValue,
                           Seq(1, 1, 1, -1).BQ -> "b0111".U.litValue,
                           Seq(4, 3, 2, 1).UQ(bw=4) -> "b0001_0010_0011_0100".U.litValue,
                      )

    for ( ((qtensor, goldenValue), idx) <- testVectors.zipWithIndex) {
        test(s"Testing QTensor to UInt conversion test vector: $idx") {
            assert(qtensor.toUInt.litValue == goldenValue.U.litValue)
        }
    }

    // Randomly test QTensor -> UInt -> QTensor
    val numUniformTests = 20
    val r = new scala.util.Random
    r.setSeed(42L)
    for (idx <- 0 until numUniformTests) {
        val signed = r.nextFloat() < 0.5 // true/false
        val bitwidth = r.nextInt(15) + 2  // 2-16
        val length = r.nextInt(128) + 1  // 1-128
        val signAdjust = if (signed) Math.pow(2, bitwidth - 1).toFloat else 0.0F
        val values = (0 until length).map(_ => r.nextInt(Math.pow(2, bitwidth).toInt).toFloat - signAdjust)
        val dtype = Some(Datatype(quantization=UNIFORM,
                                                  signed=signed,
                                                  bitwidth=bitwidth,
                                                  shift=Seq(0),
                                                  offset=Seq(0)))
        val qtensor = QTensor(dtype=dtype,
                              shape=Seq(length),
                              values=values)
        val stencil = QTensor(dtype=dtype,
                              shape=Seq(length))
        test(s"Random test QTensor -> UInt -> QTensor: $idx") {
            assert(qtensor.toUInt.toQTensor(stencil).values == qtensor.values,
                s"""QTensor was first converted to ${qtensor.toUInt}, with total bitwidth ${qtensor.totalBitwidth} and
                | then back to a qtensor: ${qtensor.toUInt.toQTensor(stencil).values}.
                | The QTensor was signed:$signed, bitwidth: $bitwidth.""".stripMargin.replaceAll("\n", ""))
        }
    }
}

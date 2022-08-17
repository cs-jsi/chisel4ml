package chisel4ml.tests

import org.scalatest.funsuite.AnyFunSuite
import _root_.chisel4ml.util.LbirUtil
import _root_.lbir.{QTensor, Datatype}
import _root_.lbir.Datatype.QuantizationType.{BINARY, UNIFORM}

class LbirUtilTests extends AnyFunSuite {
    val binaryDatatype = Some(new Datatype(quantization=BINARY,
                                           bitwidth=1,
                                           signed=true,
                                           scale=Seq(1),
                                           offset=Seq(0)))

    // QTENSOR -> BIGINT
    test("Binary tensor conversion test 0") {
        val qtensor = new QTensor(dtype = binaryDatatype,
                                  shape = Seq(4),
                                  values = Seq(-1, -1, -1, 1))

        assert(LbirUtil.qtensorToBigInt(qtensor) == BigInt("1000", radix=2))
    }

    test("Binary tensor conversion test 1") {
        val qtensor = new QTensor(dtype = binaryDatatype,
                                  shape = Seq(4),
                                  values = Seq(1, 1, 1, -1))

        assert(LbirUtil.qtensorToBigInt(qtensor) == BigInt("0111", radix=2))
    }
    
    test("Uniformly quantized tensor to 4-bits conversion test 0") {
        val uniformFourBitNoscaleType = Some(new Datatype(quantization=UNIFORM,
                                                   signed=false,
                                                   bitwidth=4,
                                                   scale=Seq(1),
                                                   offset=Seq(0)))
        val qtensor = new QTensor(dtype = uniformFourBitNoscaleType,
                                  shape = Seq(4),
                                  values = Seq(4, 3, 2, 1))

        assert(LbirUtil.qtensorToBigInt(qtensor) == BigInt("0001 0010 0011 0100".filterNot(_.isWhitespace), radix=2))
    }

    // BIGINT -> QTENSOR
    test("Convert back a BigInt to a uniformy quantized QTensor") {
        val uniformFourBitNoscaleType = Some(new Datatype(quantization=UNIFORM,
                                                   signed=false,
                                                   bitwidth=4,
                                                   scale=Seq(1),
                                                   offset=Seq(0)))

        val stencil = new QTensor(dtype = uniformFourBitNoscaleType,
                                  shape = Seq(4))

        val qtensor = new QTensor(dtype = uniformFourBitNoscaleType,
                                  shape = Seq(4),
                                  values = Seq(4, 3, 2, 1))

        assert(LbirUtil.bigIntToQtensor((LbirUtil.qtensorToBigInt(qtensor)), stencil).values == qtensor.values)
    }

    test("Convert back a BigInt to a signed uniformy quantized QTensor") {
        val uniformFourBitNoscaleType = Some(new Datatype(quantization=UNIFORM,
                                                   signed=true,
                                                   bitwidth=4,
                                                   scale=Seq(1),
                                                   offset=Seq(0)))

        val stencil = new QTensor(dtype = uniformFourBitNoscaleType,
                                  shape = Seq(4))

        val qtensor = new QTensor(dtype = uniformFourBitNoscaleType,
                                  shape = Seq(4),
                                  values = Seq(-4, -3, 2, 1))

        assert(LbirUtil.bigIntToQtensor((LbirUtil.qtensorToBigInt(qtensor)), stencil).values == qtensor.values)
    }
}

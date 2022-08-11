package chisel4ml.tests

import org.scalatest.funsuite.AnyFunSuite
import _root_.chisel4ml.util.LbirUtil
import _root_.lbir.{QTensor, Datatype}
import _root_.lbir.Datatype.QuantizationType.BINARY

class LbirUtilTests extends AnyFunSuite {

    test("Binary tensor basic") {
        val qtensor = new QTensor(
            dtype = Some(new Datatype(quantization=BINARY,
                                 bitwidth=1,
                                 scale=Seq(1),
                                 offset=Seq(0))),
            shape = Seq(4),
            values = Seq(-1, -1, -1, 1)
        )

        assert(LbirUtil.qtensorToBigInt(qtensor).toString(2) == "1000")
    }
}

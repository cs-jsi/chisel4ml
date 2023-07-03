package chisel4ml.tests

import org.scalatest.flatspec.AnyFlatSpec
import chisel3._
import chiseltest._
import _root_.services.LayerOptions
import _root_.chisel4ml._
import _root_.chisel4ml.implicits._
import _root_.lbir._


class PeSequentialTestsTemp extends AnyFlatSpec with ChiselScalatestTester {
    behavior of "ProcessingElementSequential"

    val lbirLayer = lbir.Layer(
        ltype = lbir.Layer.Type.DENSE,
        thresh = Option(lbir.QTensor(
                            Option(lbir.Datatype(
                                quantization = lbir.Datatype.QuantizationType.UNIFORM,
                                signed = true,
                                bitwidth = 32,
                                shift = Seq(0),
                                offset = Seq(0)
                                )),
                            shape = Seq(4),
                            values = Seq(-2, -2, 0, -1)
                            )),
        weights = Option(lbir.QTensor(
                            Option(lbir.Datatype(
                                quantization = lbir.Datatype.QuantizationType.UNIFORM,
                                signed = true,
                                bitwidth = 4,
                                shift = Seq(-1, -2, 0, -2),
                                offset = Seq(0, 0, 0, 0)
                                )),
                            shape = Seq(3, 4),
                            values = Seq( 1,   2,   3,   4,
                                         -4,  -3,  -2,  -1,
                                          2,  -1,   1,   1)
                            )),
        input = Option(lbir.QTensor(Option(lbir.Datatype(
                                            quantization = lbir.Datatype.QuantizationType.UNIFORM,
                                            signed = true,
                                            bitwidth = 4,
                                            shift = Seq(0),
                                            offset = Seq(0)
                                    )),
                                    shape = Seq(3))),
        output = Option(lbir.QTensor(Option(lbir.Datatype(
                                            quantization = lbir.Datatype.QuantizationType.UNIFORM,
                                            signed = true,
                                            bitwidth = 4,
                                            shift = Seq(0),
                                            offset = Seq(0)
                                    )),
                                    shape = Seq(4))),
        activation = lbir.Layer.Activation.RELU
        )

    it should "send data through the pipeline." in {
   		test(new ProcessingElementSequentialConv(lbirLayer, LayerOptions())).withAnnotations(Seq(VerilatorBackendAnnotation)) { c =>
            c.inStream.initSource()
            c.outStream.initSink()

            c.inStream.enqueuePacket(Seq(3.U(32.W), 6.U(32.W), 9.U(32.W), 12.U(32.W)), c.clock)
            c.clock.step(3)
        }
    }
    // test(new ProcessingPipeline(new lbirModel)) TODO
}

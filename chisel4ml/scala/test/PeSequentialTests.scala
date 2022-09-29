package chisel4ml.tests

import org.scalatest.flatspec.AnyFlatSpec
import chisel3._
import chiseltest._
import _root_.services.GenerateCircuitParams.Options
import _root_.chisel4ml._
import _root_.lbir._


class PeSequentialTests extends AnyFlatSpec with ChiselScalatestTester {
    behavior of "ProcessingElementSequential"

    val lbirLayer = lbir.Layer(
        ltype = lbir.Layer.Type.DENSE,
        useBias = true,
        biases = Option(lbir.QTensor(
                            Option(lbir.Datatype(
                                quantization = lbir.Datatype.QuantizationType.UNIFORM,
                                signed = true,
                                bitwidth = 32,
                                scale = Seq(1),
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
                                scale = Seq(-1, -2, 0, -2),
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
                                            scale = Seq(1),
                                            offset = Seq(0)
                                    )),
                                    shape = Seq(3))),
        output = Option(lbir.QTensor(Option(lbir.Datatype(
                                            quantization = lbir.Datatype.QuantizationType.UNIFORM,
                                            signed = true,
                                            bitwidth = 4,
                                            scale = Seq(1),
                                            offset = Seq(0)
                                    )),
                                    shape = Seq(4))),
        activation = Option(lbir.Activation(lbir.Activation.Function.RELU))
        )

    it should "Test Sequential behavior for wrapped simple PEs." in {
        test(new ProcessingElementWrapSimpleToSequential(lbirLayer, Options())) { c =>
            c.io.inStream.data.initSource()
            c.io.inStream.data.setSourceClock(c.clock)
            c.io.outStream.data.initSink()
            c.io.outStream.data.setSinkClock(c.clock)
            
            c.io.inStream.last.poke(true.B)
            c.io.inStream.data.enqueue("b0001_0001_0001".U)
            c.io.inStream.last.poke(false.B)

            c.io.outStream.data.expectDequeue("b0010_0010_0010_0010".U)
        }
    }
    // test(new ProcessingPipeline(new lbirModel)) TODO
}

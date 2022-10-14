package chisel4ml.tests

import org.scalatest.flatspec.AnyFlatSpec
import chisel3._
import chiseltest._
import _root_.services.GenerateCircuitParams.Options
import _root_.chisel4ml._
import _root_.lbir._


class PeSequentialTestsTemp extends AnyFlatSpec with ChiselScalatestTester {
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

    it should "send data through the pipeline." in {
   		test(new ProcessingElementSequentialConv(lbirLayer, Options())).withAnnotations(Seq(VerilatorBackendAnnotation)) { c =>
            c.io.inStream.data.initSource()
            c.io.inStream.data.setSourceClock(c.clock)
            c.io.outStream.data.initSink()
            c.io.outStream.data.setSinkClock(c.clock)
            

            c.io.inStream.data.enqueue(3.U(32.W))
            c.io.inStream.data.enqueue(6.U(32.W))
            c.io.inStream.data.enqueue(9.U(32.W))
            c.io.inStream.last.poke(true.B)
            c.io.inStream.data.enqueue(12.U(32.W))
            c.io.inStream.last.poke(false.B)
            c.clock.step()
            c.clock.step()
            c.clock.step()
        } 
    }
    // test(new ProcessingPipeline(new lbirModel)) TODO
}

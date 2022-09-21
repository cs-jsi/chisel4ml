package chisel4ml.tests

import org.scalatest.flatspec.AnyFlatSpec
import chisel3._
import chiseltest._
import _root_.chisel4ml.ProcessingElementSequential
import _root_.lbir._


class PeSequentialTests extends AnyFlatSpec with ChiselScalatestTester {
    behavior of "ProcessingElementSequential"

    val lbirLayer = lbir.Layer(
        ltype = lbir.Layer.Type.DENSE,
        useBias = false,
        biases = Option(lbir.QTensor()),
        weights = Option(lbir.QTensor()),
        input = Option(lbir.QTensor()),
        output = Option(lbir.QTensor()),
        activation = Option(lbir.Activation())
    )
    it should "move data into SRAM" in {
        test(new ProcessingElementSequential(lbirLayer)).withAnnotations(Seq()) { c =>
            c.io.inStream.data.initSource()
            c.io.inStream.data.setSourceClock(c.clock)
            c.io.outStream.data.initSink()
            c.io.outStream.data.setSinkClock(c.clock)
            
            c.io.inStream.data.enqueue(32.U)
            c.io.inStream.data.enqueue(64.U)
            c.io.inStream.data.enqueue(4.U)
            c.io.inStream.data.enqueue(2.U)
            c.io.inStream.data.enqueue(3.U)
            c.io.inStream.data.enqueue(5.U)
            c.io.inStream.data.enqueue(7.U)
            c.io.inStream.last.poke(true.B)
            c.io.inStream.data.enqueue(128.U)
            c.io.inStream.last.poke(false.B)

            c.io.outStream.data.expectDequeue(32.U)
            c.io.outStream.data.expectDequeue(64.U)
            c.io.outStream.data.expectDequeue(4.U)
            c.io.outStream.data.expectDequeue(2.U)
            c.io.outStream.data.expectDequeue(3.U)
            c.io.outStream.data.expectDequeue(5.U)
            c.io.outStream.data.expectDequeue(7.U)
            c.io.outStream.data.expectDequeue(128.U)
        }
    }
    // test(new ProcessingPipeline(new lbirModel)) TODO
}

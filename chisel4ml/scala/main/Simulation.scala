/*
 * Handles the simulation of procesing pipelines
 *
 */
package chisel4ml

import _root_.chisel3._
import _root_.chisel3.util._
import _root_.chiseltest._
import _root_.java.util.concurrent.{BlockingQueue, LinkedBlockingQueue}
import _root_.chisel4ml.util.LbirUtil
import _root_.chisel4ml.ProcessingPipelineSimple
import _root_.lbir.{QTensor, Model}

import _root_.org.slf4j.Logger
import _root_.org.slf4j.LoggerFactory


class Simulation(model: Model, outTensorShape: QTensor, isSimple: Boolean) extends Runnable {
    val inQueue = new LinkedBlockingQueue[BigInt]()
    val outQueue = new LinkedBlockingQueue[UInt]()
    val logger = LoggerFactory.getLogger(classOf[Simulation])

    def run() : Unit = {
        RawTester.test(new ProcessingPipelineSimple(model))(this.run(_))
    }

    def run(dut: ProcessingPipelineSimple): Unit = {
        while(true) {
            // inQueue.take() blocks execution until data is available
            dut.io.in.poke(inQueue.take())
            dut.clock.step()
            outQueue.put(dut.io.out.peek())
        }
    }
    
    def sim(x: QTensor): QTensor = {
        inQueue.put(LbirUtil.qtensorToBigInt(x))
        LbirUtil.bigIntToQtensor(outQueue.take().litValue, outTensorShape) // .take() is a blocking call
    }
}

/*
 * Handles the simulation of procesing pipelines
 *
 */
package chisel4ml

import _root_.chisel3._
import _root_.chisel3.util._
import _root_.firrtl.options.TargetDirAnnotation
import _root_.chiseltest._
import _root_.java.util.concurrent.{BlockingQueue, LinkedBlockingQueue}
import _root_.chisel4ml.util.LbirUtil
import _root_.chisel4ml.ProcessingPipelineSimple
import _root_.lbir.{QTensor, Model}
import _root_.services.GenerateCircuitParams.Options
import _root_.java.util.concurrent.atomic.AtomicBoolean

import _root_.org.slf4j.Logger
import _root_.org.slf4j.LoggerFactory


class Circuit(model: Model, options: Options, directory: String) extends Runnable {
    val logger = LoggerFactory.getLogger(classOf[Circuit])
    val inQueue = new LinkedBlockingQueue[QTensor]()
    val outQueue = new LinkedBlockingQueue[QTensor]()
    val outTensorShape = model.layers.last.output.get
    val isSimple = options.isSimple
    val isGenerated = new AtomicBoolean(false)

    def run() : Unit = {
        if (isSimple) {
            RawTester.test(new ProcessingPipelineSimple(model),
                           Seq(WriteVcdAnnotation, 
                               TargetDirAnnotation(directory),
                               VerilatorBackendAnnotation))(this.runSimple(_))
        } else {
            RawTester.test(new ProcessingPipeline(model, options))(this.runSequential(_))
        }
    }

    private def runSimple(dut: ProcessingPipelineSimple): Unit = {
        isGenerated.set(true)
        while(true && isSimple) {
            // inQueue.take() blocks execution until data is available
            dut.io.in.poke(inQueue.take().toUInt)
            dut.clock.step()
            outQueue.put(dut.io.out.peek().toQTensor(outTensorShape))
        }
    }

    private def runSequential(dut: ProcessingPipeline): Unit = {
        val seqLength: Int = dut.io.inStream.dataWidth 
        val outBitsTotal: Int = model.layers.last.output.get.totalBitwidth
        val outTrans: Int = math.ceil(outBitsTotal.toFloat / dut.io.outStream.dataWidth.toFloat).toInt
        while(true && !isSimple) {
            dut.io.inStream.data.initSource()
            dut.io.inStream.data.setSourceClock(dut.clock)
            dut.io.outStream.data.initSource()
            dut.io.outStream.data.setSourceClock(dut.clock)
            dut.clock.step()

            // inQueue.take() blocks execution until data is available
            val testSeq: Seq[UInt] = LbirUtil.toUIntSeq(inQueue.take().toUInt, seqLength)
            var outSeq: Seq[UInt] = Seq()

            dut.io.inStream.data.enqueueSeq(testSeq)
            dut.io.inStream.last.poke(true.B)
            for (_ <- 0 until outTrans){
                dut.io.outStream.data.waitForValid()
                outSeq = outSeq :+ dut.io.outStream.data.bits
            }
            outQueue.put(LbirUtil.mergeUIntSeq(outSeq).toQTensor(outTensorShape))
        }
    }
    
    def sim(x: QTensor): QTensor = {
        inQueue.put(x)
        outQueue.take() // .take() is a blocking call
    }
}

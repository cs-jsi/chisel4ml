/*
 * Handles the simulation of procesing pipelines
 *
 */
package chisel4ml

import _root_.chisel3._
import _root_.chisel3.util._
import _root_.firrtl.options.TargetDirAnnotation
import _root_.firrtl.{AnnotationSeq, VerilogEmitter}
import _root_.firrtl.stage.{FirrtlStage, OutputFileAnnotation}
import _root_.chiseltest._


import _root_.chisel4ml.util.LbirUtil
import _root_.chisel4ml.ProcessingPipelineSimple
import _root_.lbir.{QTensor, Model}
import _root_.services.GenerateCircuitParams.Options

import _root_.java.util.concurrent.CountDownLatch
import _root_.java.util.concurrent.{BlockingQueue, LinkedBlockingQueue}
import _root_.org.slf4j.Logger
import _root_.org.slf4j.LoggerFactory


class Circuit(model: Model, options: Options, directory: String, useVerilator: Boolean, writeVcd: Boolean) 
extends Runnable {
    val logger = LoggerFactory.getLogger(classOf[Circuit])
    val inQueue = new LinkedBlockingQueue[QTensor]()
    val outQueue = new LinkedBlockingQueue[QTensor]()
    val outTensorShape = model.layers.last.output.get
    val isSimple = options.isSimple
    val isGenerated = new CountDownLatch(1)

    def run() : Unit = {
        var annot: AnnotationSeq = Seq(TargetDirAnnotation(directory)) // TODO - work with .pb instead of .lo.fir
        if (writeVcd) annot = annot :+ WriteVcdAnnotation
        if (useVerilator) annot = annot :+ VerilatorBackendAnnotation
        if (isSimple) {
            RawTester.test(new ProcessingPipelineSimple(model), annot)(this.runSimple(_))
        } else {
            RawTester.test(new ProcessingPipeline(model, options), annot)(this.runSequential(_))
        }
    }

    private def runSimple(dut: ProcessingPipelineSimple): Unit = {
        (new FirrtlStage).execute(Array("--input-file", s"$directory/ProcessingPipelineSimple.lo.fir",
                                        "--start-from", "low", "-E", "sverilog"), 
                                  Seq(TargetDirAnnotation(directory))
                         )
        isGenerated.countDown()
        logger.info(s"Generated simple circuit in directory: ${directory}.")
        while(true && isSimple) {
            // inQueue.take() blocks execution until data is available
            dut.io.in.poke(inQueue.take().toUInt)
            logger.info(s"Simulating a simple circuit on a new input.")
            dut.clock.step()
            outQueue.put(dut.io.out.peek().toQTensor(outTensorShape))
        }
    }

    private def runSequential(dut: ProcessingPipeline): Unit = {
        (new FirrtlStage).execute(Array("--input-file", s"$directory/ProcessingPipeline.lo.fir",
                                        "--start-from", "low", "-E", "sverilog"), 
                                  Seq(TargetDirAnnotation(directory))
                         )
        isGenerated.countDown() // Let the main thread now that the dut has been succesfully generated
        logger.info(s"Generated sequential circuit in directory: ${directory}.")
        val outBitsTotal: Int = model.layers.last.output.get.totalBitwidth
        val outTrans: Int = dut.peList.last.numOutTrans
        
        dut.io.inStream.data.initSource()
        dut.io.inStream.data.setSourceClock(dut.clock)
        dut.io.outStream.data.initSink()
        dut.io.outStream.data.setSinkClock(dut.clock)
        while(true && !isSimple) {
            // inQueue.take() blocks execution until data is available
            val testSeq: Seq[UInt] = inQueue.take().toUInt.toUIntSeq(dut.io.inStream.dataWidth)
            logger.info(s"Simulating a sequential circuit on a new input.")
            var outSeq: Seq[BigInt] = Seq()

            dut.io.inStream.data.enqueueSeq(testSeq)
            dut.io.inStream.last.poke(true.B)
            dut.clock.step()
            dut.io.inStream.last.poke(false.B)
            dut.io.outStream.data.ready.poke(true.B)
            for (_ <- 0 until outTrans){
                dut.io.outStream.data.waitForValid()
                outSeq = outSeq :+ dut.io.outStream.data.bits.peek().litValue
            }
            outQueue.put(outSeq.toUInt(dut.io.outStream.dataWidth).toQTensor(outTensorShape))
        }
    }
    
    def sim(x: QTensor): QTensor = {
        inQueue.put(x)
        outQueue.take() // .take() is a blocking call
    }
}

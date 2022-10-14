/*
 * Copyright 2022 Computer Systems Department, Jozef Stefan Insitute
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *    https://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
package chisel4ml

import _root_.chisel3._
import _root_.chisel3.util._
import _root_.firrtl.options.TargetDirAnnotation
import _root_.firrtl.{AnnotationSeq, VerilogEmitter}
import _root_.firrtl.stage.{FirrtlStage, OutputFileAnnotation}
import _root_.chiseltest._
import _root_.chiseltest.simulator.WriteVcdAnnotation

import _root_.chisel4ml.util.LbirUtil
import _root_.chisel4ml.ProcessingPipelineSimple
import _root_.lbir.{QTensor, Model}
import _root_.services.GenerateCircuitParams.Options

import _root_.scala.util.control.Breaks._
import _root_.java.nio.file.{Path, Paths}
import _root_.java.util.concurrent.TimeUnit
import _root_.java.util.concurrent.CountDownLatch
import _root_.java.util.concurrent.{BlockingQueue, LinkedBlockingQueue}
import _root_.org.slf4j.Logger
import _root_.org.slf4j.LoggerFactory


class Circuit(model: Model, options: Options, directory: Path, useVerilator: Boolean, genVcd: Boolean) 
extends Runnable {
    case class ValidQTensor(qtensor: QTensor, valid: Boolean)
    val logger = LoggerFactory.getLogger(classOf[Circuit])
    val inQueue = new LinkedBlockingQueue[ValidQTensor]()
    val outQueue = new LinkedBlockingQueue[QTensor]()
    val outTensorShape = model.layers.last.output.get
    val isSimple = options.isSimple
    val isGenerated = new CountDownLatch(1)
    val isStoped = new CountDownLatch(1)
    val relDir = Paths.get("").toAbsolutePath().relativize(directory).toString
    LbirUtil.setDirectory(directory)
    
    var annot: AnnotationSeq = Seq(TargetDirAnnotation(relDir)) // TODO - work with .pb instead of .lo.fir
    if (genVcd) annot = annot :+ WriteVcdAnnotation
    if (useVerilator) annot = annot :+ VerilatorBackendAnnotation
    
    def stopSimulation(): Unit = {
        inQueue.put(ValidQTensor(QTensor(), false))
        isStoped.await(5, TimeUnit.SECONDS)
    }

    def run() : Unit = {
        logger.info(s"Used annotations for generated circuit are: ${annot.map(_.toString)}.")
        if (isSimple) {
            RawTester.test(new ProcessingPipelineSimple(model), annot)(this.runSimple(_))
        } else {
            RawTester.test(new ProcessingPipeline(model, options), annot)(this.runSequential(_))
        }
        isStoped.countDown()
    }

    private def runSimple(dut: ProcessingPipelineSimple): Unit = {
        if (!useVerilator) {
            (new FirrtlStage).execute(Array("--input-file", s"$relDir/ProcessingPipelineSimple.lo.fir",
                                            "--start-from", "low", "-E", "sverilog"), annot)
        }
        isGenerated.countDown()
        logger.info(s"Generated simple circuit in directory: ${directory}.")
        while(true && isSimple) {
            // inQueue.take() blocks execution until data is available
            val validQTensor = inQueue.take()
            if (validQTensor.valid == false) break()
            dut.io.in.poke(validQTensor.qtensor.toUInt)
            logger.info(s"Simulating a simple circuit on a new input.")
            dut.clock.step()
            outQueue.put(dut.io.out.peek().toQTensor(outTensorShape))
        }
    }

    private def runSequential(dut: ProcessingPipeline): Unit = {
        if (!useVerilator) {
            (new FirrtlStage).execute(Array("--input-file", s"$relDir/ProcessingPipeline.lo.fir",
                                            "--start-from", "low", "-E", "sverilog"), annot)
        }
        isGenerated.countDown() // Let the main thread now that the dut has been succesfully generated
        logger.info(s"Generated sequential circuit in directory: ${directory}.")
        val outBitsTotal: Int = model.layers.last.output.get.totalBitwidth
        val outTrans: Int = dut.peList.last.numOutTrans
        
        dut.io.inStream.data.initSource()
        dut.io.inStream.data.setSourceClock(dut.clock)
        dut.io.outStream.data.initSink()
        dut.io.outStream.data.setSinkClock(dut.clock)
        breakable { while(true && !isSimple) {
            // inQueue.take() blocks execution until data is available
            val validQTensor = inQueue.take()
            if (validQTensor.valid == false) break()
            val testSeq: Seq[UInt] = validQTensor.qtensor.toUInt.toUIntSeq(dut.io.inStream.dataWidth)
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
    }
    
    def sim(x: QTensor): QTensor = {
        inQueue.put(ValidQTensor(x, true))
        outQueue.take() // .take() is a blocking call
    }
}

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
import _root_.chisel4ml.implicits._
import _root_.chisel4ml._
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

class Circuit[+T <: Module with LBIRStream](dutGen: => T,
                                            directory: Path,
                                            useVerilator: Boolean,
                                            genVcd: Boolean)
extends Runnable {
    case class ValidQTensor(qtensor: QTensor, valid: Boolean)
    val logger = LoggerFactory.getLogger(classOf[Circuit[T]])
    val inQueue = new LinkedBlockingQueue[ValidQTensor]()
    val outQueue = new LinkedBlockingQueue[QTensor]()
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
        RawTester.test(dutGen, annot)(this.simulate(_))
        isStoped.countDown()
    }

    private[this] def simulate(dut: T): Unit = {
        isGenerated.countDown()
        logger.info(s"Generated circuit in directory: ${directory}.")
        dut.inStream.initSource()
        dut.outStream.initSink()
        breakable {
        while(true) {
            // inQueue.take() blocks execution until data is available
            val validQTensor = inQueue.take()
            if (validQTensor.valid == false) break()
            logger.info(s"Simulating a sequential circuit on a new input.")
            var outSeq: Seq[BigInt] = Seq()
            fork {
                dut.inStream.enqueueQTensor(validQTensor.qtensor, dut.clock)
            }.fork {
                outQueue.put(dut.outStream.dequeueQTensor(dut.clock))
            }.join()
         }
        }
    }

    def sim(x: Seq[QTensor]): Seq[QTensor] = {
        var result: Seq[QTensor] = Seq()
        for (qtensor <- x) {
            inQueue.put(ValidQTensor(qtensor, true))
        }
        for (_ <- x) {
            result = result :+ outQueue.take() // .take() is a blocking call
        }
        result
    }
}

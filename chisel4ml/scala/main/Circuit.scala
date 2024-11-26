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

import chisel3._
import chisel4ml._
import chisel4ml.implicits._
import chiseltest._
import chiseltest.simulator.{WriteFstAnnotation, WriteVcdAnnotation}
import firrtl2.AnnotationSeq
import firrtl2.options.TargetDirAnnotation
import firrtl2.transforms.NoCircuitDedupAnnotation
import lbir.QTensor
import memories.MemoryGenerator
import org.slf4j.LoggerFactory

import java.util.concurrent.{CountDownLatch, LinkedBlockingQueue, TimeUnit}
import scala.util.control.Breaks._

/** Contains the generated hardware module and provides a simulation interface
  *
  * @param dutGen
  *   The generated hardware container
  * @param outputStencil
  *   The expected output QTensor shape and datatype
  * @param directory
  *   The directory that holds the generated FIRRTL/Verilog files
  * @param useVerilator
  *   Use verilator for simulation?
  * @param genWaveform
  *   Generate waveform during RTL simulation?
  * @param waveformType
  *   The type of waveform to generate during RTL simulation. Either 'fst' or 'vcd'.
  */
class Circuit[+T <: Module with HasAXIStream](
  dutGen:        => T,
  outputStencil: QTensor,
  directory:     os.Path,
  useVerilator:  Boolean,
  genWaveform:   Boolean,
  waveformType:  String = "fst")
    extends Runnable {
  case class ValidQTensor(qtensor: QTensor, valid: Boolean)
  case class TimedQTensor(qtensor: QTensor, consumedCycles: Int)
  val logger = LoggerFactory.getLogger(classOf[Circuit[T]])
  val inQueue = new LinkedBlockingQueue[ValidQTensor]()
  val outQueue = new LinkedBlockingQueue[TimedQTensor]()
  val isGenerated = new CountDownLatch(1)
  val isStoped = new CountDownLatch(1)
  val relativeDirectory = directory.relativeTo(os.pwd).toString()

  // NoCircuitDedupAnnotation is needed because memory deduplication is causing problems
  // This can likely be removed when upgrading to newer chisel/firrtl versions. TODO
  var annot: AnnotationSeq =
    Seq(TargetDirAnnotation(relativeDirectory), NoCircuitDedupAnnotation) // TODO - work with .pb instead of .lo.fir
  if (genWaveform) {
    if (waveformType.toLowerCase == "vcd") {
      annot = annot :+ WriteVcdAnnotation
    } else {
      annot = annot :+ WriteFstAnnotation
    }

  }
  if (useVerilator) annot = annot :+ VerilatorBackendAnnotation

  def stopSimulation(): Unit = {
    inQueue.put(ValidQTensor(QTensor(), false))
    isStoped.await(5, TimeUnit.SECONDS)
  }

  def run(): Unit = {
    logger.info(s"Used annotations for generated circuit are: ${annot.map(_.toString)}.")
    MemoryGenerator.setGenDir(directory)
    RawTester.test(dutGen, annot)(this.simulate(_))
    isStoped.countDown()
  }

  /** Simulates a circuit using an input and output queue of QTensors.
    *
    * @param dut
    *   the hardware module to simulate.
    */
  private[this] def simulate(dut: T): Unit = {
    isGenerated.countDown()
    logger.info(s"Generated circuit in directory: ${directory}.")
    dut.inStream.initSource()
    dut.outStream.initSink()
    dut.clock.setTimeout(50000) // TODO
    breakable {
      while (true) {
        // inQueue.take() blocks execution until data is available
        val validQTensor = inQueue.take()
        if (validQTensor.valid == false) break()

        logger.info(
          s"Simulating a sequential circuit on a new input. Input shape: ${validQTensor.qtensor.shape}" +
            s", input dtype: ${validQTensor.qtensor.dtype}, output stencil: $outputStencil."
        )
        val isOver = new CountDownLatch(1)
        var clockCounter = 0
        fork {
          dut.inStream.enqueueQTensor(validQTensor.qtensor, dut.clock)
          breakable {
            while (true) {
              dut.clock.step()
              clockCounter = clockCounter + 1
              if (isOver.getCount() == 0) break()
            }
          }
        }.fork {
          val qtensor = dut.outStream.dequeueQTensor(outputStencil, dut.clock)
          isOver.countDown()
          outQueue.put(TimedQTensor(qtensor, clockCounter))
        }.join()
      }
    }
  }

  /** The user interface to the QTensor based simulation
    *
    * @param x
    *   A sequence of input QTensors to simulate
    * @return
    *   Resulting QTensors and the number of cycles used to simulate it.
    */
  def sim(x: Seq[QTensor]): (Seq[QTensor], Int) = {
    var result:         Seq[QTensor] = Seq()
    var consumedCycles: Int = 0
    for (qtensor <- x) {
      inQueue.put(ValidQTensor(qtensor, true))
    }
    for (_ <- x) {
      val out = outQueue.take()
      result = result :+ out.qtensor // .take() is a blocking call
      consumedCycles = consumedCycles + out.consumedCycles
    }
    (result, consumedCycles)
  }
}

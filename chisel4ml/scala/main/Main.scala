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
import chisel4ml.Circuit
import io.grpc.{Server, ServerBuilder}
import java.io.{File, RandomAccessFile}
import java.nio.channels.{FileChannel, FileLock}
import java.nio.file.{Files, Paths}
import java.util.concurrent.TimeUnit
import org.slf4j.LoggerFactory
import scala.concurrent.{ExecutionContext, Future}
import services.GenerateCircuitReturn.ErrorMsg
import services._

/** Contains the main function.
  *
  *  Contains the main function that is the main entry point to the the software, and it starts a chisel4ml
  *  server instance.
  */
object Chisel4mlServer {
  private val port = 50051
  var f:          File = _
  var fRwChannel: FileChannel = _
  var lock:       FileLock = _

  def main(args: Array[String]): Unit = {
    require(args.length > 0, "No argument list, you should provide an argument as a directory.")
    require(Files.exists(Paths.get(args(0))), "Provided directory doesn't exist.")
    val tempDir = args(0)
    val lockFile = Paths.get(tempDir, ".lockfile")
    // We use lockfiles to ensure only one instance of a chisel4ml server is running.
    f = new File(lockFile.toString)
    if (f.exists()) {
      f.delete() // We try deleting it, if it exists
    }
    fRwChannel = new RandomAccessFile(f, "rw").getChannel()
    lock = fRwChannel.tryLock()
    if (lock == null) {
      // Lock occupied by other instance
      fRwChannel.close()
      throw new RuntimeException("Only one instance of chisel4ml server may run at a time.")
    }
    // We add a shutdown hook to release the lock on shutdown
    sys.addShutdownHook { closeFileLockHook() }
    val server = new Chisel4mlServer(ExecutionContext.global, tempDir = tempDir)
    server.start()
    server.blockUntilShutdown()
  }

  private def closeFileLockHook(): Unit = {
    lock.release()
    fRwChannel.close()
    f.delete()
  }
}

/** The server implementation based on gRPC.
  *
  *  Implementation of the gRPC based Chisel4ml server.  It implements the services as defined by gRPC in the
  *  service.proto file. It also has conveinance functions for starting and stoping the server.
  */
class Chisel4mlServer(executionContext: ExecutionContext, tempDir: String) { self =>
  private[this] var server: Server = null
  private var circuits = Map[Int, Circuit[Module with LBIRStream]]() // Holds the circuit and simulation object
  private var nextId: Int = 0

  val logger = LoggerFactory.getLogger(classOf[Chisel4mlServer])

  private def start(): Unit = {
    server = ServerBuilder
      .forPort(Chisel4mlServer.port)
      .maxInboundMessageSize(Math.pow(2, 26).toInt)
      .addService(Chisel4mlServiceGrpc.bindService(Chisel4mlServiceImpl, executionContext))
      .build
      .start
    sys.addShutdownHook { self.stop() }
    logger.info("Started a new chisel4ml server.")
  }

  private def stop(): Unit = {
    if (server != null) {
      // we stop all simulations properly to get vcd files
      circuits.map(_._2.stopSimulation())
      logger.info("Shutting down chisel4ml server.")
      server.shutdown()
    } else { logger.error("Attempted to shut down server that was not created.") }
  }

  private def blockUntilShutdown(): Unit = { if (server != null) { server.awaitTermination() } }

  private object Chisel4mlServiceImpl extends Chisel4mlServiceGrpc.Chisel4mlService {
    override def generateCircuit(params: GenerateCircuitParams): Future[GenerateCircuitReturn] = {
      val circuit = new Circuit[ProcessingPipeline](
        dutGen = new ProcessingPipeline(params.model.get, params.options.get),
        outputStencil = params.model.get.layers.last.get.output,
        directory = Paths.get(tempDir, s"circuit$nextId"),
        useVerilator = params.useVerilator,
        genWaveform = params.genWaveform
      )
      circuits = circuits + (nextId -> circuit)
      logger.info(s"""Started generating hardware for circuit id:$nextId in temporary directory $tempDir
                     | with a timeout of ${params.generationTimeoutSec} seconds.""".stripMargin.replaceAll("\n", ""))
      nextId = nextId + 1
      new Thread(circuit).start()
      if (circuit.isGenerated.await(params.generationTimeoutSec, TimeUnit.SECONDS)) {
        logger.info("Succesfully generated circuit.")
        Future.successful(
          GenerateCircuitReturn(
            circuitId = nextId - 1,
            err = Option(ErrorMsg(errId = ErrorMsg.ErrorId.SUCCESS, msg = "Successfully generated verilog."))
          )
        )
      } else {
        logger.error("Circuit generation timed-out, please try again with a longer timeout.")
        Future.successful(
          GenerateCircuitReturn(
            circuitId = nextId - 1,
            err = Option(ErrorMsg(errId = ErrorMsg.ErrorId.FAIL, msg = "Error generating circuit."))
          )
        )
      }
    }

    override def runSimulation(params: RunSimulationParams): Future[RunSimulationReturn] = {
      logger.info(s"Simulating circuit id: ${params.circuitId} circuit on ${params.inputs.length} input/s.")
      Future.successful(
        RunSimulationReturn(values = circuits(params.circuitId).sim(params.inputs))
      )
    }

    override def deleteCircuit(params: DeleteCircuitParams): Future[DeleteCircuitReturn] = {
      logger.info(s"Deleting cirucit id: ${params.circuitId} from memory.")
      val contained = circuits.contains(params.circuitId)
      if (contained)
        circuits = circuits - params.circuitId
      Future.successful(
        DeleteCircuitReturn(
          success = contained,
          msg =
            if (contained) s"Succesfully deleted circuit id: ${params.circuitId}"
            else s"Circuit id ${params.circuitId} not present."
        )
      )
    }
  }
}

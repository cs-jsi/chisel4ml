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

import _root_.chisel4ml.Circuit
import _root_.io.grpc.{Server, ServerBuilder}
import _root_.java.io.{File, IOException, RandomAccessFile}
import _root_.java.nio.channels.{FileChannel, FileLock}
import _root_.java.nio.file.{Files, Paths}
import _root_.java.util.concurrent.TimeUnit
import _root_.org.slf4j.LoggerFactory
import _root_.scala.concurrent.{ExecutionContext, Future}
import _root_.services.GenerateCircuitReturn.ErrorMsg
import _root_.services._

/** Contains the main function.
  *
  * Contains the main function that is the main entry point to the the software, and it starts a chisel4ml server
  * instance.
  */
object Chisel4mlServer {
  private val port = 50051
  var f:          File        = _
  var fRwChannel: FileChannel = _
  var lock:       FileLock    = _

  def main(args: Array[String]): Unit = {
    require(args.length > 0, "No argument list, you should provide an argument as a directory.")
    require(Files.exists(Paths.get(args(0))), "Provided directory doesn't exist.")
    val tempDir  = args(0)
    val lockFile = Paths.get(tempDir, ".lockfile")
    // We use lockfiles to ensure only one instance of a chisel4ml server is running.
    try {
      f = new File(lockFile.toString)
      if (f.exists()) {
        f.delete() // We try deleting it, if it exists
      }
      fRwChannel = new RandomAccessFile(f, "rw").getChannel()
      lock = fRwChannel.tryLock()
    } catch {
      case e: IOException =>
        throw new RuntimeException("Could not aquire lock to start a new instace of server.", e)
    }
    if (lock == null) {
      // Lock occupied by other instance
      fRwChannel.close()
      throw new RuntimeException("Only one instance of chisel4ml server may run at a time.")
    }

    // We add a shutdown hook to release the lock on shutdown
    sys.addShutdownHook(closeFileLockHook())
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
  * Implementation of the gRPC based Chisel4ml server. It implements the services as defined by gRPC in the
  * service.proto file. It also has conveinance functions for starting and stoping the server.
  */
class Chisel4mlServer(executionContext: ExecutionContext, tempDir: String) { self =>
  private[this] var server: Server       = null
  private var circuits:     Seq[Circuit] = Seq() // Holds the circuit and simulation object
  val logger = LoggerFactory.getLogger(classOf[Chisel4mlServer])

  private def start(): Unit = {
    server = ServerBuilder
      .forPort(Chisel4mlServer.port)
      .addService(Chisel4mlServiceGrpc.bindService(Chisel4mlServiceImpl, executionContext))
      .build
      .start
    sys.addShutdownHook(self.stop())
    logger.info("Started a new chisel4ml server.")
  }

  private def stop(): Unit =
    if (server != null) {
      // we stop all simulations properly to get vcd files
      circuits.map(_.stopSimulation())
      logger.info("Shutting down chisel4ml server.")
      server.shutdown()
    } else { logger.error("Attempted to shut down server that was not created.") }

  private def blockUntilShutdown(): Unit = if (server != null) { server.awaitTermination() }

  private object Chisel4mlServiceImpl extends Chisel4mlServiceGrpc.Chisel4mlService {

    override def shutdownServer(params: ShutdownServerParams): Future[ShutdownServerReturn] = {
      server.shutdownNow()
      return Future.successful(ShutdownServerReturn())
    }

    override def generateCircuit(params: GenerateCircuitParams): Future[GenerateCircuitReturn] = {
      val circuitId = circuits.length
      circuits = circuits :+ new Circuit(
        model = params.model.get,
        options = params.options.get,
        directory = Paths.get(tempDir, s"circuit$circuitId"),
        useVerilator = params.useVerilator,
        genVcd = params.genVcd,
      )
      logger.info(
        s"""Started generating hardware for circuit id:$circuitId in temporary directory $tempDir
           | with a timeout of ${params.generationTimeoutSec} seconds.""".stripMargin.replaceAll("\n", ""),
      )
      new Thread(circuits.last).start()
      if (circuits.last.isGenerated.await(params.generationTimeoutSec, TimeUnit.SECONDS)) {
        logger.info("Succesfully generated circuit.")
        Future.successful(
          GenerateCircuitReturn(
            circuitId = circuits.length - 1,
            err = Option(ErrorMsg(errId = ErrorMsg.ErrorId.SUCCESS, msg = "Successfully generated verilog.")),
          ),
        )
      } else {
        logger.error("Circuit generation timed-out, please try again with a longer timeout.")
        Future.successful(
          GenerateCircuitReturn(
            circuitId = circuits.length - 1,
            err = Option(ErrorMsg(errId = ErrorMsg.ErrorId.FAIL, msg = "Error generating circuit.")),
          ),
        )
      }
    }

    override def runSimulation(params: RunSimulationParams): Future[RunSimulationReturn] = {
      logger.info(s"Simulating circuit id: ${params.circuitId} circuit on ${params.inputs.length} input/s.")
      try {
        val simRes = circuits(params.circuitId).sim(params.inputs(0))
        Future.successful(
          RunSimulationReturn(values = Seq(simRes)),
        )
      } catch {
        case e => {
          self.stop()
          throw e
          Future.successful(
            RunSimulationReturn(values = Seq()),
          )
        }
      }
    }
  }
}

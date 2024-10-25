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
import lbir.QTensor
import org.slf4j.LoggerFactory
import scopt.OParser
import services.GenerateCircuitReturn.ErrorMsg
import services._

import java.util.concurrent.TimeUnit
import scala.concurrent.{ExecutionContext, Future}
import scala.io.Source

case class Config(
  tempDir: os.Path = os.Path("/tmp/.chisel4ml/"),
  port:    Int = 50051)

/** Contains the main function.
  *
  *  Contains the main function that is the main entry point to the the software, and it starts a chisel4ml
  *  server instance.
  */
object Chisel4mlServer {
  // we convert git describe output to pep440
  private val chisel4mlVersion = {
    val versionRegex = raw"(\d+)\.(\d+)\.(\d+)-?(\d+)?-?(\w+)?".r
    val gitDescribe = Source.fromResource("versionInfo/gitInfo").mkString.stripLineEnd
    gitDescribe match {
      case versionRegex(major, minor, patch, null, null) => s"$major.$minor.$patch"
      case versionRegex(major, minor, patch, revision, gitTag) =>
        s"$major.$minor.${patch.toInt + 1}.dev$revision+$gitTag"
      case _ => throw new Exception(s"Parse error on git describe string: $gitDescribe")
    }
  }
  private var server: Chisel4mlServer = _

  val builder = OParser.builder[Config]
  val cliParser = {
    import builder._
    OParser.sequence(
      programName("chisel4ml-server"),
      head("chisel4ml-server;", s"v$chisel4mlVersion"),
      opt[Int]('p', "port")
        .action((x, c) => c.copy(port = x))
        .text("Which port should the chisel4ml-server use (default: 50051)."),
      opt[String]('d', "dir")
        .action((x, c) => c.copy(tempDir = os.Path(x)))
        .text("Which directory should chisel4ml-server use as its temporary directory (default: /tmp/.chisel4ml/)."),
      help("help").text("Prints this usage text.")
    )
  }

  def main(args: Array[String]): Unit = {
    OParser.parse(cliParser, args, Config()) match {
      case Some(config) => {
        if (!os.exists(config.tempDir)) {
          os.makeDir(config.tempDir, "rwxrwxrw-")
        }
        if (os.list(config.tempDir).length != 0) {
          throw new Exception(s"Directory ${config.tempDir} is not empty.")
        }
        server = new Chisel4mlServer(ExecutionContext.global, tempDir = config.tempDir, port = config.port)
        server.start()
        server.blockUntilShutdown()
      }
      case _ =>
    }
  }
}

/** The server implementation based on gRPC.
  *
  *  Implementation of the gRPC based Chisel4ml server.  It implements the services as defined by gRPC in the
  *  service.proto file. It also has conveinance functions for starting and stoping the server.
  */
class Chisel4mlServer(executionContext: ExecutionContext, tempDir: os.Path, port: Int) { self =>
  private[this] var server: Server = null
  private var circuits =
    Map[Int, Circuit[Module with HasAXIStream]]() // Holds the circuit and simulation object
  private var nextId: Int = 0
  val logger = LoggerFactory.getLogger(classOf[Chisel4mlServer])

  private def start(): Unit = {
    server = ServerBuilder
      .forPort(port)
      .maxInboundMessageSize(Math.pow(2, 26).toInt)
      .addService(Chisel4mlServiceGrpc.bindService(Chisel4mlServiceImpl, executionContext))
      .build
      .start
    sys.addShutdownHook {
      if (server != null) {
        // we stop all simulations properly to get valid vcd files
        circuits.map(_._2.stopSimulation())
        logger.info("Shutting down chisel4ml server.")
        server.shutdown()
      } else {
        logger.error("Attempted to shut down server that was not created.")
      }
    }
    logger.info(s"Started a new chisel4ml-server on port $port, using temporary directory: $tempDir.")
  }

  private def blockUntilShutdown(): Unit = { if (server != null) { server.awaitTermination() } }

  private object Chisel4mlServiceImpl extends Chisel4mlServiceGrpc.Chisel4mlService {
    override def generateCircuit(params: GenerateCircuitParams): Future[GenerateCircuitReturn] = {
      val circuit = new Circuit[Module with HasAXIStream](
        dutGen = new ProcessingPipeline(params.accelerators),
        outputStencil = params.accelerators.last.layers.last.get.output,
        directory = tempDir / s"circuit$nextId",
        useVerilator = params.useVerilator,
        genWaveform = params.genWaveform,
        waveformType = params.waveformType
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
      val res: (Seq[QTensor], Int) = circuits(params.circuitId).sim(params.inputs)
      Future.successful(
        RunSimulationReturn(
          values = res._1,
          consumedCycles = res._2
        )
      )
    }

    override def deleteCircuit(params: DeleteCircuitParams): Future[DeleteCircuitReturn] = {
      logger.info(s"Deleting cirucit id: ${params.circuitId} from memory.")
      val contained = circuits.contains(params.circuitId)
      if (contained)
        circuits(params.circuitId).stopSimulation()
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

    override def getVersion(params: GetVersionParams): Future[GetVersionReturn] = {
      Future.successful(
        GetVersionReturn(
          version = Chisel4mlServer.chisel4mlVersion
        )
      )
    }
  }
}

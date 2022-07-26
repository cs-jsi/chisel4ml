/*
 * HEADER: TODO
 *
 */
package chisel4ml

import _root_.java.nio.file.{Files, Paths}
import _root_.scala.concurrent.{ExecutionContext, Future}

import _root_.chisel3.stage._
import _root_.chisel3._

import _root_.io.grpc.{Server, ServerBuilder}
import _root_.services.{ErrorMsg, GenerateParams, PpElaborateReturn, PpHandle, PpRunParams, PpRunReturn, PpServiceGrpc}
import _root_.lbir.{Datatype, Model, QTensor}
import _root_.chisel4ml.util.LbirUtil

import _root_.treadle.TreadleTester
import _root_.org.slf4j.Logger
import _root_.org.slf4j.LoggerFactory

/** An object extending App to generate the Verilog code.
  */
object Chisel4mlServer {
    private val port = 50051

    def main(args: Array[String]): Unit = {
        val server = new Chisel4mlServer(ExecutionContext.global)
        server.start()
        server.blockUntilShutdown()
    }
}

class Chisel4mlServer(executionContext: ExecutionContext) {
    self =>
    private[this] var server: Server = null
    val logger = LoggerFactory.getLogger(classOf[Chisel4mlServer])

    private def start(): Unit = {
        server = ServerBuilder
            .forPort(Chisel4mlServer.port)
            .addService(PpServiceGrpc.bindService(new PpServiceImpl, executionContext))
            .build
            .start
        sys.addShutdownHook { self.stop() }
        logger.info("Started a new chisel4ml server.")
    }

    private def stop(): Unit = {
        if (server != null) {
            logger.info("Shutting down chisel4ml server.")
            server.shutdown()
        } else { logger.error("Attempted to shut down server that was not created.") }
    }

    private def blockUntilShutdown(): Unit = { if (server != null) { server.awaitTermination() } }


    private object PpServiceImpl {
        var model:  Model         = null
        var tester: TreadleTester = null
    }

    private class PpServiceImpl extends PpServiceGrpc.PpService {
        private[this] var model:  Model         = null
        private[this] var tester: TreadleTester = null

        override def elaborate(lbirModel: Model): Future[PpElaborateReturn] = {
            logger.info("Elaborating model: " + lbirModel.name + " to a processing pipeline circuit.")
            model = lbirModel
            tester = TreadleTester(
              (new ChiselStage)
                  .execute(Array(), Seq(ChiselGeneratorAnnotation(() => new ProcessingPipeline(lbirModel))))
            )

            val errReply          = ErrorMsg(err = ErrorMsg.ErrorId.SUCCESS, msg = "Everything went fine.")
            val ppHandle          =
                PpHandle(name = "model", input = lbirModel.layers(0).input, outShape = lbirModel.layers.last.output.get.shape)
            val ppElaborateReturn = PpElaborateReturn(ppHandle = Option(ppHandle), reply = Some(errReply))
            Future.successful(ppElaborateReturn)
        }

        override def run(ppRunParams: PpRunParams): Future[PpRunReturn] = {
            logger.info("Simulating processing pipeline: " + ppRunParams.ppHandle.get.name + " circuit on inputs.")
            tester.poke("io_in", LbirUtil.qtensorToBigInt(ppRunParams.inputs(0)))
            Future.successful(
              PpRunReturn(values = List(LbirUtil.bigIntToQtensor(tester.peek("io_out"), model.layers.last.output.get.shape(0))))
            )
        }

        override def generate(genParams: GenerateParams): Future[ErrorMsg] = {
            logger.info(
              "Generating hardware for circuit id: " + genParams.name + " in directory: " + genParams.directory + "."
            )
            (new ChiselStage).execute(
              Array("--target-dir", genParams.directory),
              Seq(ChiselGeneratorAnnotation(() => new ProcessingPipeline(model)))
            )
            Future.successful(ErrorMsg(err = ErrorMsg.ErrorId.SUCCESS, msg = "Successfully generated verilog"))
        }
    }
}

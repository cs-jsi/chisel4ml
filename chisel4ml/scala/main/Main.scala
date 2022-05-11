/*
 * HEADER: TODO
 *
 */
package chisel4ml

import java.nio.file.{Files, Paths}
import chisel3.stage._
import chisel3._

import io.grpc.{Server, ServerBuilder}
import services.{ModelServiceGrpc, ModelRunReturn, ModelRunParams, ModelHandle, ErrorMsg}
import scala.concurrent.{ExecutionContext, Future}
import treadle.TreadleTester

/**
 * An object extending App to generate the Verilog code.
 */
object Chisel4mlServer {
    private val port = 50051

    def main(args: Array[String]): Unit = {
        val server = new Chisel4mlServer(ExecutionContext.global)
        server.start()
        server.blockUntilShutdown()
    }
}

class Chisel4mlServer(executionContext: ExecutionContext) { self =>
    private[this] var server: Server = null

    private def start(): Unit = {
      server = ServerBuilder.forPort(Chisel4mlServer.port).addService(
            ModelServiceGrpc.bindService(new ModelServiceImpl, executionContext)).build.start
      sys.addShutdownHook {
        self.stop()
      }
    }

    private def stop(): Unit = {
      if (server != null) {
        server.shutdown()
      }
    }

    private def blockUntilShutdown(): Unit = {
      if (server != null) {
        server.awaitTermination()
      }
    }

    private object ModelServiceImpl {
        var processingPipeline: ProcessingPipeline = null
        var tester: TreadleTester = null
    }

    private class ModelServiceImpl extends ModelServiceGrpc.ModelService {
        private[this] var processingPipeline: ProcessingPipeline = null
        private[this] var tester: TreadleTester = null
        override def compile(model: lbir.Model): Future[ModelHandle] = {
            processingPipeline = new ProcessingPipeline(model)
            tester = TreadleTester((new ChiselStage).execute(Array(), 
                                    Seq(ChiselGeneratorAnnotation(() => processingPipeline))))
            val errReply = ErrorMsg(err = ErrorMsg.ErrorId.SUCCESS, msg="Everything went fine.")
            val modelHandle = ModelHandle(name = "model", directory = "./gen/", reply = Some(errReply))
            Future.successful(modelHandle)
        }

        override def run(modelRunParams: ModelRunParams): Future[ModelRunReturn] = {
            for (input <- modelRunParams.inputs) {
                tester.poke("io_in", 15)
            }
            Future.successful(ModelRunReturn(values = List(lbir.QTensor(values = List(12.0F)))))
        }
    }
}

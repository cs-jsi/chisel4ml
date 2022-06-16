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
import _root_.services.{PpServiceGrpc, PpRunReturn, PpRunParams, PpElaborateReturn, PpHandle, ErrorMsg}
import _root_.lbir.{Model, QTensor, Datatype}

import _root_.treadle.TreadleTester
import _root_.org.slf4j.Logger
import _root_.org.slf4j.LoggerFactory


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
    val logger = LoggerFactory.getLogger(classOf[Chisel4mlServer])

    private def start(): Unit = {
      server = ServerBuilder.forPort(Chisel4mlServer.port).addService(
            PpServiceGrpc.bindService(new PpServiceImpl, executionContext)).build.start
      sys.addShutdownHook {
        self.stop()
      }
      logger.info("Started a new chisel4ml server.")
    }

    private def stop(): Unit = {
      if (server != null) {
        logger.info("Shutting down chisel4ml server.")
        server.shutdown()
      }
      else {
        logger.error("Attempted to shut down server that was not created.")
      }
    }

    private def blockUntilShutdown(): Unit = {
      if (server != null) {
        server.awaitTermination()
      }
    }

    // TODO move this function somewhere else
    def lbirToBigInt(qtensor: QTensor): BigInt = {
        val values = qtensor.values.reverse
        val new_vals = values.map(x => (x + 1) / 2) // 1 -> 1, -1 -> 0
        val big_int = BigInt(new_vals.map(x => x.toInt).mkString, radix = 2)
        logger.info("Converted lbir.QTensor: " + values + " to BigInt: " + big_int + ".")
        big_int
    }

    def bigIntToLbir(value: BigInt): QTensor = {
        val dataType = Datatype(quantization=Datatype.QuantizationType.BINARY,
                                bitwidth=1,
                                scale=1,
                                offset=0)
        // We substract the 48 because x is an ASCII encoded symbol
        val lbir_values = value.toString().toList.map(x=>x.toFloat-48).reverse.map(x => (x * 2) -1) // 1 -> 1, 0 -> -1
        val qtensor = QTensor(dtype=Option(dataType), shape = List(value.bitCount), values=lbir_values)
        logger.info("Converted BigInt: " + value + " to lbir.QTensor: " + qtensor + ".")
        qtensor
    }

    private object PpServiceImpl {
        var processingPipeline: ProcessingPipeline = null
        var tester: TreadleTester = null
    }

    private class PpServiceImpl extends PpServiceGrpc.PpService {
        private[this] var processingPipeline: ProcessingPipeline = null
        private[this] var tester: TreadleTester = null
        override def elaborate(lbirModel: Model): Future[PpElaborateReturn] = {
            logger.info("Elaborating model: " + lbirModel.name + " to a processing pipeline circuit.")
            tester = TreadleTester((new ChiselStage).execute(Array(), 
                                    Seq(ChiselGeneratorAnnotation(() => new ProcessingPipeline(lbirModel)))))
            val errReply = ErrorMsg(err = ErrorMsg.ErrorId.SUCCESS, msg="Everything went fine.")
            val ppHandle = PpHandle(name = "model", 
                                    input=lbirModel.layers(0).input,
                                    outShape=lbirModel.layers.last.outShape)
            val ppElaborateReturn = PpElaborateReturn(ppHandle = Option(ppHandle),
                                                      reply = Some(errReply))
            Future.successful(ppElaborateReturn)
        }

        override def run(ppRunParams: PpRunParams): Future[PpRunReturn] = {
            logger.info("Simulating processing pipeline: " + ppRunParams.ppHandle.get.name + " circuit on inputs.")
            tester.poke("io_in", lbirToBigInt(ppRunParams.inputs(0)))
            Future.successful(PpRunReturn(values = List(bigIntToLbir(tester.peek("io_out")))))
        }
    }
}

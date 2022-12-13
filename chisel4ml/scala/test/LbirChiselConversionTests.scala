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
package chisel4ml.tests

import org.scalatest.funsuite.AnyFunSuite
import _root_.chisel4ml.util.LbirUtil
import _root_.lbir.{Datatype, QTensor}
import _root_.lbir.Datatype.QuantizationType.{BINARY, UNIFORM}
import _root_.chisel4ml._
import _root_.chisel4ml.implicits._
import _root_.chisel3._

class LbirChiselConversionTests extends AnyFunSuite {
    val binaryDatatype = Some(
      new Datatype(quantization = BINARY, bitwidth = 1, signed = true, shift = Seq(0), offset = Seq(0))
    )

    // QTENSOR -> UInt
    test("Binary tensor conversion test 0") {
        val qtensor = new QTensor(dtype = binaryDatatype, shape = Seq(4), values = Seq(-1, -1, -1, 1))

        assert(qtensor.toUInt.litValue == "b1000".U.litValue)
    }

    test("Binary tensor conversion test 1") {
        val qtensor = new QTensor(dtype = binaryDatatype, shape = Seq(4), values = Seq(1, 1, 1, -1))

        assert(qtensor.toUInt.litValue == "b0111".U.litValue)
    }

    test("Uniformly quantized tensor to 4-bits conversion test 0") {
        val uniformFourBitNoscaleType =
            Some(new Datatype(quantization = UNIFORM, signed = false, bitwidth = 4, shift = Seq(0), offset = Seq(0)))
        val qtensor                   = new QTensor(dtype = uniformFourBitNoscaleType, shape = Seq(4), values = Seq(4, 3, 2, 1))

        assert(qtensor.toUInt.litValue == "b0001_0010_0011_0100".U.litValue)
    }

    // UInt -> QTENSOR
    test("Convert back a UInt to a uniformy quantized QTensor") {
        val uniformFourBitNoscaleType =
            Some(new Datatype(quantization = UNIFORM, signed = false, bitwidth = 4, shift = Seq(0), offset = Seq(0)))

        val stencil = new QTensor(dtype = uniformFourBitNoscaleType, shape = Seq(4))

        val qtensor = new QTensor(dtype = uniformFourBitNoscaleType, shape = Seq(4), values = Seq(4, 3, 2, 1))

        assert(
          qtensor.toUInt.toQTensor(stencil).values == qtensor.values,
          s"""QTensor was first converted to ${qtensor.toUInt}, with total bitwidth ${qtensor.totalBitwidth} and
             | then back to a qtensor: ${qtensor.toUInt.toQTensor(stencil).values}.""".stripMargin.replaceAll("\n", "")
        )
    }

    test("Convert back a UInt to a signed uniformy quantized QTensor") {
        val uniformFourBitNoscaleType =
            Some(new Datatype(quantization = UNIFORM, signed = true, bitwidth = 4, shift = Seq(0), offset = Seq(0)))

        val stencil = new QTensor(dtype = uniformFourBitNoscaleType, shape = Seq(4))

        val qtensor = new QTensor(dtype = uniformFourBitNoscaleType, shape = Seq(4), values = Seq(-4, -3, 2, 1))

        assert(
          qtensor.toUInt.toQTensor(stencil).values == qtensor.values,
          s"""QTensor was first converted to ${qtensor.toUInt}, with total bitwidth ${qtensor.totalBitwidth} and
             | then back to a qtensor: ${qtensor.toUInt.toQTensor(stencil).values}.""".stripMargin.replaceAll("\n", "")
        )
    }
}

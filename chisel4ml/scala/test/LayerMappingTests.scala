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

import chisel4ml.LayerMapping
import lbir.Datatype.QuantizationType.UNIFORM
import lbir.{Datatype, QTensor}
import org.scalatest.flatspec.AnyFlatSpec
import org.slf4j.LoggerFactory

class LayerMappingTests extends AnyFlatSpec {
  val logger = LoggerFactory.getLogger(classOf[LayerMappingTests])

  val dtype = Datatype(quantization = UNIFORM, signed = false, bitwidth = 4, shift = Seq(0), offset = Seq(0))

  behavior.of("LayerMapping module")
  it should "get the right map for a basic single dimensional convolution" in {
    /*  0 1 2
        3 4 5
        6 7 8
     */
    val reference = Seq(Seq(0, 1, 3, 4), Seq(1, 2, 4, 5), Seq(3, 4, 6, 7), Seq(4, 5, 7, 8))
    val qtensor = QTensor(dtype = dtype, shape = Seq(1, 3, 3))
    val res = LayerMapping.slidingWindowMap(
      qtensor,
      kernelSize = Seq(2, 2),
      stride = Seq(1, 1),
      padding = Seq(0, 0, 0, 0),
      dilation = Seq(),
      groups = 1,
      outChannels = 1
    )
    assert(res == reference)
  }
  it should "get the right map for a single dimensional convolution with symmetric padding of one" in {
    /*  (padding = (1,1))    -1 -1 -1 -1 -1
      A B C                  -1  0  1  2 -1
      D E F         =>       -1  3  4  5 -1
      G I H                  -1  6  7  8 -1
                             -1 -1 -1 -1 -1
     */
    val reference = Seq(
      Seq(-1, -1, -1, 0),
      Seq(-1, -1, 0, 1),
      Seq(-1, -1, 1, 2),
      Seq(-1, -1, 2, -1),
      Seq(-1, 0, -1, 3),
      Seq(0, 1, 3, 4),
      Seq(1, 2, 4, 5),
      Seq(2, -1, 5, -1),
      Seq(-1, 3, -1, 6),
      Seq(3, 4, 6, 7),
      Seq(4, 5, 7, 8),
      Seq(5, -1, 8, -1),
      Seq(-1, 6, -1, -1),
      Seq(6, 7, -1, -1),
      Seq(7, 8, -1, -1),
      Seq(8, -1, -1, -1)
    )
    val qtensor = QTensor(dtype = dtype, shape = Seq(1, 3, 3))
    val res = LayerMapping.slidingWindowMap(
      qtensor,
      kernelSize = Seq(2, 2),
      stride = Seq(1, 1),
      padding = Seq(1, 1, 1, 1), // (top, left, bottom, right)
      dilation = Seq(),
      groups = 1,
      outChannels = 1
    )
    assert(res == reference)
  }
  it should "get the right map for a 3 dimensional convolution" in {
    /*  0 1 2   9  10 11  18 19 20
        3 4 5   12 13 14  21 22 23
        6 7 8   15 16 17  24 25 26
     */
    val reference = Seq(
      Seq(0, 1, 3, 4, 9, 10, 12, 13, 18, 19, 21, 22),
      Seq(1, 2, 4, 5, 10, 11, 13, 14, 19, 20, 22, 23),
      Seq(3, 4, 6, 7, 12, 13, 15, 16, 21, 22, 24, 25),
      Seq(4, 5, 7, 8, 13, 14, 16, 17, 22, 23, 25, 26)
    )
    val qtensor = QTensor(dtype = dtype, shape = Seq(3, 3, 3))
    val res = LayerMapping.slidingWindowMap(
      qtensor,
      kernelSize = Seq(2, 2),
      stride = Seq(1, 1),
      padding = Seq(0, 0, 0, 0),
      dilation = Seq(),
      groups = 1,
      outChannels = 1
    )
    assert(res == reference)
  }
}

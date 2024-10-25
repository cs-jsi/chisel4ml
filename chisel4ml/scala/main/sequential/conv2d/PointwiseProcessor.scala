/*
 * Copyright 2024 Computer Systems Department, Jozef Stefan Insitute
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
package chisel4ml.sequential

/*
 * Computes a Pointwise convolution over a NCHW input tensor.
 * It uses the input stationary approach to do this, and instead
 * loads the kernels rapidly.
 *
 */
/*
class PointwiseProcessor(layer: Conv2DConfig) extends Module  with HasLBIRStream {
  val inStream: AXIStreamIO[UInt] = AXIStream(UInt(8.W), 4)
  val outStream: AXIStreamIO[UInt] = AXIStream(UInt(8.W), 4)
  outStream <> inStream
}
 */

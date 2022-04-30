// Generated by the Scala Plugin for the Protocol Buffer Compiler.
// Do not edit!
//
// Protofile syntax: PROTO3

package lbir

object LbirProto extends _root_.scalapb.GeneratedFileObject {
  lazy val dependencies: Seq[_root_.scalapb.GeneratedFileObject] = Seq.empty
  lazy val messagesCompanions: Seq[_root_.scalapb.GeneratedMessageCompanion[_ <: _root_.scalapb.GeneratedMessage]] =
    Seq[_root_.scalapb.GeneratedMessageCompanion[_ <: _root_.scalapb.GeneratedMessage]](
      lbir.Model,
      lbir.Layer,
      lbir.QTensor,
      lbir.Datatype,
      lbir.Activation
    )
  private lazy val ProtoBytes: _root_.scala.Array[Byte] =
      scalapb.Encoding.fromBase64(scala.collection.immutable.Seq(
  """CgpsYmlyLnByb3RvEgljaGlzZWw0bWwiXQoFTW9kZWwSHQoEbmFtZRgBIAEoCUIJ4j8GEgRuYW1lUgRuYW1lEjUKBmxheWVyc
  xgCIAMoCzIQLmNoaXNlbDRtbC5MYXllckIL4j8IEgZsYXllcnNSBmxheWVycyKnAwoFTGF5ZXISNwoFbHR5cGUYASABKA4yFS5ja
  GlzZWw0bWwuTGF5ZXIuVHlwZUIK4j8HEgVsdHlwZVIFbHR5cGUSJwoIdXNlX2JpYXMYAiABKAhCDOI/CRIHdXNlQmlhc1IHdXNlQ
  mlhcxI3CgZiaWFzZXMYAyABKAsyEi5jaGlzZWw0bWwuUVRlbnNvckIL4j8IEgZiaWFzZXNSBmJpYXNlcxI6Cgd3ZWlnaHRzGAQgA
  SgLMhIuY2hpc2VsNG1sLlFUZW5zb3JCDOI/CRIHd2VpZ2h0c1IHd2VpZ2h0cxI0CgVpbnB1dBgFIAEoCzISLmNoaXNlbDRtbC5RV
  GVuc29yQgriPwcSBWlucHV0UgVpbnB1dBJGCgphY3RpdmF0aW9uGAYgASgLMhUuY2hpc2VsNG1sLkFjdGl2YXRpb25CD+I/DBIKY
  WN0aXZhdGlvblIKYWN0aXZhdGlvbhIqCglvdXRfc2hhcGUYByADKA1CDeI/ChIIb3V0U2hhcGVSCG91dFNoYXBlIh0KBFR5cGUSC
  QoFREVOU0UQABIKCgZDT05WMkQQASKHAQoHUVRlbnNvchI1CgVkdHlwZRgBIAEoCzITLmNoaXNlbDRtbC5EYXRhdHlwZUIK4j8HE
  gVkdHlwZVIFZHR5cGUSIAoFc2hhcGUYAiADKA1CCuI/BxIFc2hhcGVSBXNoYXBlEiMKBnZhbHVlcxgDIAMoAkIL4j8IEgZ2YWx1Z
  XNSBnZhbHVlcyKTAgoIRGF0YXR5cGUSWwoMcXVhbnRpemF0aW9uGAEgASgOMiQuY2hpc2VsNG1sLkRhdGF0eXBlLlF1YW50aXphd
  GlvblR5cGVCEeI/DhIMcXVhbnRpemF0aW9uUgxxdWFudGl6YXRpb24SKQoIYml0d2lkdGgYAiABKA1CDeI/ChIIYml0d2lkdGhSC
  GJpdHdpZHRoEiAKBXNjYWxlGAMgASgCQgriPwcSBXNjYWxlUgVzY2FsZRIjCgZvZmZzZXQYBCABKAJCC+I/CBIGb2Zmc2V0UgZvZ
  mZzZXQiOAoQUXVhbnRpemF0aW9uVHlwZRILCgdVTklGT1JNEAASCgoGQklOQVJZEAESCwoHVEVSTkFSWRADIpcBCgpBY3RpdmF0a
  W9uEjcKAmZuGAEgASgOMh4uY2hpc2VsNG1sLkFjdGl2YXRpb24uRnVuY3Rpb25CB+I/BBICZm5SAmZuEikKCGJpdHdpZHRoGAIgA
  SgNQg3iPwoSCGJpdHdpZHRoUghiaXR3aWR0aCIlCghGdW5jdGlvbhIPCgtCSU5BUllfU0lHThAAEggKBFJFTFUQAUIVCgRsYmlyQ
  gtNb2RlbFByb3Rvc1ABYgZwcm90bzM="""
      ).mkString)
  lazy val scalaDescriptor: _root_.scalapb.descriptors.FileDescriptor = {
    val scalaProto = com.google.protobuf.descriptor.FileDescriptorProto.parseFrom(ProtoBytes)
    _root_.scalapb.descriptors.FileDescriptor.buildFrom(scalaProto, dependencies.map(_.scalaDescriptor))
  }
  lazy val javaDescriptor: com.google.protobuf.Descriptors.FileDescriptor = {
    val javaProto = com.google.protobuf.DescriptorProtos.FileDescriptorProto.parseFrom(ProtoBytes)
    com.google.protobuf.Descriptors.FileDescriptor.buildFrom(javaProto, _root_.scala.Array(
    ))
  }
  @deprecated("Use javaDescriptor instead. In a future version this will refer to scalaDescriptor.", "ScalaPB 0.5.47")
  def descriptor: com.google.protobuf.Descriptors.FileDescriptor = javaDescriptor
}
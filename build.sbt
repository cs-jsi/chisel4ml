ThisBuild / scalaVersion     := "2.13.8"
ThisBuild / version          := "0.1.0"
ThisBuild / organization     := "JSI"

Compile / scalaSource := baseDirectory.value / "chisel4ml" / "scala" / "main"
Compile / unmanagedSourceDirectories += baseDirectory.value / "chisel4ml" / "scala" / "lbir"
Compile / unmanagedSourceDirectories += baseDirectory.value / "chisel4ml" / "scala" / "services"


crossTarget := baseDirectory.value / "bin"
assembly / assemblyJarName  := "chisel4ml.jar"
ThisBuild / assemblyMergeStrategy := {
    case PathList("META-INF", "MANIFEST.MF") => MergeStrategy.discard
    case _ => MergeStrategy.first
}

PB.deleteTargetDirectory := false
PB.additionalDependencies := Nil
Compile / PB.includePaths := Seq(file(root.base.getAbsolutePath))
Compile / PB.protoSources := Seq(baseDirectory.value / "chisel4ml" / "lbir")
Compile / PB.targets := Seq(
  scalapb.gen(flatPackage = true) ->  baseDirectory.value / "chisel4ml" / "scala"
)

val chiselVersion = "3.5.1"
lazy val root = (project in file("."))
  .settings(
    name := "chisel4ml",
    libraryDependencies ++= Seq(
      "edu.berkeley.cs"            %% "chisel3"              % chiselVersion,
      "edu.berkeley.cs"            %% "chiseltest"           % "0.5.2"                                 % "test",
      "edu.berkeley.cs"            %% "treadle"              % "1.5.3",
      "com.thesamet.scalapb"       %% "scalapb-runtime"      % scalapb.compiler.Version.scalapbVersion % "protobuf",
      "io.grpc"                    %  "grpc-netty"           % scalapb.compiler.Version.grpcJavaVersion,
      "com.thesamet.scalapb"       %% "scalapb-runtime-grpc" % scalapb.compiler.Version.scalapbVersion,
      "org.slf4j"                  %  "slf4j-api"             % "1.7.5",
      "org.slf4j"                  %  "slf4j-simple"          % "1.7.5"
    ),
    scalacOptions ++= Seq(
      "-language:reflectiveCalls",
      "-deprecation",
      "-feature",
      "-Xcheckinit",
      "-P:chiselplugin:genBundleElements",
    ),
    addCompilerPlugin("edu.berkeley.cs" % "chisel3-plugin" % chiselVersion cross CrossVersion.full),
  )

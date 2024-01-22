import scala.sys.process._


ThisBuild / scalaVersion := "2.13.10"
ThisBuild / organization := "JSI"
ThisBuild / version := Process("git describe --tags").lineStream(0)

fork := true

Compile / scalaSource := baseDirectory.value / "chisel4ml" / "scala" / "main"
Test / scalaSource := baseDirectory.value / "chisel4ml" / "scala" / "test"
Compile / unmanagedSourceDirectories += baseDirectory.value / "chisel4ml" / "scala" / "lbir"
Compile / unmanagedSourceDirectories += baseDirectory.value / "chisel4ml" / "scala" / "services"
Compile / unmanagedResourceDirectories += baseDirectory.value / "mel-engine" / "src" / "main" / "resources"

// This removes the (anoying) import warnings when running a console
Compile / console / scalacOptions -= "-Ywarn-unused"

crossTarget := baseDirectory.value / "chisel4ml" / "bin"
assembly / assemblyJarName := "chisel4ml.jar"
ThisBuild / assemblyMergeStrategy := {
  case PathList("META-INF", "MANIFEST.MF") => MergeStrategy.discard
  case _                                   => MergeStrategy.first
}

PB.deleteTargetDirectory := false
PB.additionalDependencies := Nil
Compile / PB.includePaths += file(root.base.getAbsolutePath)
Compile / PB.protoSources := Seq(baseDirectory.value / "chisel4ml" / "lbir")
Compile / PB.targets := Seq(
  scalapb.gen(flatPackage = true) -> baseDirectory.value / "chisel4ml" / "scala"
)

Test / logBuffered := false
Test / parallelExecution := true

val chiselVersion = "3.5.6"
val slf4jVersion = "1.7.5"
val scalatestVersion = "3.2.7"
val dependencies = Seq(
  "edu.berkeley.cs" %% "chisel3" % chiselVersion,
  "edu.berkeley.cs" %% "chiseltest" % "0.5.6",
  "edu.berkeley.cs" %% "dsptools" % "1.5.6",
  "com.thesamet.scalapb" %% "scalapb-runtime" % scalapb.compiler.Version.scalapbVersion % "protobuf",
  "io.grpc" % "grpc-netty" % scalapb.compiler.Version.grpcJavaVersion,
  "com.thesamet.scalapb" %% "scalapb-runtime-grpc" % scalapb.compiler.Version.scalapbVersion,
  "org.slf4j" % "slf4j-api" % slf4jVersion,
  "org.slf4j" % "slf4j-simple" % slf4jVersion,
  "org.scalatest" %% "scalatest" % scalatestVersion,
  "org.scalatest" %% "scalatest" % scalatestVersion % "test",
  "org.reflections" % "reflections" % "0.10.2",
  "com.github.scopt" %% "scopt" % "4.1.0"
)

val scalaOptions = Seq(
  "-language:reflectiveCalls",
  "-deprecation",
  "-feature",
  "-Xcheckinit",
  "-Ywarn-unused",
  "-P:chiselplugin:genBundleElements"
)

val commonSettings = Seq(
  libraryDependencies ++= dependencies,
  scalacOptions ++= scalaOptions,
  addCompilerPlugin(("edu.berkeley.cs" % "chisel3-plugin" % chiselVersion).cross(CrossVersion.full))
)
lazy val cde = (project in file("cde")).settings(commonSettings, name := "cde")
lazy val interfaces = (project in file("interfaces")).settings(commonSettings, name := "interfaces")
lazy val memories = (project in file("memories")).settings(commonSettings, name := "memories")
lazy val rocketchip = (project in file("sdf-fft/generators/rocket-chip"))
  .settings(commonSettings, name := "rocket-chip")
lazy val sdf_fft = (project in file("sdf-fft"))
  .settings(commonSettings, name := "sdf-fft")
lazy val melengine = (project in file("mel-engine"))
  .dependsOn(interfaces, memories)
  .settings(commonSettings, name := "melengine")

lazy val root = (project in file("."))
  .dependsOn(melengine, sdf_fft, interfaces, memories)
  .settings(
    commonSettings,
    assembly / mainClass := Some("chisel4ml.Chisel4mlServer"),
    name := "chisel4ml",
    Compile / doc / scalacOptions := Seq("-groups", "-implicits")
  )

inThisBuild(
  List(
    scalaVersion := "2.13.10",
    semanticdbEnabled := true,
    semanticdbVersion := scalafixSemanticdb.revision
  )
)

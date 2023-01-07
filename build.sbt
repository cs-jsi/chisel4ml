ThisBuild / scalaVersion     := "2.13.10"
ThisBuild / version          := "0.1.0"
ThisBuild / organization     := "JSI"
ThisBuild / scalafixDependencies += "com.github.liancheng" %% "organize-imports" % "0.6.0"
inThisBuild(
    List(
        scalaVersion := "2.13.10",
        semanticdbEnabled := true,
        semanticdbVersion := scalafixSemanticdb.revision
    )
)
Compile / scalaSource := baseDirectory.value / "chisel4ml" / "scala" / "main"
Test / scalaSource := baseDirectory.value / "chisel4ml" / "scala" / "test"
Compile / unmanagedSourceDirectories += baseDirectory.value / "chisel4ml" / "scala" / "lbir"
Compile / unmanagedSourceDirectories += baseDirectory.value / "chisel4ml" / "scala" / "services"


crossTarget := baseDirectory.value / "chisel4ml" / "bin"
assembly / assemblyJarName  := "chisel4ml.jar"
ThisBuild / assemblyMergeStrategy := {
    case PathList("META-INF", "MANIFEST.MF") => MergeStrategy.discard
    case _ => MergeStrategy.first
}

PB.deleteTargetDirectory := false
PB.additionalDependencies := Nil
Compile / PB.includePaths += file(root.base.getAbsolutePath)
Compile / PB.protoSources := Seq(baseDirectory.value / "chisel4ml" / "lbir")
Compile / PB.targets := Seq(
  scalapb.gen(flatPackage = true) ->  baseDirectory.value / "chisel4ml" / "scala"
)

Test / logBuffered := false
Test / parallelExecution := true

testOptions += Tests.Setup( cl =>
   cl.loadClass("org.slf4j.LoggerFactory").
     getMethod("getLogger",cl.loadClass("java.lang.String")).
     invoke(null,"ROOT")
)

val chiselVersion = "3.5.5"
val slf4jVersion = "2.0.5"
val scalatestVersion = "3.2.7"
lazy val root = (project in file("."))
  .settings(
    name := "chisel4ml",
    Compile / doc / scalacOptions := Seq("-groups", "-implicits"),
    libraryDependencies ++= Seq(
      "edu.berkeley.cs"            %% "chisel3"              % chiselVersion,
      "edu.berkeley.cs"            %% "chiseltest"           % "0.5.4",
      "com.thesamet.scalapb"       %% "scalapb-runtime"      % scalapb.compiler.Version.scalapbVersion % "protobuf",
      "io.grpc"                    %  "grpc-netty"           % scalapb.compiler.Version.grpcJavaVersion,
      "com.thesamet.scalapb"       %% "scalapb-runtime-grpc" % scalapb.compiler.Version.scalapbVersion,
      "org.slf4j"                  %  "slf4j-api"            % slf4jVersion,
      "org.slf4j"                  %  "slf4j-simple"         % slf4jVersion,
      "org.scalatest"              %% "scalatest"            % scalatestVersion,
      "org.scalatest"              %% "scalatest"            % scalatestVersion % "test",
      "org.nd4j"                   %  "nd4j-native"          % "0.5.0",
      "org.nd4j"                   %  "nd4j-native"          % "0.5.0",
    ),
    scalacOptions ++= Seq(
      "-unchecked",
      "-deprecation",
      "-language:reflectiveCalls",
      "-feature",
      "-Xcheckinit",
      "-Xfatal-warnings",
      "-Ywarn-dead-code",
      "-Ywarn-unused",
      "-P:chiselplugin:genBundleElements",
    ),
    addCompilerPlugin("edu.berkeley.cs" % "chisel3-plugin" % chiselVersion cross CrossVersion.full),
  )

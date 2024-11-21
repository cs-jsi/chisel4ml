import mill._, scalalib._
// import Mill dependency
import mill._
import mill.define.Sources
import mill.modules.Util
import mill.scalalib.TestModule.ScalaTest
import scalalib._
// support BSP
import mill.bsp._
// protobuf stuff
import $ivy.`com.lihaoyi::mill-contrib-scalapblib:`
import contrib.scalapblib._
// scalafix
import $ivy.`com.goyeau::mill-scalafix::0.4.2`
import com.goyeau.mill.scalafix.ScalafixModule

trait ScalaVersionModule extends SbtModule {
    override def scalaVersion = "2.13.10"
    override def scalacOptions = Seq(
        "-language:reflectiveCalls",
        "-deprecation",
        "-feature",
        "-Xcheckinit",
        "-Ywarn-unused",
    )
}

trait BaseChiselModule extends ScalaVersionModule {
    override def ivyDeps = Agg(
        ivy"org.chipsalliance::chisel:5.0.0",
        ivy"edu.berkeley.cs::chiseltest:5.0.0",
    )
    override def scalacPluginIvyDeps = Agg(
        ivy"org.chipsalliance:::chisel-plugin:5.0.0",
    )
}

object chisel4ml extends BaseChiselModule with ScalaPBModule with ScalafixModule { m =>
    override def millSourcePath = os.pwd
    def sources = T.sources(Seq(PathRef(millSourcePath / "chisel4ml" / "scala")))

    def gitInfo = T.input {
        os.proc("git", "describe", "--tags").call().out.text()
    }

    def gitInfoFileResourceDir = T {
        val dest = T.dest / "versionInfo" / "gitInfo"
        os.write(dest, gitInfo(), createFolders = true)
        PathRef(T.dest)
    }

    override def resources = T.sources { Seq(gitInfoFileResourceDir()) }

    override def scalaPBVersion = "0.11.15"
    override def scalaPBGrpc = true
    override def scalaPBSearchDeps = true
    override def scalaPBFlatPackage: T[Boolean] = true
    override def scalaPBSources = T.sources {
        millSourcePath / "chisel4ml" / "lbir"
    }
    def compileScalaPB = T {
        val destPath = millSourcePath / "chisel4ml" / "scala"
        os.copy(super.compileScalaPB().path, destPath, mergeFolders = true, replaceExisting = true)
        PathRef(destPath).withRevalidateOnce
    }

    def ivyDeps = super.ivyDeps() ++ Agg(
        ivy"org.slf4j:slf4j-api:1.7.5",
        ivy"org.slf4j:slf4j-simple:1.7.5",
        ivy"com.github.scopt::scopt:4.1.0",
        ivy"org.reflections:reflections:0.10.2",
        ivy"io.grpc:grpc-netty:1.63.0",
    )
    def moduleDeps = Seq(interfaces,
                         memories,
                         dsptools,
                         rocketchip,
                         `sdf-fft`,
                         `mel-engine`,
    )
    object test extends SbtModuleTests with TestModule.ScalaTest {
        def sources = T.sources(Seq(PathRef(millSourcePath / "chisel4ml" / "scala" / "test")))
        override def ivyDeps = super.ivyDeps() ++ Agg(
            ivy"org.scalatest::scalatest::3.2.16",
        )
    }
}

object interfaces extends BaseChiselModule with ScalafixModule
object memories extends BaseChiselModule  with ScalafixModule {
    def ivyDeps = super.ivyDeps() ++ Agg(
        ivy"org.slf4j:slf4j-api:1.7.5",
        ivy"org.slf4j:slf4j-simple:1.7.5",
    )
}
object dsptools extends BaseChiselModule {
    override def millSourcePath = os.pwd / "rocket-dsp-utils" / "tools" / "dsptools"
    override def sources = T.sources(Seq(PathRef(millSourcePath / "src" / "main" / "scala" / "dsptools")))
    def ivyDeps = super.ivyDeps() ++ Agg(
        ivy"org.typelevel::spire:0.18.0",
        ivy"org.scalanlp::breeze:2.1.0",
    )
    def moduleDeps = Seq(fixedpoint)
}
object fixedpoint extends BaseChiselModule {
    override def millSourcePath = os.pwd / "rocket-dsp-utils" / "tools" / "dsptools" / "fixedpoint"
}

object `sdf-fft` extends  BaseChiselModule {
    def sources = T.sources(Seq(PathRef(millSourcePath / "src" / "main" / "scala")))
    def moduleDeps = Seq(dsptools,
                         `rocket-dsp-utils`,
                         rocketchip,
                         fixedpoint,
    )
}

object `mel-engine` extends BaseChiselModule with ScalafixModule {
    def sources = T.sources(Seq(PathRef(millSourcePath / "src" / "main" / "scala")))
    def moduleDeps = Seq(memories,
                         interfaces,
                         fixedpoint,
    )
}

object rocketchip extends BaseChiselModule {
    override def millSourcePath = os.pwd / "rocket-dsp-utils" / "tools" / "rocket-chip"
    def moduleDeps = Seq(cde, hardfloat, macros)
        def ivyDeps = super.ivyDeps() ++ Agg(
            ivy"${scalaOrganization()}:scala-reflect:${scalaVersion()}",
            ivy"org.json4s::json4s-jackson:4.0.5",
            ivy"org.scalatest::scalatest:3.2.0",
            ivy"com.lihaoyi::mainargs:0.5.0"
    )
}

object hardfloat extends BaseChiselModule {
    override def millSourcePath = os.pwd / "rocket-dsp-utils" / "tools" / "rocket-chip" / "hardfloat"
}

object macros extends ScalaVersionModule {
    override def millSourcePath = os.pwd / "rocket-dsp-utils" / "tools" / "rocket-chip" / "macros"
    def ivyDeps = Agg(
        ivy"org.scala-lang:scala-reflect:2.13.10"
    )
}

object cde extends BaseChiselModule {
    override def millSourcePath = os.pwd / "rocket-dsp-utils" / "tools" / "rocket-chip" / "cde"
    override def sources = T.sources(Seq(PathRef(millSourcePath / "cde" / "src" / "chipsalliance" / "rocketchip")))
}

object `rocket-dsp-utils` extends BaseChiselModule {
    def moduleDeps = Seq(dsptools,
                         rocketchip,
                         cde,
                         fixedpoint
    )
}

ThisBuild / version := "0.1.0-SNAPSHOT"

ThisBuild / scalaVersion := "2.13.16"

lazy val root = (project in file("."))
  .settings(
    name := "spark-nlp-labs",

    fork := true,
    javaOptions ++= Seq(
      "--add-opens=java.base/java.nio=ALL-UNNAMED",
      "--add-opens=java.base/java.nio.channels=ALL-UNNAMED",
      "--add-opens=java.base/java.lang=ALL-UNNAMED",
      "--add-opens=java.base/java.io=ALL-UNNAMED",
      "--add-opens=java.base/sun.nio.ch=ALL-UNNAMED",
      "--add-opens=java.base/java.lang.invoke=ALL-UNNAMED"
      ,"--add-opens=java.base/java.util=ALL-UNNAMED"
    ),

    libraryDependencies ++= Seq(
      "org.apache.spark" %% "spark-core" % "4.0.1",
      "org.apache.spark" %% "spark-sql" % "4.0.1",
      "org.apache.spark" %% "spark-mllib" % "4.0.1"
    )
  )

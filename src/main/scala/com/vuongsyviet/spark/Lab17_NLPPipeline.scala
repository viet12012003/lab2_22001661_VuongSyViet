package com.vuongsyviet.spark

import org.apache.spark.sql.SparkSession
import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.feature.{HashingTF, IDF, RegexTokenizer, StopWordsRemover, Tokenizer}
import org.apache.spark.sql.functions._
import java.io.{File, PrintWriter}
// import com.harito.spark.Utils._

object Lab17_NLPPipeline {
  def main(args: Array[String]): Unit = {
    val spark = SparkSession.builder
      .appName("NLP Pipeline Example")
      .master("local[*]")
      .getOrCreate()

    import spark.implicits._
    println("Spark Session created successfully.")
    println(s"Spark UI available at http://localhost:4040")
    println("Pausing for 10 seconds to allow you to open the Spark UI...")
    Thread.sleep(10000)

    val dataPath = """F:/NLP/lab2_22001661_VuongSyViet/data/c4-train.00000-of-01024-30K.json"""
    val initialDF = spark.read.json(dataPath).limit(1000)

    // 1. Switch Tokenizers
    val tokenizer1 = new Tokenizer().setInputCol("text").setOutputCol("tokens")
    val stopWordsRemover1 = new StopWordsRemover().setInputCol("tokens").setOutputCol("filtered_tokens")
    val hashingTF1 = new HashingTF().setInputCol("filtered_tokens").setOutputCol("raw_features").setNumFeatures(20000)
    val idf1 = new IDF().setInputCol("raw_features").setOutputCol("features")
    val pipeline1 = new Pipeline().setStages(Array(tokenizer1, stopWordsRemover1, hashingTF1, idf1))
    val log_path1 = """F:/NLP/lab2_22001661_VuongSyViet/log/lab17_metrics1.log"""
    val startTime1 = java.time.LocalDateTime.now()
    val fitStart1 = System.nanoTime()
    val pipelineModel1 = pipeline1.fit(initialDF)
    val fitDuration1 = (System.nanoTime() - fitStart1) / 1e9d
    val transformStart1 = System.nanoTime()
    val transformedDF1 = pipelineModel1.transform(initialDF)
    val transformDuration1 = (System.nanoTime() - transformStart1) / 1e9d
    val recordCount1 = transformedDF1.count()
    val actualVocabSize1 = transformedDF1.select(explode(col("filtered_tokens")).as("word")).filter(length(col("word")) > 1).distinct().count()
    val results1 = transformedDF1.select("text", "features").take(20)
    val result_path1 = """F:/NLP/lab2_22001661_VuongSyViet/results/lab17_pipeline_output1.txt"""
    new File(result_path1).getParentFile.mkdirs()
    val resultWriter1 = new PrintWriter(new File(result_path1))
    try {
      resultWriter1.println("--- NLP Pipeline Output (First 20 results) ---")
      resultWriter1.println(s"Output file generated at: ${new File(result_path1).getAbsolutePath}\n")
      results1.foreach { row =>
        resultWriter1.println("="*80)
        val text = row.getAs[String]("text")
        val features = row.getAs[org.apache.spark.ml.linalg.Vector]("features")
        resultWriter1.println(s"Original Text: ${text.substring(0, Math.min(text.length, 100))}...")
        resultWriter1.println(s"TF-IDF Vector: ${features.toString}")
        resultWriter1.println("="*80)
        resultWriter1.println()
      }
      println(s"Successfully wrote results to $result_path1")
    } finally { resultWriter1.close() }
    val logWriter1 = new PrintWriter(new File(log_path1))
    try {
      logWriter1.println("--- Performance Metrics ---")
      logWriter1.println(f"Pipeline fitting duration: $fitDuration1%.2f seconds")
      logWriter1.println(f"Data transformation duration: $transformDuration1%.2f seconds")
      logWriter1.println(s"Actual vocabulary size (after preprocessing): $actualVocabSize1 unique terms")
      logWriter1.println(s"HashingTF numFeatures set to: 20000")
      if (20000 < actualVocabSize1) {
        logWriter1.println(s"Note: numFeatures (20000) is smaller than actual vocabulary size ($actualVocabSize1). Hash collisions are expected.")
      }
      logWriter1.println(s"Metrics file generated at: ${new File(log_path1).getAbsolutePath}")
      logWriter1.println("\nFor detailed stage-level metrics, view the Spark UI at http://localhost:4040 during execution.")
    } finally { logWriter1.close() }

    // 2. Adjust Feature Vector Size
    val tokenizer2 = new RegexTokenizer().setInputCol("text").setOutputCol("tokens").setPattern("\\s+|[.,;!?()\"']")
    val stopWordsRemover2 = new StopWordsRemover().setInputCol("tokens").setOutputCol("filtered_tokens")
    val hashingTF2 = new HashingTF().setInputCol("filtered_tokens").setOutputCol("raw_features").setNumFeatures(1000)
    val idf2 = new IDF().setInputCol("raw_features").setOutputCol("features")
    val pipeline2 = new Pipeline().setStages(Array(tokenizer2, stopWordsRemover2, hashingTF2, idf2))
    val log_path2 = """F:/NLP/lab2_22001661_VuongSyViet/log/lab17_metrics2.log"""
    val startTime2 = java.time.LocalDateTime.now()
    val fitStart2 = System.nanoTime()
    val pipelineModel2 = pipeline2.fit(initialDF)
    val fitDuration2 = (System.nanoTime() - fitStart2) / 1e9d
    val transformStart2 = System.nanoTime()
    val transformedDF2 = pipelineModel2.transform(initialDF)
    val transformDuration2 = (System.nanoTime() - transformStart2) / 1e9d
    val recordCount2 = transformedDF2.count()
    val actualVocabSize2 = transformedDF2.select(explode(col("filtered_tokens")).as("word")).filter(length(col("word")) > 1).distinct().count()
    val results2 = transformedDF2.select("text", "features").take(20)
    val result_path2 = """F:/NLP/lab2_22001661_VuongSyViet/results/lab17_pipeline_output2.txt"""
    new File(result_path2).getParentFile.mkdirs()
    val resultWriter2 = new PrintWriter(new File(result_path2))
    try {
      resultWriter2.println("--- NLP Pipeline Output (First 20 results) ---")
      resultWriter2.println(s"Output file generated at: ${new File(result_path2).getAbsolutePath}\n")
      results2.foreach { row =>
        resultWriter2.println("="*80)
        val text = row.getAs[String]("text")
        val features = row.getAs[org.apache.spark.ml.linalg.Vector]("features")
        resultWriter2.println(s"Original Text: ${text.substring(0, Math.min(text.length, 100))}...")
        resultWriter2.println(s"TF-IDF Vector: ${features.toString}")
        resultWriter2.println("="*80)
        resultWriter2.println()
      }
      println(s"Successfully wrote results to $result_path2")
    } finally { resultWriter2.close() }
    val logWriter2 = new PrintWriter(new File(log_path2))
    try {
      logWriter2.println("--- Performance Metrics ---")
      logWriter2.println(f"Pipeline fitting duration: $fitDuration2%.2f seconds")
      logWriter2.println(f"Data transformation duration: $transformDuration2%.2f seconds")
      logWriter2.println(s"Actual vocabulary size (after preprocessing): $actualVocabSize2 unique terms")
      logWriter2.println(s"HashingTF numFeatures set to: 1000")
      if (1000 < actualVocabSize2) {
        logWriter2.println(s"Note: numFeatures (1000) is smaller than actual vocabulary size ($actualVocabSize2). Hash collisions are expected.")
      }
      logWriter2.println(s"Metrics file generated at: ${new File(log_path2).getAbsolutePath}")
      logWriter2.println("\nFor detailed stage-level metrics, view the Spark UI at http://localhost:4040 during execution.")
    } finally { logWriter2.close() }

    // 3. Extend the Pipeline (LogisticRegression)
    import org.apache.spark.ml.classification.LogisticRegression
    // Add a dummy label column for demonstration (random 0/1)
    val withLabelDF = initialDF.withColumn("label", (rand()*2).cast("int"))
    val tokenizer3 = new RegexTokenizer().setInputCol("text").setOutputCol("tokens").setPattern("\\s+|[.,;!?()\"']")
    val stopWordsRemover3 = new StopWordsRemover().setInputCol("tokens").setOutputCol("filtered_tokens")
    val hashingTF3 = new HashingTF().setInputCol("filtered_tokens").setOutputCol("raw_features").setNumFeatures(20000)
    val idf3 = new IDF().setInputCol("raw_features").setOutputCol("features")
    val lr = new LogisticRegression().setFeaturesCol("features").setLabelCol("label").setMaxIter(10)
    val pipeline3 = new Pipeline().setStages(Array(tokenizer3, stopWordsRemover3, hashingTF3, idf3, lr))
    val log_path3 = """F:/NLP/lab2_22001661_VuongSyViet/log/lab17_metrics3.log"""
    val startTime3 = java.time.LocalDateTime.now()
    val fitStart3 = System.nanoTime()
    val model3 = pipeline3.fit(withLabelDF)
    val fitDuration3 = (System.nanoTime() - fitStart3) / 1e9d
    val transformStart3 = System.nanoTime()
    val transformedDF3 = model3.transform(withLabelDF)
    val transformDuration3 = (System.nanoTime() - transformStart3) / 1e9d
    val recordCount3 = transformedDF3.count()
    val actualVocabSize3 = transformedDF3.select(explode(col("filtered_tokens")).as("word")).filter(length(col("word")) > 1).distinct().count()
    val results3 = transformedDF3.select("text", "label", "prediction", "probability").take(20)
    val result_path3 = """F:/NLP/lab2_22001661_VuongSyViet/results/lab17_pipeline_output3.txt"""
    new File(result_path3).getParentFile.mkdirs()
    val resultWriter3 = new PrintWriter(new File(result_path3))
    try {
      resultWriter3.println("--- NLP Pipeline Output (First 20 results) ---")
      resultWriter3.println(s"Output file generated at: ${new File(result_path3).getAbsolutePath}\n")
      results3.foreach { row =>
        resultWriter3.println("="*80)
        val text = row.getAs[String]("text")
        val label = row.getAs[Int]("label")
        val prediction = row.getAs[Double]("prediction")
        val probability = row.getAs[org.apache.spark.ml.linalg.Vector]("probability")
        resultWriter3.println(s"Original Text: ${text.substring(0, Math.min(text.length, 100))}...")
        resultWriter3.println(s"Label: $label, Prediction: $prediction, Probability: $probability")
        resultWriter3.println("="*80)
        resultWriter3.println()
      }
      println(s"Successfully wrote results to $result_path3")
    } finally { resultWriter3.close() }
    val logWriter3 = new PrintWriter(new File(log_path3))
    try {
      logWriter3.println("--- Performance Metrics ---")
      logWriter3.println(f"Pipeline fitting duration: $fitDuration3%.2f seconds")
      logWriter3.println(f"Data transformation duration: $transformDuration3%.2f seconds")
      logWriter3.println(s"Actual vocabulary size (after preprocessing): $actualVocabSize3 unique terms")
      logWriter3.println(s"HashingTF numFeatures set to: 20000")
      logWriter3.println(s"Tokenizer: RegexTokenizer (regex)")
      logWriter3.println(s"Classifier: LogisticRegression")
      logWriter3.println(s"Metrics file generated at: ${new File(log_path3).getAbsolutePath}")
      logWriter3.println("\nFor detailed stage-level metrics, view the Spark UI at http://localhost:4040 during execution.")
    } finally { logWriter3.close() }

    // 4. Try a Different Vectorizer (Word2Vec)
    import org.apache.spark.ml.feature.Word2Vec
    val tokenizer4 = new RegexTokenizer().setInputCol("text").setOutputCol("tokens").setPattern("\\s+|[.,;!?()\"']")
    val stopWordsRemover4 = new StopWordsRemover().setInputCol("tokens").setOutputCol("filtered_tokens")
    val word2vec = new Word2Vec().setInputCol("filtered_tokens").setOutputCol("embeddings").setVectorSize(100).setMinCount(1)
    val pipeline4 = new Pipeline().setStages(Array(tokenizer4, stopWordsRemover4, word2vec))
    val log_path4 = """F:/NLP/lab2_22001661_VuongSyViet/log/lab17_metrics4.log"""
    val startTime4 = java.time.LocalDateTime.now()
    val fitStart4 = System.nanoTime()
    val pipelineModel4 = pipeline4.fit(initialDF)
    val fitDuration4 = (System.nanoTime() - fitStart4) / 1e9d
    val transformStart4 = System.nanoTime()
    val transformedDF4 = pipelineModel4.transform(initialDF)
    val transformDuration4 = (System.nanoTime() - transformStart4) / 1e9d
    val recordCount4 = transformedDF4.count()
    val actualVocabSize4 = transformedDF4.select(explode(col("filtered_tokens")).as("word")).filter(length(col("word")) > 1).distinct().count()
    val results4 = transformedDF4.select("text", "embeddings").take(20)
    val result_path4 = """F:/NLP/lab2_22001661_VuongSyViet/results/lab17_pipeline_output4.txt"""
    new File(result_path4).getParentFile.mkdirs()
    val resultWriter4 = new PrintWriter(new File(result_path4))
    try {
      resultWriter4.println("--- NLP Pipeline Output (First 20 results) ---")
      resultWriter4.println(s"Output file generated at: ${new File(result_path4).getAbsolutePath}\n")
      results4.foreach { row =>
        resultWriter4.println("="*80)
        val text = row.getAs[String]("text")
        val embeddings = row.getAs[org.apache.spark.ml.linalg.Vector]("embeddings")
        resultWriter4.println(s"Original Text: ${text.substring(0, Math.min(text.length, 100))}...")
        resultWriter4.println(s"Word2Vec Embeddings: ${embeddings.toString}")
        resultWriter4.println("="*80)
        resultWriter4.println()
      }
      println(s"Successfully wrote results to $result_path4")
    } finally { resultWriter4.close() }
    val logWriter4 = new PrintWriter(new File(log_path4))
    try {
      logWriter4.println("--- Performance Metrics ---")
      logWriter4.println(f"Pipeline fitting duration: $fitDuration4%.2f seconds")
      logWriter4.println(f"Data transformation duration: $transformDuration4%.2f seconds")
      logWriter4.println(s"Actual vocabulary size (after preprocessing): $actualVocabSize4 unique terms")
      logWriter4.println(s"Vectorizer: Word2Vec, vectorSize=100")
      logWriter4.println(s"Tokenizer: RegexTokenizer (regex)")
      logWriter4.println(s"Metrics file generated at: ${new File(log_path4).getAbsolutePath}")
      logWriter4.println("\nFor detailed stage-level metrics, view the Spark UI at http://localhost:4040 during execution.")
    } finally { logWriter4.close() }

    spark.stop()
    println("Spark Session stopped.")
  }
}
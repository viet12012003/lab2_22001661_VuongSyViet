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
    // Thread.sleep(10000)

    val dataPath = """F:/NLP/lab2_22001661_VuongSyViet/data/c4-train.00000-of-01024-30K.json"""

    val limitDocuments = 1000
    val initialDF = spark.read.json(dataPath).limit(limitDocuments)

    def time[R](block: => R): (R, Double) = {
      val t0 = System.nanoTime()
      val result = block
      val t1 = System.nanoTime()
      (result, (t1 - t0) / 1e9d)
    }

    // 1. Switch Tokenizers
    val tokenizer1 = new Tokenizer().setInputCol("text").setOutputCol("tokens")
    val stopWordsRemover1 = new StopWordsRemover().setInputCol("tokens").setOutputCol("filtered_tokens")
    val hashingTF1 = new HashingTF().setInputCol("filtered_tokens").setOutputCol("raw_features").setNumFeatures(20000)
    val idf1 = new IDF().setInputCol("raw_features").setOutputCol("features")
    val pipeline1 = new Pipeline().setStages(Array(tokenizer1, stopWordsRemover1, hashingTF1, idf1))
    val log_path1 = """F:/NLP/lab2_22001661_VuongSyViet/log/lab17_metrics1.log"""
    val (pipelineModel1, fitDuration1) = time { pipeline1.fit(initialDF) }
    val (transformedDF1, transformDuration1) = time { pipelineModel1.transform(initialDF) }
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
    val (pipelineModel2, fitDuration2) = time { pipeline2.fit(initialDF) }
    val (transformedDF2, transformDuration2) = time { pipelineModel2.transform(initialDF) }
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
    val (model3, fitDuration3) = time { pipeline3.fit(withLabelDF) }
    val (transformedDF3, transformDuration3) = time { model3.transform(withLabelDF) }
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

    // 4. Try a Different Vectorizer (Word2Vec) + Vector Normalization:
    import org.apache.spark.ml.feature.{Word2Vec, Normalizer}
    val tokenizer4 = new RegexTokenizer().setInputCol("text").setOutputCol("tokens").setPattern("\\s+|[.,;!?()\"']")
    val stopWordsRemover4 = new StopWordsRemover().setInputCol("tokens").setOutputCol("filtered_tokens")
    val word2vec = new Word2Vec().setInputCol("filtered_tokens").setOutputCol("embeddings").setVectorSize(100).setMinCount(1)
    val normalizer = new Normalizer().setInputCol("embeddings").setOutputCol("norm_embeddings").setP(2.0)
    val pipeline4 = new Pipeline().setStages(Array(tokenizer4, stopWordsRemover4, word2vec, normalizer))
    val log_path4 = """F:/NLP/lab2_22001661_VuongSyViet/log/lab17_metrics4.log"""
    val (pipelineModel4, fitDuration4) = time { pipeline4.fit(initialDF) }
    val (transformedDF4, transformDuration4) = time { pipelineModel4.transform(initialDF) }
    val recordCount4 = transformedDF4.count()
    val actualVocabSize4 = transformedDF4.select(explode(col("filtered_tokens")).as("word")).filter(length(col("word")) > 1).distinct().count()
    val results4 = transformedDF4.select("text", "norm_embeddings").take(20)
    val result_path4 = """F:/NLP/lab2_22001661_VuongSyViet/results/lab17_pipeline_output4.txt"""
    new File(result_path4).getParentFile.mkdirs()
    val resultWriter4 = new PrintWriter(new File(result_path4))
    try {
      resultWriter4.println("--- NLP Pipeline Output (First 20 results) ---")
      resultWriter4.println(s"Output file generated at: ${new File(result_path4).getAbsolutePath}\n")
      results4.foreach { row =>
        resultWriter4.println("="*80)
        val text = row.getAs[String]("text")
        val embeddings = row.getAs[org.apache.spark.ml.linalg.Vector]("norm_embeddings")
        resultWriter4.println(s"Original Text: ${text.substring(0, Math.min(text.length, 100))}...")
        resultWriter4.println(s"Word2Vec Normalized Embeddings: ${embeddings.toString}")
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
      logWriter4.println(s"Vectorizer: Word2Vec, vectorSize=100 (normalized)")
      logWriter4.println(s"Tokenizer: RegexTokenizer (regex)")
      logWriter4.println(s"Metrics file generated at: ${new File(log_path4).getAbsolutePath}")
      logWriter4.println("\nFor detailed stage-level metrics, view the Spark UI at http://localhost:4040 during execution.")
    } finally { logWriter4.close() }

    // --- Yêu cầu 4: Tìm kiếm văn bản tương tự nhất bằng cosine similarity ---
    import breeze.linalg.{DenseVector => BreezeVector, norm, sum}
    def cosineSimilarity(v1: org.apache.spark.ml.linalg.Vector, v2: org.apache.spark.ml.linalg.Vector): Double = {
      val bv1 = BreezeVector(v1.toArray)
      val bv2 = BreezeVector(v2.toArray)
      val dot = sum(bv1 *:* bv2)
      val n1 = norm(bv1)
      val n2 = norm(bv2)
      if (n1 == 0.0 || n2 == 0.0) 0.0 else dot / (n1 * n2)
    }

    // Lấy 1 văn bản làm query (dòng đầu tiên)
    val queryRow = transformedDF4.select("text", "norm_embeddings").head()
    val queryText = queryRow.getAs[String]("text")
    val queryVec = queryRow.getAs[org.apache.spark.ml.linalg.Vector]("norm_embeddings")
    // Tính cosine similarity với các văn bản còn lại
    val similarities = transformedDF4.select("text", "norm_embeddings").rdd.zipWithIndex.filter(_._2 > 0).map { case (row, idx) =>
      val text = row.getAs[String]("text")
      val vec = row.getAs[org.apache.spark.ml.linalg.Vector]("norm_embeddings")
      val sim = cosineSimilarity(queryVec, vec)
      (text, sim)
    }.collect().sortBy(-_._2)
    val simResultPath = """F:/NLP/lab2_22001661_VuongSyViet/results/lab17_cosine_similarity.txt"""
    new File(simResultPath).getParentFile.mkdirs()
    val simWriter = new PrintWriter(new File(simResultPath))
    try {
      simWriter.println("--- Cosine Similarity Search Results ---")
      simWriter.println(s"Query text (first 100 chars): ${queryText.substring(0, Math.min(queryText.length, 100))}...")
      simWriter.println("\nTop 5 most similar documents:")
      similarities.take(5).zipWithIndex.foreach { case ((text, sim), idx) =>
        simWriter.println(s"#${idx+1} | Similarity: %.4f".format(sim))
        simWriter.println(s"Text: ${text.substring(0, Math.min(text.length, 100))}...")
        simWriter.println("-"*60)
      }
      println(s"Successfully wrote cosine similarity results to $simResultPath")
    } finally { simWriter.close() }

    spark.stop()
    println("Spark Session stopped.")
  }
}
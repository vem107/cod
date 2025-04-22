import pandas as pd
from pyspark.sql import SparkSession
from pyspark.ml.feature import StandardScaler, VectorAssembler, StringIndexer
from pyspark.sql.functions import col, when, avg as spark_avg
from pyspark.sql import Window
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
spark = SparkSession.builder.appName("StockMarketPrediction").getOrCreate()
df_spark = spark.read.csv("", header=True, inferSchema=True)
df_spark = df_spark.dropna()
windowSpec = Window.partitionBy("ticker").orderBy("date").rowsBetween(-4, 0)
columns_for_ma = ["open", "close", "volume", "INCREMENTO", "diff"]
for col_name in columns_for_ma:
    df_spark = df_spark.withColumn(f"MovingAvg{col_name}", spark_avg(col_name).over(windowSpec))
features = [
    'open', 'high', 'low', 'close', 'adjclose', 'volume',
    'RSIadjclose15', 'RSIvolume15', 'RSIadjclose25', 'RSIvolume25',
    'RSIadjclose50', 'RSIvolume50', 'MACDadjclose15', 'MACDvolume15',
    'MACDadjclose25', 'MACDvolume25', 'MACDadjclose50', 'MACDvolume50',
    'MovingAvgopen', 'MovingAvgclose', 'MovingAvgvolume',
    'MovingAvgINCREMENTO', 'MovingAvgdiff'
]
indexer = StringIndexer(inputCol="ticker", outputCol="tickerIndex")
df_indexed = indexer.fit(df_spark).transform(df_spark)
features.append("tickerIndex")
assembler = VectorAssembler(inputCols=features, outputCol="features")
df_assembled = assembler.transform(df_indexed)
scaler = StandardScaler(inputCol="features", outputCol="scaledFeatures", withStd=True, withMean=True)
scaler_model = scaler.fit(df_assembled)
df_scaled = scaler_model.transform(df_assembled)
train_data, test_data = df_scaled.randomSplit([0.8, 0.2], seed=42)
lr = LogisticRegression(featuresCol="scaledFeatures", labelCol="TARGET", maxIter=10)
print("Training Logistic Regression model...")
lr_model = lr.fit(train_data)
predictions = lr_model.transform(test_data)
evaluator = MulticlassClassificationEvaluator(labelCol="TARGET", predictionCol="prediction", metricName="accuracy")
accuracy = evaluator.evaluate(predictions)
print(f"Logistic Regression Accuracy: {accuracy:.4f}")
evaluator.setMetricName("f1")
f1_score = evaluator.evaluate(predictions)
print(f"Logistic Regression F1 Score: {f1_score:.4f}")

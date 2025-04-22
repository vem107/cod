from pyspark.sql import SparkSession
from pyspark.sql.functions import col
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.feature import StandardScaler, VectorAssembler, StringIndexer
from pyspark.sql.functions import to_timestamp, hour, minute, second

spark = SparkSession.builder.appName('Traffic').getOrCreate()

df_spark = spark.read.csv("", header=True, inferSchema=True)

spark.conf.set("spark.sql.legacy.timeParserPolicy", "LEGACY")
df_spark_with_time = df_spark.withColumn("ExtractedTime", to_timestamp(col("Time"), "hh:mm:ss a"))
df_spark_with_time = df_spark_with_time.withColumn("Hour", hour(col("ExtractedTime")))
df_spark_with_time = df_spark_with_time.withColumn("Minute", minute(col("ExtractedTime")))
df_spark_with_time = df_spark_with_time.withColumn("Second", second(col("ExtractedTime")))

index = StringIndexer(inputCol="Day of the week", outputCol='Day')
indexing = index.fit(df_spark_with_time)
indexed_df = indexing.transform(df_spark_with_time)
print("Columns after indexing:", indexed_df.columns)
label_index = StringIndexer(inputCol="Traffic Situation", outputCol='Traffic Situation encoded')
label_indexing = label_index.fit(df_spark_with_time)
label_indexed_df = label_indexing.transform(indexed_df)
print("Columns after label indexing:", label_indexed_df.columns)
features = ['Day', 'CarCount', 'BikeCount', 'BusCount', 'TruckCount', 'Total', 'Hour', 'Minute', 'Second']
assembler = VectorAssembler(inputCols=features, outputCol="vectorized_features")
vectorized_df = assembler.transform(label_indexed_df) 
scale = StandardScaler(inputCol="vectorized_features", outputCol="scaled_features")
scaler = scale.fit(vectorized_df)
scaled_df = scaler.transform(vectorized_df)
final_df = scaled_df.select("scaled_features", "Traffic Situation encoded")
lr = LogisticRegression(featuresCol="scaled_features", labelCol="Traffic Situation encoded")
lr_model = lr.fit(final_df)
predictions = lr_model.transform(final_df)
predictions.show()

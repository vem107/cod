# H code
from pyspark.sql import SparkSession
from pyspark.ml.feature import VectorAssembler, PCA
import pandas as pd
import matplotlib.pyplot as plt

spark = SparkSession.builder.appName("IrisPCA").getOrCreate()
df = spark.read.csv("Iris.csv", header=True, inferSchema=True)
df = df.drop("species")

assembler = VectorAssembler(inputCols=df.columns, outputCol="features")
df_vector = assembler.transform(df)

pca = PCA(k=2, inputCol="features", outputCol="pca_features")
model = pca.fit(df_vector)
result = model.transform(df_vector).select("pca_features")

# Convert to Pandas for plotting
pd_df = result.toPandas()
pd_df[['PC1', 'PC2']] = pd_df['pca_features'].apply(lambda x: pd.Series(x.toArray()))

# Plot PCA result
plt.figure(figsize=(8, 6))
plt.scatter(pd_df['PC1'], pd_df['PC2'], color='blue', alpha=0.6, edgecolor='k')
plt.title("PCA of Iris Data")
plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.grid(True)
plt.show()

spark.stop()




# S code 
from pyspark.sql import SparkSession
from pyspark.ml.feature import StandardScaler,VectorAssembler,StringIndexer
from pyspark.sql.functions import col,when
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.evaluation import MulticlassClassificationEvaluator

spark = SparkSession.builder.appName("MV").getOrCreate()
df_spark = spark.read.csv("",header=True,inferSchema=True)
# encoding 
indexer = StringIndexer(inputCol="Species",outputCol="Enocded_Species")
index = indexer.fit(dataset=df_spark)
indexed_df=index.transform(df_spark)
df_spark = df_spark.dropna()
# vectorizing and scaling
features = [
 'SepalLengthCm',
 'SepalWidthCm',
 'PetalLengthCm',
 'PetalWidthCm']
assembler = VectorAssembler(inputCols=features,outputCol="vectorized_features")
scaler = StandardScaler(inputCol="vectorized_features",outputCol="scaled_features")
vectorized_df = assembler.transform(indexed_df)
scale = scaler.fit(vectorized_df)
scaled_df = scale.transform(vectorized_df)
LR = LogisticRegression(featuresCol="scaled_features",labelCol="Enocded_Species")
train,test = scaled_df.randomSplit([0.8,0.2])
model=LR.fit(train)
predictions = model.transform(test)
original_labels = index.labels


evaluator = MulticlassClassificationEvaluator(labelCol="Enocded_Species", predictionCol="prediction", metricName="accuracy")

# Evaluate model
accuracy = evaluator.evaluate(predictions)
print(f"Accuracy: {accuracy}")

for i,j in enumerate(original_labels):
    predictions=predictions.withColumn("prediction",when(col("prediction")==i,j).otherwise(col("prediction")))
predictions.show()

# H code 
from pyspark.sql import SparkSession
from pyspark.ml.feature import VectorAssembler, PCA
from pyspark.ml.clustering import KMeans
import pandas as pd
import matplotlib.pyplot as plt

# Start Spark session
spark = SparkSession.builder.appName("KMeansPCA").getOrCreate()
df = spark.read.csv("/content/segmentation data.csv", header=True, inferSchema=True)

# Assemble features
assembler = VectorAssembler(inputCols=df.columns, outputCol="features")
df_vector = assembler.transform(df)

# KMeans Clustering
kmeans = KMeans(k=3, seed=42, featuresCol="features")
model = kmeans.fit(df_vector)
clustered = model.transform(df_vector)

# PCA for visualization
pca = PCA(k=2, inputCol="features", outputCol="pca_features")
pca_model = pca.fit(clustered)
pca_result = pca_model.transform(clustered).select("pca_features", "prediction")

# Convert to Pandas for plotting
pandas_df = pca_result.toPandas()
pandas_df[['PC1', 'PC2']] = pandas_df['pca_features'].apply(lambda x: pd.Series(x.toArray()))

# Project cluster centers using PCA components
import numpy as np
centroids = model.clusterCenters()
centroids_np = np.array(centroids)
centroids_pca = centroids_np.dot(pca_model.pc.toArray())  # Project to 2D


# Plot clusters and centroids
plt.figure(figsize=(8, 6))
plt.scatter(pandas_df['PC1'], pandas_df['PC2'], c=pandas_df['prediction'], cmap='viridis', edgecolor='k', alpha=0.7)

# Plot centroids
plt.scatter(centroids_pca[:, 0], centroids_pca[:, 1], c='red', marker='X', s=200, edgecolor='black', linewidths=1.5)

plt.title("KMeans Clustering with PCA (Centroids Highlighted)")
plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.grid(True)
plt.show()

spark.stop()



# S code 
from pyspark.sql import SparkSession
from pyspark.sql.functions import when,col
from pyspark.ml.clustering import KMeans
from pyspark.ml.feature import StandardScaler,VectorAssembler,PCA
import matplotlib.pyplot as plt

spark = SparkSession.builder.appName('Clustering').getOrCreate()
df_spark = spark.read.csv("",header=True,inferSchema=True)
df_spark.columns
features =[
 'Sex',
 'Marital status',
 'Age',
 'Education',
 'Income',
 'Occupation',
 'Settlement size']
assembler = VectorAssembler(inputCols=features,outputCol="vectorized_features")
vectorized_df = assembler.transform(df_spark)
scale = StandardScaler(inputCol="vectorized_features",outputCol="scaled_features")
scaler = scale.fit(vectorized_df)
scaled_df = scaler.transform(vectorized_df)
pca = PCA(k=3, inputCol="scaled_features", outputCol="pca_features")
pca_model = pca.fit(scaled_df)
pca_result = pca_model.transform(scaled_df)

# Create Clusters

cluster = KMeans(featuresCol="pca_features", k=6)
model = cluster.fit(pca_result)
predictions = model.transform(pca_result)
predictions.show(10)

# Plot Clusters

pca_plot = predictions.select(["pca_features","prediction"]).toPandas()
pca_plot['PC1']= pca_plot["pca_features"].apply(lambda x:x[0])
pca_plot["PC2"] = pca_plot["pca_features"].apply(lambda x:x[1])
plt.scatter(pca_plot['PC1'], pca_plot['PC2'], c=pca_plot['prediction'], cmap='viridis', s=50, alpha=0.6)

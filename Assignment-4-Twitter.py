from pyspark.sql import SparkSession 
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.feature import StopWordsRemover,Tokenizer,HashingTF,IDF,StandardScaler 
spark = SparkSession.builder.appName("Twitter").getOrCreate() 
df_spark = spark.read.csv("",header=True,inferSchema=True)  

tokenizer= Tokenizer(inputCol="tweet",outputCol="tokenized_tweet") 
df_tokenized = tokenizer.transform(df_spark)  

stopWordsRemover= StopWordsRemover(inputCol="tokenized_tweet",outputCol="clean_tweet") 
df_clean = stopWordsRemover.transform(df_tokenized)  

hashingTF= HashingTF(inputCol="clean_tweet",outputCol="tf_tweet") 
df_tf = hashingTF.transform(df_clean)  

idf = IDF(inputCol="tf_tweet",outputCol="idf_tweet") 
idf_model = idf.fit(df_tf) 
df_idf = idf_model.transform(df_tf)  

scale = StandardScaler(inputCol="idf_tweet", outputCol="scaled_idf") 
scaler = scale.fit(df_idf) 
df_scaled = scaler.transform(df_idf)  
lr = LogisticRegression(featuresCol="scaled_idf",labelCol="label")
train,test= df_scaled.randomSplit([0.8,0.2])
model =lr.fit(train)
predictions = model.transform(test)
predictions.show()

from pyspark import SparkContext
sc = SparkContext()
text_rdd = sc.textFile("random.txt")
word_count = text_rdd.flatMap(lambda x:x.split(" ")).map(lambda x:(x,1)).reduceByKey(lambda x,y:x+y).collect()
print(word_count)
sc.stop()

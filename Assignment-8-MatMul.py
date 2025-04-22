from pyspark.sql import SparkSession
spark = SparkSession.builder.appName("MatrixMultiplcation").getOrCreate().sparkContext
matrixA = [
    (0,0,1),(0,1,2),
    (1,0,3),(1,1,4)
]
matrixB = [
    (0,0,1),(0,1,2),
    (1,0,3),(1,1,4)
]
rddA = spark.parallelize(matrixA)
rddB = spark.parallelize(matrixB)
mappedA = rddA.map(lambda x:(x[1],(x[0],x[2])))
mappedB = rddB.map(lambda x:(x[0],(x[1],x[2])))
joined = mappedA.join(mappedB)
# Joined will have element like common_key k (k,(rowA,valA),(colB,valB))
partial_products=joined.map(lambda x:((x[1][0][0], x[1][1][0]), x[1][0][1] * x[1][1][1]))
result = partial_products.reduceByKey(lambda x,y:x+y)
output = result.collect()
for ((row, col), value) in sorted(output):
    print(f"({row}, {col}) -> {value}")

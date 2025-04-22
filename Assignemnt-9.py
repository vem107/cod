from pyspark import SparkContext

sc = SparkContext(appName="WeatherAnalysis")

data = [
    (2025, "2025-01-01", "Station_1", 5),
    (2025, "2025-01-02", "Station_1", 8),
    (2025, "2025-01-03", "Station_2", 12),
    (2025, "2025-01-04", "Station_2", 3),
    (2025, "2025-01-05", "Station_1", 10)
]

rdd = sc.parallelize(data)
rdd_with_key = rdd.map(lambda x: ((x[2], x[0]), (x[1], x[3])))
def find_max_snowfall(a, b):
    return a if a[1] > b[1] else b

max_snowfall_rdd = rdd_with_key.reduceByKey(find_max_snowfall)
result = max_snowfall_rdd.collect()

for key, value in result:
    print(f"Station: {key[0]}, Year: {key[1]}, Day: {value[0]}, Max Snowfall: {value[1]} inches")

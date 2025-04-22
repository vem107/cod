# h code 
from pyspark import SparkContext
import kagglehub

def parse_line(line):
    """Parses each line of input data into (movie_id, rating)."""
    if line.startswith("userId,movieId,rating,timestamp"):
        return None
    parts = line.split(",")
    return (int(parts[1]), float(parts[2]))

def main():
    sc = SparkContext("local", "MovieRatings")

    # Download dataset
    path = kagglehub.dataset_download("rounakbanik/the-movies-dataset")
    dataset_file = f"{path}/ratings.csv"

    # Read the input data
    input_rdd = sc.textFile(dataset_file)

    # Parse and filter the data
    mapped_rdd = input_rdd.filter(lambda line: not line.startswith("userId,movieId,rating,timestamp")) \
                          .map(parse_line)

    # Calculate average ratings for each movie
    reduced_rdd = mapped_rdd.groupByKey().mapValues(lambda ratings: sum(ratings) / len(ratings))

    # Collect results and print
    results = reduced_rdd.collect()
    for movie_id, avg_rating in results:
        print(f"Movie {movie_id} has an average rating of {avg_rating:.2f}")

    sc.stop()

if __name__ == "__main__":
    main()




# S code 
from pyspark import SparkContext

sc = SparkContext(appName="MovieRatingAnalysis")

data = [
    (1, 1, 3.5, 1612300000),
    (1, 2, 4.0, 1612301000),
    (1, 3, 3.0, 1612302000),
    (2, 1, 5.0, 1612303000),
    (2, 2, 4.5, 1612304000),
    (3, 1, 2.5, 1612305000)
]

rdd = sc.parallelize(data)
rdd_with_key = rdd.map(lambda x: (x[0], (x[2], 1)))
def sum_ratings(a, b):
    return (a[0] + b[0], a[1] + b[1])
ratings_sum_rdd = rdd_with_key.reduceByKey(sum_ratings)
average_ratings_rdd = ratings_sum_rdd.mapValues(lambda x: x[0] / x[1])
result = average_ratings_rdd.collect()
for movie_id, avg_rating in result:
    print(f"Movie ID: {movie_id}, Average Rating: {avg_rating:.2f}")

import os
import requests
import pandas as pd
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, month, dayofyear, max as spark_max, when, expr
from pyspark.ml.regression import LinearRegression
from pyspark.ml.feature import VectorAssembler

# Define the base URLs for the data
base_url_1 = "https://www.ncei.noaa.gov/data/global-summary-of-the-day/access/{}/99495199999.csv"
base_url_2 = "https://www.ncei.noaa.gov/data/global-summary-of-the-day/access/{}/72429793812.csv"

# Define the range of years
years = range(2021, 2023)

# Base directory to save the downloaded files
base_output_dir = "./weather_data/"

# Download data for each year and station
for year in years:
    year_dir = os.path.join(base_output_dir, str(year))
    os.makedirs(year_dir, exist_ok=True)

    for base_url, station_id in [(base_url_1, "99495199999"), (base_url_2, "72429793812")]:
        url = base_url.format(year)
        response = requests.get(url)

        if response.status_code == 200:
            file_path = os.path.join(year_dir, f"{station_id}.csv")
            with open(file_path, "wb") as file:
                file.write(response.content)
            print(f"Downloaded: {file_path}")
        else:
            print(f"Failed to download {url}. Status code: {response.status_code}")

# Clean the data by filtering out invalid values
base_input_dir = "./weather_data/"
base_output_dir = "./cleaned_weather_data/"

invalid_values = {
    "MXSPD": 999.9,
    "MAX": 9999.9,
    "MIN": 9999.9,
}

for year in range(2021, 2023):
    year_dir = os.path.join(base_input_dir, str(year))

    if os.path.exists(year_dir):
        for station_id in ["99495199999", "72429793812"]:
            file_path = os.path.join(year_dir, f"{station_id}.csv")
            if os.path.exists(file_path):
                df = pd.read_csv(file_path)
                for column, invalid_value in invalid_values.items():
                    df = df[df[column] != invalid_value]

                output_year_dir = os.path.join(base_output_dir, str(year))
                os.makedirs(output_year_dir, exist_ok=True)
                cleaned_file_path = os.path.join(output_year_dir, f"{station_id}.csv")
                df.to_csv(cleaned_file_path, index=False)
                print(f"Cleaned data saved to: {cleaned_file_path}")
            else:
                print(f"File not found: {file_path}")
    else:
        print(f"Year directory not found: {year_dir}")

# Spark session setup
spark = SparkSession.builder.appName("Weather Analysis").getOrCreate()

# Question 2: Load the CSV files and display the count of each dataset.
for year in range(2021, 2023):
    for station_id in ["99495199999", "72429793812"]:
        file_path = os.path.join(base_output_dir, str(year), f"{station_id}.csv")
        if os.path.exists(file_path):
            df = spark.read.csv(file_path, header=True, inferSchema=True)
            print(f"Year: {year}, Station: {station_id}, Row count: {df.count()}")

# Question 6: Count missing values for 'GUST' in 2024
base_path = "./cleaned_weather_data/2024/"
station_codes = ['99495199999', '72429793812']
results = []

for station_code in station_codes:
    file_path = os.path.join(base_path, f"{station_code}.csv")
    if os.path.exists(file_path):
        df = spark.read.csv(file_path, header=True, inferSchema=True)
        total_count = df.count()
        missing_count = df.filter(df.GUST == 999.9).count()

        missing_percentage = (missing_count / total_count) * 100 if total_count > 0 else 0.0
        results.append((station_code, missing_percentage))

for station_code, missing_percentage in results:
    print(f"Station Code: {station_code}, Missing GUST Percentage in 2024: {missing_percentage:.2f}%")

# Question 7: Mean, median, mode, and standard deviation for Cincinnati temperature in 2020
df = spark.read.csv("./cleaned_weather_data/2021/72429793812.csv", header=True, inferSchema=True)
df_cincinnati = df.withColumn("MONTH", month(col("DATE")))
result = df_cincinnati.groupBy("MONTH").agg(
    expr("mean(TEMP)").alias("Mean"),
    expr("percentile_approx(TEMP, 0.5)").alias("Median"),
    expr("mode(TEMP)").alias("Mode"),
    expr("stddev(TEMP)").alias("Standard Deviation")
)

result.orderBy("MONTH").show()

# Question 8: Top 10 lowest Wind Chill days in Cincinnati 2017
df = spark.read.csv("./cleaned_weather_data/2022/72429793812.csv", header=True, inferSchema=True)
df_cincinnati = df.filter((col("TEMP") < 50) & (col("WDSP") > 3))

df_cincinnati = df_cincinnati.withColumn(
    "Wind Chill",
    35.74 + (0.6215 * col("TEMP")) - (35.75 * (col("WDSP") ** 0.16)) + (0.4275 * col("TEMP") * (col("WDSP") ** 0.16))
)

df_cincinnati = df_cincinnati.withColumn("DATE", expr("date_format(DATE, 'yyyy-MM-dd')"))
result = df_cincinnati.select("DATE", "Wind Chill").orderBy("Wind Chill").limit(10)
result.show()

# Question 9: Count extreme weather days for Florida
base_directory = './cleaned_weather_data/'
file_paths = []
for year in range(2015, 2025):
    file_path = os.path.join(base_directory, str(year), '99495199999.csv')
    if os.path.exists(file_path):
        file_paths.append(file_path)

df = spark.read.csv(file_paths, header=True, inferSchema=True)
extreme_weather_count = df.filter(col("FRSHTT") != 0).count()
print(f"Number of days with extreme weather conditions in Florida: {extreme_weather_count}")

# Question 10: Predict max Temperature for Cincinnati in Nov and Dec 2024
base_directory = './cleaned_weather_data'
file_paths = []

for year in [2022, 2023]:
    file_path = os.path.join(base_directory, str(year), '72429793812.csv')
    if os.path.exists(file_path):
        file_paths.append(file_path)

historical_data = spark.read.csv(file_paths, header=True, inferSchema=True)
historical_df = historical_data.filter(
    (col("STATION") == "72429793812") & (month("DATE").isin([11, 12]))
)

training_data = historical_df.withColumn("DAY_OF_YEAR", dayofyear("DATE"))
assembler = VectorAssembler(inputCols=["DAY_OF_YEAR"], outputCol="features")
train_data = assembler.transform(training_data).select("features", col("MAX").alias("label"))

lr = LinearRegression()
lr_model = lr.fit(train_data)

predictions_df = spark.createDataFrame([(day,) for day in range(305, 366)], ["DAY_OF_YEAR"])
predictions = assembler.transform(predictions_df)
predicted_temps = lr_model.transform(predictions)

max_predictions = predicted_temps.withColumn(
    "MONTH", when(col("DAY_OF_YEAR") < 335, 11).otherwise(12)
).groupBy("MONTH").agg(spark_max("prediction").alias("Max Predicted Temp"))

max_predictions.show()

# Stop the Spark session
spark.stop()

from pyspark.sql import SparkSession
from pyspark.ml.recommendation import ALS
from pyspark.sql.functions import col, collect_list, explode

spark = SparkSession.builder.appName("ALS_Recommender").getOrCreate()

# โหลดข้อมูล
ratings = spark.read.csv("ratings.csv", header=True, inferSchema=True)
movies = spark.read.csv("movies.csv", header=True, inferSchema=True)

# Train Model
als = ALS(maxIter=10, regParam=0.1, userCol="userId", itemCol="movieId", ratingCol="rating", coldStartStrategy="drop")
model = als.fit(ratings)

# 1. บันทึกผล Batch (สำหรับส่วนที่ 1 ของ UI)
user_recs = model.recommendForAllUsers(10)
recs_exploded = user_recs.withColumn("rec", explode("recommendations")).select("userId", col("rec.movieId").alias("movieId"))
recs_with_titles = recs_exploded.join(movies, on="movieId", how="left").select("userId", "title")
final_df = recs_with_titles.groupBy("userId").agg(collect_list("title").alias("recommended_movies"))

final_df.withColumn("recommended_movies", col("recommended_movies").cast("string")) \
        .coalesce(1).write.mode("overwrite").csv("/app/output/final_recommend", header=True)

movies.select("movieId", "title").coalesce(1).write.mode("overwrite").csv("/app/output/movie_list", header=True)

# 2. บันทึก Item Factors (สำคัญมาก! สำหรับส่วนที่ 2 ของ UI)
# ต้องบันทึกก่อน spark.stop()
model.itemFactors.write.mode("overwrite").parquet("/app/model/als_model/itemFactors")

# 3. บันทึก Model เต็ม
model.write().overwrite().save("/app/model/als_model")

spark.stop() # ย้ายมาบรรทัดสุดท้าย
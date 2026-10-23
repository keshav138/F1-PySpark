"""01_rdd_sql.py - RDD operations (Unit III) and Spark SQL aggregates (Unit IV).

Run from the f1_project folder:
    spark-submit 01_rdd_sql.py
or inside the pyspark shell:
    exec(open("01_rdd_sql.py").read())
"""
import os
import pandas as pd
from common import get_spark, load_csv, OUT_DIR

def banner(text):
    print("\n" + "=" * 70 + f"\n{text}\n" + "=" * 70)

if __name__ == "__main__" or "spark" in globals():
    spark = get_spark("F1-RDD-SQL")
    sc = spark.sparkContext

    banner("1. Loading Data")
    races = load_csv(spark, "races")
    results = load_csv(spark, "results")
    drivers = load_csv(spark, "drivers")
    constructors = load_csv(spark, "constructors")
    pit_stops = load_csv(spark, "pit_stops")

    # Register Temp Views for Spark SQL
    races.createOrReplaceTempView("races")
    results.createOrReplaceTempView("results")
    drivers.createOrReplaceTempView("drivers")
    constructors.createOrReplaceTempView("constructors")
    pit_stops.createOrReplaceTempView("pit_stops")
    print("DataFrames loaded and SQL views registered.")

    banner("2. RDD Operations (Unit III)")
    # 2.1 Broadcast Variable: Map driverId to surname for quick lookups
    driver_map = drivers.select("driverId", "surname").rdd.map(
        lambda r: (r.driverId, r.surname)
    ).collectAsMap()
    broadcast_drivers = sc.broadcast(driver_map)

    # 2.2 Accumulator: Count total podiums across history
    podium_acc = sc.accumulator(0)

    def extract_driver_points(row):
        # Update accumulator if it's a podium finish (positions 1, 2, or 3)
        if row.positionOrder is not None and int(row.positionOrder) <= 3:
            podium_acc.add(1)
        
        points = float(row.points) if row.points else 0.0
        return (row.driverId, points)

    # 2.3 RDD map, filter, and reduceByKey
    rdd = results.rdd.map(extract_driver_points)
    
    # Calculate career points and filter out drivers with 0 points
    career_points_rdd = (rdd
                         .reduceByKey(lambda a, b: a + b)
                         .filter(lambda x: x[1] > 0))
    
    # Map driverId to driver name using the broadcast variable
    named_points_rdd = career_points_rdd.map(
        lambda x: (broadcast_drivers.value.get(x[0], "Unknown"), x[1])
    )
    
    top_5 = named_points_rdd.sortBy(lambda x: x[1], ascending=False).take(5)
    print("Top 5 Drivers by Career Points (calculated via RDD reduceByKey):")
    for driver, points in top_5:
        print(f"  {driver}: {points} pts")
    
    print(f"\nTotal podiums in F1 history (counted via RDD Accumulator): {podium_acc.value}")


    banner("3. Spark SQL Operations (Unit IV) & CSV Exports")

    # Query 1: Wins per driver
    q1 = """
        SELECT d.forename || ' ' || d.surname AS driver, COUNT(*) AS wins
        FROM results r
        JOIN drivers d ON r.driverId = d.driverId
        WHERE r.positionOrder = '1'
        GROUP BY driver
        ORDER BY wins DESC
        LIMIT 20
    """
    wins_df = spark.sql(q1).toPandas()
    wins_df.to_csv(os.path.join(OUT_DIR, "sql_wins.csv"), index=False)
    print("Exported: sql_wins.csv")

    # Query 2: Fastest laps per driver
    q2 = """
        SELECT d.forename || ' ' || d.surname AS driver, COUNT(*) AS fastest_laps
        FROM results r
        JOIN drivers d ON r.driverId = d.driverId
        WHERE r.rank = '1'
        GROUP BY driver
        ORDER BY fastest_laps DESC
        LIMIT 20
    """
    fl_df = spark.sql(q2).toPandas()
    fl_df.to_csv(os.path.join(OUT_DIR, "sql_fastest_laps.csv"), index=False)
    print("Exported: sql_fastest_laps.csv")

    # Query 3: Constructor points per season (Modern Era)
    q3 = """
        SELECT r.year, c.name AS constructor, SUM(CAST(rs.points AS FLOAT)) AS total_points
        FROM results rs
        JOIN races r ON rs.raceId = r.raceId
        JOIN constructors c ON rs.constructorId = c.constructorId
        WHERE r.year >= 2000
        GROUP BY r.year, constructor
        ORDER BY r.year DESC, total_points DESC
    """
    cons_df = spark.sql(q3).toPandas()
    cons_df.to_csv(os.path.join(OUT_DIR, "sql_constructor_points.csv"), index=False)
    print("Exported: sql_constructor_points.csv")

    # Query 4: Average Finish by Grid Position
    q4 = """
        SELECT CAST(grid AS INT) AS grid_position, AVG(CAST(positionOrder AS FLOAT)) AS avg_finish
        FROM results
        WHERE grid > 0 AND grid <= 20
        GROUP BY grid_position
        ORDER BY grid_position
    """
    grid_df = spark.sql(q4).toPandas()
    grid_df.to_csv(os.path.join(OUT_DIR, "sql_grid_vs_finish.csv"), index=False)
    print("Exported: sql_grid_vs_finish.csv")

    # Query 5: Average Pit Stop Time per Constructor (filtering out extreme anomalies > 50s)
    q5 = """
        SELECT c.name AS constructor, AVG(CAST(p.milliseconds AS FLOAT))/1000 AS avg_pit_time_sec, COUNT(*) as pit_count
        FROM pit_stops p
        JOIN results r ON p.raceId = r.raceId AND p.driverId = r.driverId
        JOIN constructors c ON r.constructorId = c.constructorId
        WHERE p.milliseconds < 50000 
        GROUP BY constructor
        HAVING pit_count > 50
        ORDER BY avg_pit_time_sec ASC
        LIMIT 10
    """
    pit_df = spark.sql(q5).toPandas()
    pit_df.to_csv(os.path.join(OUT_DIR, "sql_avg_pit_stops.csv"), index=False)
    print("Exported: sql_avg_pit_stops.csv")

    banner(f"Done. All aggregations prepared for the dashboard in {OUT_DIR}/")
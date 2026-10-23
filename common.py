"""common.py - shared helpers for the F1 PySpark project."""
import os
from pyspark.sql import SparkSession

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
OUT_DIR = os.path.join(BASE_DIR, "output")
os.makedirs(OUT_DIR, exist_ok=True)


def get_spark(app_name="F1-Cluster-Project"):
    """Return the active SparkSession (reuses the pyspark shell's session if present)."""
    spark = (
        SparkSession.builder.appName(app_name)
        .master("local[*]")
        .config("spark.sql.shuffle.partitions", "8")
        .config("spark.driver.memory", "4g")
        .config("spark.ui.showConsoleProgress", "false")
        .getOrCreate()
    )
    spark.sparkContext.setLogLevel("WARN")
    return spark


def load_csv(spark, name):
    """Load data/<name>.csv. The Ergast data encodes missing values as \\N."""
    path = os.path.join(DATA_DIR, f"{name}.csv")
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing {path} - check the data folder.")
    return spark.read.csv(path, header=True, inferSchema=True, nullValue="\\N")
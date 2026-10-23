"""02_ml.py - Feature engineering, Spark ML models, evaluation and export.

Run from the f1_project folder:
    spark-submit 02_ml.py
or inside the pyspark shell:
    exec(open("02_ml.py").read())
"""
import os
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from pyspark.sql import functions as F, Window
from pyspark.ml import Pipeline
from pyspark.ml.feature import VectorAssembler, StandardScaler
from pyspark.ml.classification import (
    LogisticRegression, DecisionTreeClassifier, NaiveBayes, LinearSVC,
)
from pyspark.ml.regression import LinearRegression
from pyspark.ml.clustering import KMeans
from pyspark.ml.evaluation import (
    MulticlassClassificationEvaluator, BinaryClassificationEvaluator,
    RegressionEvaluator, ClusteringEvaluator,
)
from pyspark.ml.stat import Correlation

from common import get_spark, load_csv, OUT_DIR

# ----------------------------- configuration -----------------------------
MIN_YEAR = 2000          # modern-era races only
TRAIN_END_YEAR = 2017    # train: <= 2017, test: > 2017 (time-based split)
FEATURES = [
    "grid_pos",            # starting grid position (pit-lane start = 22)
    "quali_pos",           # qualifying position (falls back to grid)
    "driver_age",          # age in years on race day
    "driver_podium_rate",  # driver's career podium rate BEFORE this race
    "driver_form",         # driver's avg points over previous 5 races
    "constructor_form",    # team's avg points per car over previous 10 races
]


def banner(text):
    print("\n" + "=" * 70 + f"\n{text}\n" + "=" * 70)


# --------------------------- feature engineering ---------------------------
def build_dataset(spark):
    results = load_csv(spark, "results").select(
        "raceId", "driverId", "constructorId", "grid", "positionOrder", "points")
    races = load_csv(spark, "races").select("raceId", "year", "round", F.to_date("date").alias("date"))
    drivers = load_csv(spark, "drivers").select(
        "driverId", "forename", "surname", F.to_date("dob").alias("dob"))
    quali = load_csv(spark, "qualifying").select(
        "raceId", "driverId", F.col("position").alias("quali_raw"))

    # Chronological race index so window functions can look back N races
    races = races.withColumn("race_idx", F.dense_rank().over(Window.orderBy("year", "round")))

    df = (results.join(races, "raceId")
          .join(drivers, "driverId")
          .join(quali, ["raceId", "driverId"], "left"))

    df = (df
          .withColumn("grid_pos", F.when(F.col("grid") == 0, 22).otherwise(F.col("grid")).cast("double"))
          .withColumn("quali_pos", F.coalesce(F.col("quali_raw"), F.col("grid_pos")).cast("double"))
          .withColumn("driver_age", F.datediff("date", "dob") / 365.25)
          .withColumn("is_podium", (F.col("positionOrder") <= 3).cast("double")))

    # Rolling features use only PRE-race information (no leakage)
    w_driver_career = (Window.partitionBy("driverId").orderBy("race_idx")
                       .rowsBetween(Window.unboundedPreceding, -1))
    w_driver_form = Window.partitionBy("driverId").orderBy("race_idx").rangeBetween(-5, -1)
    w_team_form = Window.partitionBy("constructorId").orderBy("race_idx").rangeBetween(-10, -1)

    df = (df
          .withColumn("driver_podium_rate", F.avg("is_podium").over(w_driver_career))
          .withColumn("driver_form", F.avg("points").over(w_driver_form))
          .withColumn("constructor_form", F.avg("points").over(w_team_form)))

    # Filter AFTER the windows so early-2000s rows still have history
    df = (df.filter(F.col("year") >= MIN_YEAR)
          .fillna({"driver_podium_rate": 0.0, "driver_form": 0.0, "constructor_form": 0.0})
          .dropna(subset=["driver_age"])
          .withColumn("label", F.col("is_podium")))
    return df


# ------------------------------ correlation ------------------------------
def correlation_heatmap(df):
    cols = FEATURES + ["positionOrder"]
    vec = VectorAssembler(inputCols=cols, outputCol="v").transform(df).select("v")
    mat = Correlation.corr(vec, "v").head()[0].toArray()
    cdf = pd.DataFrame(mat, index=cols, columns=cols)
    cdf.to_csv(os.path.join(OUT_DIR, "correlation_matrix.csv"))
    plt.figure(figsize=(9, 7))
    sns.heatmap(cdf, annot=True, fmt=".2f", cmap="coolwarm", vmin=-1, vmax=1)
    plt.title("Feature Correlation Heatmap")
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "correlation_heatmap.png"), dpi=150)
    plt.close()
    print(cdf.round(2))


# ----------------------------- classification -----------------------------
def evaluate_classifier(name, model, test):
    pred = model.transform(test).cache()
    acc = MulticlassClassificationEvaluator(metricName="accuracy").evaluate(pred)
    f1_weighted = MulticlassClassificationEvaluator(metricName="f1").evaluate(pred)
    auc = BinaryClassificationEvaluator(metricName="areaUnderROC").evaluate(pred)

    counts = {(r["label"], r["prediction"]): r["count"]
              for r in pred.groupBy("label", "prediction").count().collect()}
    tp = counts.get((1.0, 1.0), 0)
    fp = counts.get((0.0, 1.0), 0)
    fn = counts.get((1.0, 0.0), 0)
    tn = counts.get((0.0, 0.0), 0)
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    podium_f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0

    print(f"\n[{name}]  accuracy={acc:.4f}  weighted_F1={f1_weighted:.4f}  "
          f"podium_F1={podium_f1:.4f}  AUC={auc:.4f}")
    print("Confusion matrix (rows = actual, cols = predicted)")
    print(f"              pred_0   pred_1\n  actual_0   {tn:7d}  {fp:7d}\n  actual_1   {fn:7d}  {tp:7d}")
    pred.unpersist()
    return {"model": name, "accuracy": acc, "weighted_f1": f1_weighted,
            "podium_precision": precision, "podium_recall": recall,
            "podium_f1": podium_f1, "auc": auc, "tn": tn, "fp": fp, "fn": fn, "tp": tp}


def run_classification(train, test):
    pos = train.filter("label = 1").count()
    neg = train.count() - pos
    train = train.withColumn("weight", F.when(F.col("label") == 1, neg / pos).otherwise(1.0))
    print(f"Train podium rows: {pos} | non-podium rows: {neg} (class weight {neg / pos:.2f} applied)")

    assembler = VectorAssembler(inputCols=FEATURES, outputCol="raw_features")
    scaler = StandardScaler(inputCol="raw_features", outputCol="features", withMean=False, withStd=True)
    kw = dict(featuresCol="features", labelCol="label", weightCol="weight")
    classifiers = {
        "Logistic Regression": LogisticRegression(maxIter=100, **kw),
        "Decision Tree": DecisionTreeClassifier(maxDepth=6, seed=42, **kw),
        "Naive Bayes": NaiveBayes(modelType="gaussian", **kw),
        "Linear SVM": LinearSVC(maxIter=100, **kw),
    }

    fitted, rows = {}, []
    for name, clf in classifiers.items():
        model = Pipeline(stages=[assembler, scaler, clf]).fit(train)
        fitted[name] = model
        rows.append(evaluate_classifier(name, model, test))

    metrics = pd.DataFrame(rows)
    metrics.to_csv(os.path.join(OUT_DIR, "classification_metrics.csv"), index=False)

    # Dashboard needs a probability output, so Linear SVM is excluded from this choice
    candidates = metrics[metrics["model"] != "Linear SVM"]
    best_name = candidates.sort_values("podium_f1", ascending=False).iloc[0]["model"]
    path = os.path.join(OUT_DIR, "best_model")
    fitted[best_name].write().overwrite().save(path)
    with open(os.path.join(OUT_DIR, "best_model.txt"), "w") as f:
        f.write(best_name)
    print(f"\nBest probabilistic model for the dashboard: {best_name} (saved to {path})")
    return metrics


# ------------------------------- regression -------------------------------
def run_regression(train, test):
    assembler = VectorAssembler(inputCols=FEATURES, outputCol="raw_features")
    scaler = StandardScaler(inputCol="raw_features", outputCol="features", withMean=False, withStd=True)
    lr = LinearRegression(featuresCol="features", labelCol="positionOrder", maxIter=100)
    model = Pipeline(stages=[assembler, scaler, lr]).fit(train)
    pred = model.transform(test)

    out = {}
    for m in ["r2", "rmse", "mae"]:
        out[m] = RegressionEvaluator(labelCol="positionOrder", predictionCol="prediction",
                                     metricName=m).evaluate(pred)
    print(f"Linear Regression (finishing position): R2={out['r2']:.4f}  "
          f"RMSE={out['rmse']:.4f}  MAE={out['mae']:.4f}")
    pd.DataFrame([out]).to_csv(os.path.join(OUT_DIR, "regression_metrics.csv"), index=False)
    model.write().overwrite().save(os.path.join(OUT_DIR, "regression_model"))
    return out


# -------------------------------- clustering --------------------------------
def run_clustering(df, k=4):
    profile = (df.groupBy("driverId", "forename", "surname").agg(
        F.count("*").alias("races"),
        F.avg("grid_pos").alias("avg_grid"),
        F.avg("positionOrder").alias("avg_finish"),
        F.avg("is_podium").alias("podium_rate"),
        F.avg(F.when(F.col("positionOrder") == 1, 1.0).otherwise(0.0)).alias("win_rate"),
        F.avg("points").alias("avg_points"),
    ).filter("races >= 30"))

    cols = ["avg_grid", "avg_finish", "podium_rate", "win_rate", "avg_points"]
    vec = VectorAssembler(inputCols=cols, outputCol="raw").transform(profile)
    scaled = StandardScaler(inputCol="raw", outputCol="features", withMean=True, withStd=True) \
        .fit(vec).transform(vec)
    model = KMeans(k=k, seed=42, featuresCol="features").fit(scaled)
    clustered = model.transform(scaled)
    sil = ClusteringEvaluator(featuresCol="features").evaluate(clustered)
    print(f"K-means (k={k}) silhouette score: {sil:.4f}")

    out = clustered.select("forename", "surname", "races", *cols,
                           F.col("prediction").alias("cluster")).toPandas()
    out["driver"] = out["forename"] + " " + out["surname"]
    out.drop(columns=["forename", "surname"]).to_csv(os.path.join(OUT_DIR, "driver_clusters.csv"), index=False)
    summary = out.groupby("cluster")[cols + ["races"]].mean().round(3)
    summary["drivers"] = out.groupby("cluster").size()
    summary.to_csv(os.path.join(OUT_DIR, "cluster_summary.csv"))
    print(summary)
    return sil


# ----------------------------------- main -----------------------------------
if __name__ == "__main__" or "spark" in globals():
    spark = get_spark("F1-ML")

    banner("1. Building feature dataset")
    data = build_dataset(spark).cache()
    print(f"Rows: {data.count()} | Seasons: {MIN_YEAR}+")
    data.select("year", "forename", "surname", *FEATURES, "positionOrder", "label").show(10, truncate=False)

    train = data.filter(F.col("year") <= TRAIN_END_YEAR).cache()
    test = data.filter(F.col("year") > TRAIN_END_YEAR).cache()
    print(f"Train rows: {train.count()} (<= {TRAIN_END_YEAR}) | Test rows: {test.count()} (> {TRAIN_END_YEAR})")

    banner("2. Correlation heatmap")
    correlation_heatmap(data)

    banner("3. Classification: podium prediction")
    cls_metrics = run_classification(train, test)
    print("\nSummary:")
    print(cls_metrics[["model", "accuracy", "weighted_f1", "podium_f1", "auc"]].round(4).to_string(index=False))

    banner("4. Regression: finishing position")
    run_regression(train, test)

    banner("5. K-means: driver clustering")
    run_clustering(data)

    banner(f"Done. Outputs saved in {OUT_DIR}")
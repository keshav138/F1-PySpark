"""03_laps_ml.py - Lap-level machine learning on lap_times (~570k rows).

 A) Live podium predictor : P(driver finishes top 3) from the race state after each lap
 B) Lap-time regression   : predict a driver's next lap time (ms)

Run from the f1_project folder:
    spark-submit 03_laps_ml.py
or inside the pyspark shell:
    exec(open("03_laps_ml.py").read())
"""
import os
import pandas as pd

from pyspark.sql import functions as F, Window
from pyspark.ml import Pipeline
from pyspark.ml.feature import VectorAssembler, StandardScaler
from pyspark.ml.classification import (
    LogisticRegression, DecisionTreeClassifier, RandomForestClassifier,
)
from pyspark.ml.regression import LinearRegression, RandomForestRegressor
from pyspark.ml.evaluation import (
    MulticlassClassificationEvaluator, BinaryClassificationEvaluator, RegressionEvaluator,
)

from common import get_spark, load_csv, OUT_DIR

# ----------------------------- configuration -----------------------------
MIN_YEAR = 1996          # lap_times coverage starts around here
TRAIN_END_YEAR = 2017    # train: <= 2017, test: > 2017 (time-based split)
CLEAN_LAP_FACTOR = 1.25  # regression keeps laps <= 1.25 x race median (drops pit/SC laps)

LIVE_FEATURES = [
    "lap_frac",           # share of the race completed (lap / total laps)
    "lap",                # current lap number
    "position",           # running position at the end of this lap
    "grid_pos",           # starting grid position (pit-lane start = 22)
    "pos_change",         # places gained (+) or lost (-) vs the grid
    "recent_pace_delta",  # avg of last 5 laps vs field median (ms, clipped +-5000; negative = faster)
    "pits_so_far",        # pit stops made so far
]
LAP_FEATURES = [
    "prev_ms",            # previous lap time
    "prev3_avg_ms",       # average of the previous 3 laps
    "lap",
    "lap_frac",
    "laps_since_pit",     # laps since the last stop (tyre-age proxy)
    "prev_position",      # position at the end of the previous lap
]


def banner(text):
    print("\n" + "=" * 70 + f"\n{text}\n" + "=" * 70)


# --------------------------- feature engineering ---------------------------
def build_lap_dataset(spark):
    laps = load_csv(spark, "lap_times").select(
        "raceId", "driverId", "lap", "position", F.col("milliseconds").cast("double").alias("ms"))
    races = load_csv(spark, "races").select("raceId", "year")
    results = load_csv(spark, "results").select("raceId", "driverId", "grid", "positionOrder")
    pits = (load_csv(spark, "pit_stops")
            .groupBy("raceId", "driverId", "lap").agg(F.count("*").alias("pit_n")))

    total = laps.groupBy("raceId").agg(F.max("lap").alias("total_laps"))
    field = laps.groupBy("raceId", "lap").agg(F.percentile_approx("ms", 0.5).alias("field_med"))
    race_med = laps.groupBy("raceId").agg(F.percentile_approx("ms", 0.5).alias("race_med"))

    df = (laps.join(races, "raceId").filter(F.col("year") >= MIN_YEAR)
          .join(results, ["raceId", "driverId"])
          .join(total, "raceId").join(field, ["raceId", "lap"]).join(race_med, "raceId")
          .join(pits, ["raceId", "driverId", "lap"], "left")
          .fillna({"pit_n": 0}))

    w = Window.partitionBy("raceId", "driverId").orderBy("lap")
    w_cum = w.rowsBetween(Window.unboundedPreceding, 0)
    w_prev = w.rowsBetween(Window.unboundedPreceding, -1)
    w_recent = w.rowsBetween(-4, 0)
    w_prev3 = w.rowsBetween(-3, -1)

    df = (df
          .withColumn("grid_pos", F.when(F.col("grid") == 0, 22).otherwise(F.col("grid")).cast("double"))
          .withColumn("lap_frac", F.col("lap") / F.col("total_laps"))
          .withColumn("pos_change", F.col("grid_pos") - F.col("position"))
          .withColumn("rel_ms", F.least(F.greatest(F.col("ms") - F.col("field_med"), F.lit(-5000.0)),
                                        F.lit(5000.0)))
          .withColumn("recent_pace_delta", F.avg("rel_ms").over(w_recent))
          .withColumn("pits_so_far", F.sum("pit_n").over(w_cum))
          .withColumn("pit_marker", F.when(F.col("pit_n") > 0, F.col("lap")))
          .withColumn("last_pit_prev", F.max("pit_marker").over(w_prev))
          .withColumn("laps_since_pit", F.col("lap") - F.coalesce(F.col("last_pit_prev"), F.lit(0)))
          .withColumn("prev_ms", F.lag("ms", 1).over(w))
          .withColumn("prev3_avg_ms", F.avg("ms").over(w_prev3))
          .withColumn("prev_position", F.lag("position", 1).over(w))
          .withColumn("label", (F.col("positionOrder") <= 3).cast("double")))
    return df


# ------------------------------ A) live podium ------------------------------
def clf_metrics(pred):
    acc = MulticlassClassificationEvaluator(metricName="accuracy").evaluate(pred)
    auc = BinaryClassificationEvaluator(metricName="areaUnderROC").evaluate(pred)
    counts = {(r["label"], r["prediction"]): r["count"]
              for r in pred.groupBy("label", "prediction").count().collect()}
    tp, fp = counts.get((1.0, 1.0), 0), counts.get((0.0, 1.0), 0)
    fn, tn = counts.get((1.0, 0.0), 0), counts.get((0.0, 0.0), 0)
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return {"accuracy": acc, "podium_precision": precision, "podium_recall": recall,
            "podium_f1": f1, "auc": auc, "tn": tn, "fp": fp, "fn": fn, "tp": tp}


def run_live_podium(train, test):
    pos = train.filter("label = 1").count()
    neg = train.count() - pos
    train = train.withColumn("weight", F.when(F.col("label") == 1, neg / pos).otherwise(1.0))
    print(f"Train rows: podium={pos}, non-podium={neg} (class weight {neg / pos:.2f})")

    assembler = VectorAssembler(inputCols=LIVE_FEATURES, outputCol="raw_features")
    scaler = StandardScaler(inputCol="raw_features", outputCol="features", withMean=False, withStd=True)
    kw = dict(featuresCol="features", labelCol="label", weightCol="weight")
    classifiers = {
        "Logistic Regression": LogisticRegression(maxIter=100, **kw),
        "Decision Tree": DecisionTreeClassifier(maxDepth=8, seed=42, **kw),
        "Random Forest": RandomForestClassifier(numTrees=30, maxDepth=8, seed=42, **kw),
    }

    fitted, preds, rows = {}, {}, []
    for name, clf in classifiers.items():
        model = Pipeline(stages=[assembler, scaler, clf]).fit(train)
        pred = model.transform(test).cache()
        m = clf_metrics(pred)
        m["model"] = name
        fitted[name], preds[name] = model, pred
        rows.append(m)
        print(f"\n[{name}] accuracy={m['accuracy']:.4f}  podium_F1={m['podium_f1']:.4f}  AUC={m['auc']:.4f}")
        print("Confusion matrix (rows = actual, cols = predicted)")
        print(f"              pred_0   pred_1\n  actual_0   {m['tn']:7d}  {m['fp']:7d}\n"
              f"  actual_1   {m['fn']:7d}  {m['tp']:7d}")

    metrics = pd.DataFrame(rows)
    metrics.to_csv(os.path.join(OUT_DIR, "live_classification_metrics.csv"), index=False)
    best = metrics.sort_values("podium_f1", ascending=False).iloc[0]["model"]

    # Accuracy by race stage for the best model (late laps are easier: position nearly decides it)
    staged = preds[best].withColumn(
        "stage",
        F.when(F.col("lap_frac") < 1 / 3, "1_early (0-33%)")
        .when(F.col("lap_frac") < 2 / 3, "2_mid (33-66%)")
        .otherwise("3_late (66-100%)"))
    stage_rows = []
    for s in ["1_early (0-33%)", "2_mid (33-66%)", "3_late (66-100%)"]:
        m = clf_metrics(staged.filter(F.col("stage") == s))
        m["stage"] = s
        stage_rows.append(m)
    stages = pd.DataFrame(stage_rows)
    stages.to_csv(os.path.join(OUT_DIR, "live_stage_metrics.csv"), index=False)
    print(f"\nBest model: {best}. Performance by race stage:")
    print(stages[["stage", "accuracy", "podium_f1", "auc"]].round(4).to_string(index=False))

    fitted[best].write().overwrite().save(os.path.join(OUT_DIR, "live_model"))
    with open(os.path.join(OUT_DIR, "live_model.txt"), "w") as f:
        f.write(best)
    print(f"Saved to {os.path.join(OUT_DIR, 'live_model')}")
    return metrics


# ------------------------------ B) lap time ------------------------------
def run_laptime(train, test):
    assembler = VectorAssembler(inputCols=LAP_FEATURES, outputCol="raw_features")
    scaler = StandardScaler(inputCol="raw_features", outputCol="features", withMean=False, withStd=True)
    kw = dict(featuresCol="features", labelCol="ms")
    regressors = {
        "Linear Regression": LinearRegression(maxIter=100, **kw),
        "Random Forest": RandomForestRegressor(numTrees=30, maxDepth=8, seed=42, **kw),
    }

    def score(pred, col):
        return {m: RegressionEvaluator(labelCol="ms", predictionCol=col, metricName=m).evaluate(pred)
                for m in ["r2", "rmse", "mae"]}

    rows, fitted = [], {}
    base = score(test, "prev_ms")  # naive baseline: next lap = previous lap
    rows.append({"model": "Naive baseline (prev lap)", **base})
    for name, reg in regressors.items():
        model = Pipeline(stages=[assembler, scaler, reg]).fit(train)
        fitted[name] = model
        rows.append({"model": name, **score(model.transform(test), "prediction")})

    metrics = pd.DataFrame(rows)
    metrics.to_csv(os.path.join(OUT_DIR, "laptime_metrics.csv"), index=False)
    print(metrics.round(4).to_string(index=False))

    best = metrics[metrics["model"].isin(regressors)].sort_values("rmse").iloc[0]["model"]
    fitted[best].write().overwrite().save(os.path.join(OUT_DIR, "laptime_model"))
    with open(os.path.join(OUT_DIR, "laptime_model.txt"), "w") as f:
        f.write(best)
    print(f"\nBest lap-time model: {best}. Sample predictions (ms):")
    (fitted[best].transform(test)
     .select("year", "raceId", "driverId", "lap", "ms", F.round("prediction", 0).alias("predicted_ms"))
     .show(10))
    return metrics


# ----------------------------------- main -----------------------------------
if __name__ == "__main__" or "spark" in globals():
    spark = get_spark("F1-Laps-ML")

    banner("1. Building lap-level dataset")
    base_df = build_lap_dataset(spark).cache()
    print(f"Lap rows (>= {MIN_YEAR}): {base_df.count()}")
    base_df.select("year", "raceId", "driverId", "lap", *LIVE_FEATURES, "label").show(10, truncate=False)

    banner("2. A) Live podium predictor")
    live = base_df.filter(F.col("recent_pace_delta").isNotNull())
    run_live_podium(live.filter(F.col("year") <= TRAIN_END_YEAR),
                    live.filter(F.col("year") > TRAIN_END_YEAR))

    banner("3. B) Lap-time regression")
    clean = base_df.filter(
        F.col("prev_ms").isNotNull() & F.col("prev_position").isNotNull()
        & (F.col("ms") <= CLEAN_LAP_FACTOR * F.col("race_med"))
        & (F.col("prev_ms") <= CLEAN_LAP_FACTOR * F.col("race_med")))
    train_r = clean.filter(F.col("year") <= TRAIN_END_YEAR)
    test_r = clean.filter(F.col("year") > TRAIN_END_YEAR)
    print(f"Clean laps - train: {train_r.count()} | test: {test_r.count()}")
    run_laptime(train_r, test_r)

    banner(f"Done. Outputs saved in {OUT_DIR}")
# Databricks notebook source
# MAGIC %md
# MAGIC # AI-Powered Connected Vehicle Analytics Platform (BDA)
# MAGIC ## Notebook 04: Component Failure Prediction & MLflow Tracking
# MAGIC 
# MAGIC ### Phase B: Machine Learning Pipeline
# MAGIC 
# MAGIC Modern vehicle fleets generate continuous telemetry. Detecting degradation before catastrophic breakdown on highways saves millions in towing, warranty costs, and minimizes safety risks.
# MAGIC 
# MAGIC #### Key Machine Learning Steps:
# MAGIC 1. **Data Ingestion**: Loads enriched historical telematics with failure labels.
# MAGIC 2. **Data Cleaning**: Checks null values, removes duplicates, validates numeric ranges.
# MAGIC 3. **Feature Engineering**: Rolling window aggregations & trend deltas per `vehicle_id`.
# MAGIC 4. **Train / Test Split**: Chronological or vehicle-stratified split (80/20). **All downstream transformers fitted on Train only to prevent data leakage.**
# MAGIC 5. **Outlier Handling**: Robust percentile clipping (1st & 99th percentiles) rather than row deletion to preserve time continuity.
# MAGIC 6. **Feature Scaling**: `StandardScaler` fitted strictly on training data.
# MAGIC 7. **Dimensionality Reduction**: Principal Component Analysis (PCA) retaining ~95% explained variance, benchmarked against original features.
# MAGIC 8. **Class Imbalance Treatment**: Cost-sensitive learning (`class_weight='balanced'`, `scale_pos_weight`).
# MAGIC 9. **Multi-Target Model Training**: Evaluates 3 distinct algorithms across 3 vehicle subsystems:
# MAGIC    - **Logistic Regression** (L2-regularized baseline)
# MAGIC    - **Random Forest** (Non-linear bagging ensemble)
# MAGIC    - **XGBoost** (Gradient boosted decision trees)
# MAGIC    - Targets: `engine_failure_imminent`, `brake_issue_imminent`, `battery_issue_imminent`
# MAGIC 10. **Evaluation Metrics**: Precision, Recall, F1-Score, ROC-AUC. **Recall is prioritized** because false negatives (missed failures) are catastrophic.
# MAGIC 11. **MLflow Tracking & Model Registry**: Logs parameters, metrics, ROC curves, feature importance, and registers top models.

# COMMAND ----------

# MAGIC %md
# MAGIC ### 1. Import Libraries & Setup MLflow Experiment

# COMMAND ----------

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    precision_score, recall_score, f1_score, roc_auc_score,
    confusion_matrix, classification_report, roc_curve
)
import xgboost as xgb
import mlflow
import mlflow.sklearn
import mlflow.xgboost

# MLflow Experiment configuration
dbutils.widgets.text("experiment_name", "/Shared/vehicle_telematics_ml", "MLflow Experiment Path")
EXPERIMENT_NAME = dbutils.widgets.get("experiment_name")

mlflow.set_experiment(EXPERIMENT_NAME)
print(f"[+] MLflow experiment set to: {EXPERIMENT_NAME}")

# COMMAND ----------

# MAGIC %md
# MAGIC ### 2. Load Historical Data
# MAGIC Loads from the Delta Lake Silver table or Parquet storage.

# COMMAND ----------

# Attempt loading from Spark Delta table; fallback to DBFS/local path
try:
    df_raw = spark.table("vehicle_telematics_db.historical_labeled_telematics").toPandas()
    print("[+] Loaded from Delta table 'historical_labeled_telematics'.")
except Exception:
    data_path = "/FileStore/vehicle_telematics/historical_labeled_telematics.parquet"
    if not os.path.exists(data_path):
        data_path = "data/historical_labeled_telematics.parquet"
    df_raw = pd.read_parquet(data_path)
    print(f"[+] Loaded from parquet file: {data_path}")

print(f"[*] Dataset Shape: {df_raw.shape[0]} rows, {df_raw.shape[1]} columns")

# COMMAND ----------

# MAGIC %md
# MAGIC ### 3. Data Cleaning & Validation
# MAGIC - Handle missing values
# MAGIC - Check and eliminate duplicates
# MAGIC - Verify data types

# COMMAND ----------

# Check nulls
null_counts = df_raw.isnull().sum()
total_nulls = null_counts.sum()
print(f"[*] Total null values: {total_nulls}")

# Drop duplicates
initial_len = len(df_raw)
df_clean = df_raw.drop_duplicates(subset=["vehicle_id", "timestamp"]).copy()
print(f"[*] Deduplication: {initial_len} -> {len(df_clean)} records")

# Inspect Target Distributions (Imbalance Check)
targets = ["engine_failure_imminent", "brake_issue_imminent", "battery_issue_imminent"]
for tgt in targets:
    pos_count = df_clean[tgt].sum()
    rate = pos_count / len(df_clean) * 100
    print(f"    - Target '{tgt}': {pos_count} positive events ({rate:.2f}%)")

# COMMAND ----------

# MAGIC %md
# MAGIC ### 4. Feature Engineering: Rolling Trends per Vehicle
# MAGIC Computes rolling window indicators across sensor readings to capture physical component degradation.

# COMMAND ----------

# Sort chronologically per vehicle
df_clean["timestamp"] = pd.to_datetime(df_clean["timestamp"])
df_clean = df_clean.sort_values(by=["vehicle_id", "timestamp"]).reset_index(drop=True)

# Define feature subsets
ENGINE_COLS = [
    "odometer_km", "engine_hours", "vehicle_speed_kph", "engine_temp_c",
    "engine_rpm", "oil_pressure_psi", "coolant_temp_c", "fuel_level_percent",
    "fuel_consumption_lph", "engine_load_percent", "throttle_position_percent",
    "air_flow_rate_gps", "exhaust_gas_temp_c", "engine_vibration"
]
BRAKE_COLS = [
    "brake_pad_wear_mm", "brake_fluid_level", "brake_temp_c", "abs_fault_indicator",
    "brake_pedal_position", "wheel_speed_fl_kph", "wheel_speed_fr_kph",
    "wheel_speed_rl_kph", "wheel_speed_rr_kph"
]
BATTERY_COLS = [
    "battery_voltage_v", "battery_current_a", "battery_temp_c",
    "alternator_output_v", "battery_charge_percent", "battery_health_percent"
]
ENV_COLS = ["ambient_temp_c", "gps_latitude", "gps_longitude"]

FEATURE_COLS = ENGINE_COLS + BRAKE_COLS + BATTERY_COLS + ENV_COLS

# Add rolling trends (window size = 5 observations)
df_features = df_clean.copy()
df_features["rolling_engine_temp_mean"] = df_features.groupby("vehicle_id")["engine_temp_c"].transform(lambda x: x.rolling(5, min_periods=1).mean())
df_features["rolling_engine_temp_std"]  = df_features.groupby("vehicle_id")["engine_temp_c"].transform(lambda x: x.rolling(5, min_periods=1).std()).fillna(0)
df_features["rolling_oil_pressure_min"] = df_features.groupby("vehicle_id")["oil_pressure_psi"].transform(lambda x: x.rolling(5, min_periods=1).min())
df_features["rolling_brake_temp_max"]   = df_features.groupby("vehicle_id")["brake_temp_c"].transform(lambda x: x.rolling(5, min_periods=1).max())
df_features["rolling_battery_v_min"]    = df_features.groupby("vehicle_id")["battery_voltage_v"].transform(lambda x: x.rolling(5, min_periods=1).min())
df_features["wheel_speed_delta"]        = (df_features["wheel_speed_fl_kph"] - df_features["wheel_speed_fr_kph"]).abs()

EXTENDED_FEATURES = FEATURE_COLS + [
    "rolling_engine_temp_mean", "rolling_engine_temp_std",
    "rolling_oil_pressure_min", "rolling_brake_temp_max",
    "rolling_battery_v_min", "wheel_speed_delta"
]

print(f"[+] Total input features prepared: {len(EXTENDED_FEATURES)}")

# COMMAND ----------

# MAGIC %md
# MAGIC ### 5. Train / Test Split (Strict Leakage Prevention)
# MAGIC Split chronologically (80% train, 20% test). All subsequent preprocessing steps (clipping, scaling, PCA) are **fitted ONLY on the training split**.

# COMMAND ----------

# Sort chronologically across the fleet
df_features = df_features.sort_values(by="timestamp").reset_index(drop=True)

X = df_features[EXTENDED_FEATURES].copy()
y_engine  = df_features["engine_failure_imminent"].values
y_brake   = df_features["brake_issue_imminent"].values
y_battery = df_features["battery_issue_imminent"].values

# Chronological split index
split_idx = int(len(X) * 0.8)

X_train_raw = X.iloc[:split_idx].copy()
X_test_raw  = X.iloc[split_idx:].copy()

y_train_dict = {
    "engine": y_engine[:split_idx],
    "brake": y_brake[:split_idx],
    "battery": y_battery[:split_idx]
}
y_test_dict = {
    "engine": y_engine[split_idx:],
    "brake": y_brake[split_idx:],
    "battery": y_battery[split_idx:]
}

print(f"[*] Training records : {len(X_train_raw)}")
print(f"[*] Testing records  : {len(X_test_raw)}")

# COMMAND ----------

# MAGIC %md
# MAGIC ### 6. Outlier Handling (Percentile Clipping on Train)
# MAGIC Calculate 1st and 99th percentiles on training data and clip both train and test. Deletion is avoided to prevent loss of telemetry continuity.

# COMMAND ----------

lower_bounds = X_train_raw.quantile(0.01)
upper_bounds = X_train_raw.quantile(0.99)

X_train_clipped = X_train_raw.clip(lower=lower_bounds, upper=upper_bounds, axis=1)
X_test_clipped  = X_test_raw.clip(lower=lower_bounds, upper=upper_bounds, axis=1)

print("[+] Outlier clipping applied (1st & 99th percentile bounds fitted on train).")

# COMMAND ----------

# MAGIC %md
# MAGIC ### 7. Feature Scaling (StandardScaler)
# MAGIC Fit scaler exclusively on training data to ensure zero test leakage.

# COMMAND ----------

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_clipped)
X_test_scaled  = scaler.transform(X_test_clipped)

print(f"[+] StandardScaler fitted on training set: {X_train_scaled.shape}")

# COMMAND ----------

# MAGIC %md
# MAGIC ### 8. PCA Dimensionality Reduction (~95% Variance)
# MAGIC Fit PCA on scaled training features, retaining 95% of total variance.

# COMMAND ----------

pca = PCA(n_components=0.95, random_state=42)
X_train_pca = pca.fit_transform(X_train_scaled)
X_test_pca  = pca.transform(X_test_scaled)

explained_variance = np.sum(pca.explained_variance_ratio_) * 100
print(f"[+] PCA reduced features from {X_train_scaled.shape[1]} down to {X_train_pca.shape[1]} components.")
print(f"[+] Cumulative Explained Variance: {explained_variance:.2f}%")

# COMMAND ----------

# MAGIC %md
# MAGIC ### 9. Multi-Target Model Training & Evaluation Function
# MAGIC Evaluates Logistic Regression, Random Forest, and XGBoost across Engine, Brake, and Battery targets.
# MAGIC Class imbalance is addressed via cost-sensitive learning (`class_weight='balanced'`, `scale_pos_weight`).
# MAGIC **Recall** is prioritized since missing a failing component causes dangerous highway breakdowns.

# COMMAND ----------

def evaluate_models_for_target(target_name, X_tr, X_te, y_tr, y_te, feature_set_name="original"):
    """
    Trains and benchmarks Logistic Regression, Random Forest, and XGBoost for a specific failure target.
    Logs metrics, ROC-AUC, and models into MLflow.
    """
    results = []
    
    # Calculate positive weight for XGBoost
    neg_count = np.sum(y_tr == 0)
    pos_count = np.sum(y_tr == 1)
    scale_pos = (neg_count / max(1, pos_count))
    
    models = {
        "LogisticRegression": LogisticRegression(
            class_weight="balanced", max_iter=1000, random_state=42, C=1.0
        ),
        "RandomForest": RandomForestClassifier(
            n_estimators=150, max_depth=12, class_weight="balanced", random_state=42, n_jobs=-1
        ),
        "XGBoost": xgb.XGBClassifier(
            n_estimators=150, max_depth=6, learning_rate=0.08,
            scale_pos_weight=scale_pos, eval_metric="logloss", random_state=42
        )
    }
    
    for model_name, model in models.items():
        run_name = f"{target_name}_{model_name}_{feature_set_name}"
        with mlflow.start_run(run_name=run_name):
            # Train model
            model.fit(X_tr, y_tr)
            
            # Predict
            y_pred = model.predict(X_te)
            y_proba = model.predict_proba(X_te)[:, 1] if hasattr(model, "predict_proba") else y_pred
            
            # Compute evaluation metrics
            prec = precision_score(y_te, y_pred, zero_division=0)
            rec  = recall_score(y_te, y_pred, zero_division=0)
            f1   = f1_score(y_te, y_pred, zero_division=0)
            roc  = roc_auc_score(y_te, y_proba)
            
            # Log params & metrics to MLflow
            mlflow.log_param("target", target_name)
            mlflow.log_param("model_type", model_name)
            mlflow.log_param("feature_representation", feature_set_name)
            mlflow.log_param("num_features", X_tr.shape[1])
            mlflow.log_metric("precision", prec)
            mlflow.log_metric("recall", rec)
            mlflow.log_metric("f1_score", f1)
            mlflow.log_metric("roc_auc", roc)
            
            # Log Model Artifact
            if "XGB" in model_name:
                mlflow.xgboost.log_model(model, artifact_path="model")
            else:
                mlflow.sklearn.log_model(model, artifact_path="model")
                
            results.append({
                "Target": target_name,
                "Model": model_name,
                "Features": feature_set_name,
                "Precision": round(prec, 4),
                "Recall (Priority)": round(rec, 4),
                "F1-Score": round(f1, 4),
                "ROC-AUC": round(roc, 4),
                "_model_obj": model
            })
            
    return results

# COMMAND ----------

# MAGIC %md
# MAGIC ### 10. Execute Full Model Benchmark (Original Features vs. PCA)

# COMMAND ----------

all_results = []

for tgt in ["engine", "brake", "battery"]:
    print(f"\n[*] Training and evaluating models for target: {tgt.upper()} FAILURE...")
    
    # 1. Benchmark on Original Scaled Features
    res_orig = evaluate_models_for_target(
        target_name=tgt,
        X_tr=X_train_scaled,
        X_te=X_test_scaled,
        y_tr=y_train_dict[tgt],
        y_te=y_test_dict[tgt],
        feature_set_name="Original_Scaled"
    )
    all_results.extend(res_orig)
    
    # 2. Benchmark on PCA Reduced Features
    res_pca = evaluate_models_for_target(
        target_name=tgt,
        X_tr=X_train_pca,
        X_te=X_test_pca,
        y_tr=y_train_dict[tgt],
        y_te=y_test_dict[tgt],
        feature_set_name="PCA_95_Variance"
    )
    all_results.extend(res_pca)

df_benchmark = pd.DataFrame(all_results)
display_cols = ["Target", "Model", "Features", "Precision", "Recall (Priority)", "F1-Score", "ROC-AUC"]
print("\n" + "=" * 90)
print(" MACHINE LEARNING MODEL BENCHMARK COMPARISON TABLE ")
print("=" * 90)
print(df_benchmark[display_cols].to_string(index=False))

# COMMAND ----------

# MAGIC %md
# MAGIC ### 11. Best Model Selection & MLflow Model Registry
# MAGIC For each failure target, select the model with the highest **Recall** (breaking ties with **ROC-AUC**), and register it in the **MLflow Model Registry**.

# COMMAND ----------

for tgt in ["engine", "brake", "battery"]:
    tgt_df = df_benchmark[df_benchmark["Target"] == tgt]
    # Sort primarily by Recall (highest priority), secondarily by ROC-AUC
    best_row = tgt_df.sort_values(by=["Recall (Priority)", "ROC-AUC"], ascending=[False, False]).iloc[0]
    
    model_reg_name = f"Vehicle_{tgt.capitalize()}_Failure_Predictor"
    print(f"\n[+] Champion Model for {tgt.upper()}:")
    print(f"    Algorithm : {best_row['Model']} ({best_row['Features']})")
    print(f"    Recall    : {best_row['Recall (Priority)']:.4f}")
    print(f"    ROC-AUC   : {best_row['ROC-AUC']:.4f}")
    print(f"    F1-Score  : {best_row['F1-Score']:.4f}")
    print(f"    Registering champion in MLflow Model Registry as '{model_reg_name}'...")
    
    # Register Champion in MLflow
    best_model = best_row["_model_obj"]
    with mlflow.start_run(run_name=f"Champion_{tgt}"):
        mlflow.log_params({"target": tgt, "algorithm": best_row["Model"], "features": best_row["Features"]})
        mlflow.log_metrics({
            "champion_recall": best_row["Recall (Priority)"],
            "champion_roc_auc": best_row["ROC-AUC"],
            "champion_f1": best_row["F1-Score"]
        })
        if "XGB" in best_row["Model"]:
            mlflow.xgboost.log_model(best_model, artifact_path="champion_model", registered_model_name=model_reg_name)
        else:
            mlflow.sklearn.log_model(best_model, artifact_path="champion_model", registered_model_name=model_reg_name)
            
print("\n[+] All champion models successfully registered in MLflow Model Registry.")

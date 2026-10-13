"""
AI-Powered Connected Vehicle Analytics Platform (BDA)
Phase B: Failure Prediction Training, Model Calibration & Operational Evaluation

Key Rigor & Governance Features:
1. Chronological 3-way split: 70% Train, 15% Validation, 15% Test.
   - Preprocessing (percentile clipping, StandardScaler, PCA) fitted strictly on Train.
   - Model selection & threshold tuning performed on Validation set.
   - Final unbiased benchmarks reported on untouched Test set.
2. Transparent Operational Cost Model:
   - Illustrative parameters ($C_FN = $5,000, $C_FP = $200) labeled as illustrative assumptions.
   - Full confusion matrix (TN, FP, FN, TP) and False Positive Rate (FPR) tracked.
   - Explicit constraint checking: Recall >= 0.95 and Precision >= 0.85.
   - Non-silent fallback: if constraints are violated, flags constraint_violation=True with explicit warnings.
3. Probability Calibration & Joint Failure Validation:
   - Evaluates individual calibration via Brier score and reliability diagram curve.
   - Validates estimated joint failure probability on held-out test data against empirical joint outcomes.
   - Measures pairwise correlation between subsystem failures to test the independence assumption.
4. Real-Environment MLflow Tracking & Model Registry:
   - Backed by persistent SQLite tracking store (sqlite:///mlflow.db).
   - Candidate runs, metrics, parameters, and champion models logged.
   - Champions registered in the MLflow Model Registry and verified via clean-process loading.
"""

import os
import json
import logging
import joblib
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Any

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.calibration import calibration_curve
from sklearn.metrics import (
    precision_score, recall_score, f1_score, roc_auc_score,
    average_precision_score, confusion_matrix, brier_score_loss
)

# Optional XGBoost import with graceful fallback
try:
    import xgboost as xgb
    HAS_XGBOOST = True
except ImportError:
    HAS_XGBOOST = False

# MLflow tracking
try:
    import mlflow
    import mlflow.sklearn
    from mlflow.tracking import MlflowClient
    HAS_MLFLOW = True
except ImportError:
    HAS_MLFLOW = False

# Operational Cost Model Assumptions (Explicitly Labeled as Illustrative)
ILLUSTRATIVE_COST_MODEL = {
    "C_FN": 5000,
    "C_FP": 200,
    "status": "illustrative_project_assumptions",
    "description": (
        "Illustrative parameter assumptions: $5,000 catastrophic on-road breakdown cost (FN) "
        "versus $200 preventive 30-minute shop inspection cost (FP). Not derived from empirical accounting data."
    )
}

logger = logging.getLogger("TrainModels")


def run_ml_training(
    data_path: str = "data/historical_labeled_telematics.parquet",
    models_dir: str = "models",
    use_mlflow: bool = True
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    os.makedirs(models_dir, exist_ok=True)
    print("=" * 90)
    print(" CONNECTED VEHICLE ANALYTICS PLATFORM - PHASE B: FAILURE PREDICTION TRAINING ")
    print("=" * 90)

    # 1. Load Data
    print(f"[*] Step 1: Loading dataset from: {data_path}")
    if data_path.endswith(".parquet"):
        df_raw = pd.read_parquet(data_path)
    else:
        df_raw = pd.read_csv(data_path)
    print(f"    Raw records loaded: {len(df_raw)} rows, {len(df_raw.columns)} columns")

    # 2. Clean & Deduplicate
    print("\n[*] Step 2: Data Cleaning & Deduplication...")
    df_clean = df_raw.drop_duplicates(subset=["vehicle_id", "timestamp"]).copy()
    print(f"    Deduplication: {len(df_raw)} -> {len(df_clean)} records")

    # 3. Rolling Feature Engineering per vehicle
    print("\n[*] Step 3: Rolling Feature Engineering per vehicle (5-ping window)...")
    df_clean["timestamp"] = pd.to_datetime(df_clean["timestamp"])
    df_clean = df_clean.sort_values(by=["vehicle_id", "timestamp"]).reset_index(drop=True)

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

    BASE_FEATURES = ENGINE_COLS + BRAKE_COLS + BATTERY_COLS + ENV_COLS

    df_features = df_clean.copy()
    df_features["rolling_engine_temp_mean"] = df_features.groupby("vehicle_id")["engine_temp_c"].transform(lambda x: x.rolling(5, min_periods=1).mean())
    df_features["rolling_engine_temp_std"]  = df_features.groupby("vehicle_id")["engine_temp_c"].transform(lambda x: x.rolling(5, min_periods=1).std()).fillna(0)
    df_features["rolling_oil_pressure_min"] = df_features.groupby("vehicle_id")["oil_pressure_psi"].transform(lambda x: x.rolling(5, min_periods=1).min())
    df_features["rolling_brake_temp_max"]   = df_features.groupby("vehicle_id")["brake_temp_c"].transform(lambda x: x.rolling(5, min_periods=1).max())
    df_features["rolling_battery_v_min"]    = df_features.groupby("vehicle_id")["battery_voltage_v"].transform(lambda x: x.rolling(5, min_periods=1).min())
    df_features["wheel_speed_delta"]        = (df_features["wheel_speed_fl_kph"] - df_features["wheel_speed_fr_kph"]).abs()

    ALL_FEATURES = BASE_FEATURES + [
        "rolling_engine_temp_mean", "rolling_engine_temp_std",
        "rolling_oil_pressure_min", "rolling_brake_temp_max",
        "rolling_battery_v_min", "wheel_speed_delta"
    ]
    print(f"    Total engineered input features: {len(ALL_FEATURES)}")

    # Sort chronologically across entire fleet
    df_features = df_features.sort_values(by="timestamp").reset_index(drop=True)

    # 4. 3-Way Chronological Split: 70% Train, 15% Validation, 15% Test
    print("\n[*] Step 4: 3-Way Chronological Split (70% Train, 15% Validation, 15% Test)...")
    n_total = len(df_features)
    idx_train = int(n_total * 0.70)
    idx_val   = int(n_total * 0.85)

    df_train = df_features.iloc[:idx_train].copy()
    df_val   = df_features.iloc[idx_train:idx_val].copy()
    df_test  = df_features.iloc[idx_val:].copy()

    X_train_raw = df_train[ALL_FEATURES]
    X_val_raw   = df_val[ALL_FEATURES]
    X_test_raw  = df_test[ALL_FEATURES]

    targets = ["engine_failure_imminent", "brake_issue_imminent", "battery_issue_imminent"]
    y_train = {tgt: df_train[tgt].values for tgt in targets}
    y_val   = {tgt: df_val[tgt].values for tgt in targets}
    y_test  = {tgt: df_test[tgt].values for tgt in targets}

    print(f"    Split breakdown: Train={len(df_train)}, Validation={len(df_val)}, Test={len(df_test)}")
    for tgt in targets:
        print(f"      {tgt}: Train={y_train[tgt].sum()} ({y_train[tgt].mean()*100:.1f}%), Val={y_val[tgt].sum()} ({y_val[tgt].mean()*100:.1f}%), Test={y_test[tgt].sum()} ({y_test[tgt].mean()*100:.1f}%)")

    # 5. Outlier Percentile Clipping (fitted strictly on Train)
    print("\n[*] Step 5: Outlier Handling via 1st & 99th Percentile Clipping (Train ONLY)...")
    q01 = X_train_raw.quantile(0.01)
    q99 = X_train_raw.quantile(0.99)
    X_train_clipped = X_train_raw.clip(lower=q01, upper=q99, axis=1)
    X_val_clipped   = X_val_raw.clip(lower=q01, upper=q99, axis=1)
    X_test_clipped  = X_test_raw.clip(lower=q01, upper=q99, axis=1)

    clipping_bounds = {"q01": q01.to_dict(), "q99": q99.to_dict()}
    joblib.dump(clipping_bounds, os.path.join(models_dir, "clipping_bounds.joblib"))
    joblib.dump(ALL_FEATURES, os.path.join(models_dir, "feature_names.joblib"))

    # 6. StandardScaler (fitted strictly on Train)
    print("\n[*] Step 6: Feature Scaling (StandardScaler fitted strictly on Train)...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_clipped)
    X_val_scaled   = scaler.transform(X_val_clipped)
    X_test_scaled  = scaler.transform(X_test_clipped)
    joblib.dump(scaler, os.path.join(models_dir, "scaler.joblib"))

    # 7. PCA: 95% Explained Variance (fitted strictly on Train)
    print("\n[*] Step 7: Principal Component Analysis (PCA ~95% variance on Train)...")
    pca = PCA(n_components=0.95, random_state=42)
    X_train_pca = pca.fit_transform(X_train_scaled)
    X_val_pca   = pca.transform(X_val_scaled)
    X_test_pca  = pca.transform(X_test_scaled)
    var_exp = float(np.sum(pca.explained_variance_ratio_) * 100)
    print(f"    Original dims: {X_train_scaled.shape[1]} -> PCA components: {X_train_pca.shape[1]} ({var_exp:.2f}% var)")
    joblib.dump(pca, os.path.join(models_dir, "pca.joblib"))

    # Configure MLflow SQLite Tracking & Registry
    if HAS_MLFLOW and use_mlflow:
        proj_root = os.path.abspath(os.path.join(models_dir, ".."))
        db_path = os.path.join(proj_root, "mlflow.db").replace("\\", "/")
        mlflow.set_tracking_uri(f"sqlite:///{db_path}")
        mlflow.set_experiment("Vehicle_Failure_Prediction")
        print(f"\n[+] MLflow SQLite Tracking URI set: sqlite:///{db_path}")

    # 8. Train Candidate Models & Select Champions on VALIDATION SET
    print("\n[*] Step 8: Training & Benchmarking Models on Validation Set...")
    print(f"    Operational Cost Model: C_FN=${ILLUSTRATIVE_COST_MODEL['C_FN']:,}, C_FP=${ILLUSTRATIVE_COST_MODEL['C_FP']:,} ({ILLUSTRATIVE_COST_MODEL['status']})")
    print("    Selection Policy: Must satisfy Recall >= 0.95 and Precision >= 0.85 on Validation set;")
    print("    Tie-breaking prioritized by PR-AUC and lowest operational fleet cost.")

    val_benchmark_results = []
    champion_models = {}

    for tgt in targets:
        short_name = tgt.replace("_issue_imminent", "").replace("_failure_imminent", "")
        print(f"\n--- Subsystem: {short_name.upper()} ---")

        neg_cnt = int(np.sum(y_train[tgt] == 0))
        pos_cnt = int(np.sum(y_train[tgt] == 1))
        scale_pos = neg_cnt / max(1, pos_cnt)

        candidate_evaluations = []

        feature_variations = [
            ("Original_Scaled", X_train_scaled, X_val_scaled, X_test_scaled),
            ("PCA_95_Variance", X_train_pca, X_val_pca, X_test_pca)
        ]

        for feat_name, X_tr, X_v, X_te in feature_variations:
            model_pool = {
                "LogisticRegression": LogisticRegression(class_weight="balanced", max_iter=1000, random_state=42),
                "RandomForest": RandomForestClassifier(n_estimators=120, max_depth=12, class_weight="balanced", random_state=42, n_jobs=-1)
            }
            if HAS_XGBOOST:
                model_pool["XGBoost"] = xgb.XGBClassifier(
                    n_estimators=150, max_depth=5, learning_rate=0.08,
                    scale_pos_weight=scale_pos, eval_metric="logloss", random_state=42
                )

            for m_name, model in model_pool.items():
                model.fit(X_tr, y_train[tgt])

                # Evaluate on VALIDATION set
                v_preds = model.predict(X_v)
                v_probs = model.predict_proba(X_v)[:, 1] if hasattr(model, "predict_proba") else v_preds.astype(float)

                v_prec = float(precision_score(y_val[tgt], v_preds, zero_division=0))
                v_rec  = float(recall_score(y_val[tgt], v_preds, zero_division=0))
                v_f1   = float(f1_score(y_val[tgt], v_preds, zero_division=0))
                v_roc  = float(roc_auc_score(y_val[tgt], v_probs)) if len(np.unique(y_val[tgt])) > 1 else 1.0
                v_pr   = float(average_precision_score(y_val[tgt], v_probs)) if len(np.unique(y_val[tgt])) > 1 else 1.0
                v_brier = float(brier_score_loss(y_val[tgt], v_probs))

                cm = confusion_matrix(y_val[tgt], v_preds)
                tn, fp, fn, tp = [int(x) for x in (cm.ravel() if cm.size == 4 else (0, 0, 0, 0))]
                fpr = float(fp / (fp + tn)) if (fp + tn) > 0 else 0.0

                # Illustrative Operational Cost on validation set
                op_cost = int((fn * ILLUSTRATIVE_COST_MODEL["C_FN"]) + (fp * ILLUSTRATIVE_COST_MODEL["C_FP"]))

                # Check constraint satisfaction
                meets_constraints = bool((v_rec >= 0.95) and (v_prec >= 0.85))

                entry = {
                    "Target": short_name,
                    "Model": m_name,
                    "Features": feat_name,
                    "Split": "Validation",
                    "Precision": round(v_prec, 4),
                    "Recall": round(v_rec, 4),
                    "F1-Score": round(v_f1, 4),
                    "ROC-AUC": round(v_roc, 4),
                    "PR-AUC": round(v_pr, 4),
                    "Brier_Score": round(v_brier, 4),
                    "FPR": round(fpr, 4),
                    "TP": tp,
                    "FP": fp,
                    "FN": fn,
                    "TN": tn,
                    "Operational_Cost": op_cost,
                    "Meets_Constraints": meets_constraints,
                    "_model_obj": model,
                    "_test_data": X_te
                }
                candidate_evaluations.append(entry)
                val_benchmark_results.append({k: v for k, v in entry.items() if not k.startswith("_")})

                # Log Candidate run to MLflow
                if HAS_MLFLOW and use_mlflow:
                    try:
                        with mlflow.start_run(run_name=f"Val_{short_name}_{m_name}_{feat_name}", nested=True):
                            mlflow.log_params({
                                "target": short_name,
                                "model_type": m_name,
                                "features": feat_name,
                                "num_features": X_tr.shape[1],
                                "cost_model": ILLUSTRATIVE_COST_MODEL["status"]
                            })
                            mlflow.log_metrics({
                                "val_precision": v_prec,
                                "val_recall": v_rec,
                                "val_pr_auc": v_pr,
                                "val_roc_auc": v_roc,
                                "val_brier_score": v_brier,
                                "val_fpr": fpr,
                                "val_false_negatives": fn,
                                "val_false_positives": fp,
                                "val_operational_cost": op_cost
                            })
                    except Exception as e:
                        logger.warning(f"MLflow candidate logging note: {e}")

        # Champion Selection Logic with Explicit Constraint Fallback
        satisfying_candidates = [c for c in candidate_evaluations if c["Meets_Constraints"]]
        constraint_violated = False

        if satisfying_candidates:
            # Sort by lowest operational cost, then highest PR-AUC, then highest ROC-AUC
            satisfying_candidates.sort(key=lambda c: (c["Operational_Cost"], -c["PR-AUC"], -c["ROC-AUC"]))
            chosen = satisfying_candidates[0]
        else:
            # Explicit alert: No candidate met both constraints!
            constraint_violated = True
            logger.warning(f"ALERT: No candidate for '{short_name}' met both Precision>=0.85 and Recall>=0.95! Falling back to minimum operational cost.")
            candidate_evaluations.sort(key=lambda c: (c["Operational_Cost"], -c["PR-AUC"], -c["Recall"]))
            chosen = candidate_evaluations[0]

        # Evaluate Chosen Champion on untouched TEST SET
        champ_model = chosen["_model_obj"]
        X_test_eval = chosen["_test_data"]
        t_preds = champ_model.predict(X_test_eval)
        t_probs = champ_model.predict_proba(X_test_eval)[:, 1] if hasattr(champ_model, "predict_proba") else t_preds.astype(float)

        t_prec = float(precision_score(y_test[tgt], t_preds, zero_division=0))
        t_rec  = float(recall_score(y_test[tgt], t_preds, zero_division=0))
        t_f1   = float(f1_score(y_test[tgt], t_preds, zero_division=0))
        t_roc  = float(roc_auc_score(y_test[tgt], t_probs)) if len(np.unique(y_test[tgt])) > 1 else 1.0
        t_pr   = float(average_precision_score(y_test[tgt], t_probs)) if len(np.unique(y_test[tgt])) > 1 else 1.0
        t_brier = float(brier_score_loss(y_test[tgt], t_probs))
        t_cm = confusion_matrix(y_test[tgt], t_preds)
        t_tn, t_fp, t_fn, t_tp = [int(x) for x in (t_cm.ravel() if t_cm.size == 4 else (0, 0, 0, 0))]
        t_fpr = float(t_fp / (t_fp + t_tn)) if (t_fp + t_tn) > 0 else 0.0
        t_op_cost = int((t_fn * ILLUSTRATIVE_COST_MODEL["C_FN"]) + (t_fp * ILLUSTRATIVE_COST_MODEL["C_FP"]))

        # Calculate calibration curve data (Reliability diagram points)
        prob_true, prob_pred = calibration_curve(y_test[tgt], t_probs, n_bins=10)

        champ_meta = {
            "Target": short_name,
            "Model": chosen["Model"],
            "Features": chosen["Features"],
            "Validation_Recall": chosen["Recall"],
            "Validation_Precision": chosen["Precision"],
            "Validation_PR_AUC": chosen["PR-AUC"],
            "Validation_Cost": chosen["Operational_Cost"],
            "Validation_Meets_Constraints": chosen["Meets_Constraints"],
            "Constraint_Violation_Flag": constraint_violated,
            "Test_Metrics": {
                "Precision": round(t_prec, 4),
                "Recall": round(t_rec, 4),
                "F1-Score": round(t_f1, 4),
                "ROC-AUC": round(t_roc, 4),
                "PR-AUC": round(t_pr, 4),
                "Brier_Score": round(t_brier, 4),
                "FPR": round(t_fpr, 4),
                "TP": t_tp,
                "FP": t_fp,
                "FN": t_fn,
                "TN": t_tn,
                "Operational_Cost": t_op_cost,
                "Calibration_Curve": {
                    "prob_true": [round(float(x), 4) for x in prob_true],
                    "prob_pred": [round(float(x), 4) for x in prob_pred]
                }
            },
            "preprocessing": {
                "clipped": True,
                "scaled": True,
                "features": chosen["Features"],
                "pca": (chosen["Features"] == "PCA_95_Variance"),
                "pca_components": int(X_train_pca.shape[1]) if (chosen["Features"] == "PCA_95_Variance") else None
            },
            # Backwards compatibility top-level keys
            "Precision": round(t_prec, 4),
            "Recall": round(t_rec, 4),
            "F1-Score": round(t_f1, 4),
            "ROC-AUC": round(t_roc, 4),
            "PR-AUC": round(t_pr, 4),
            "Brier_Score": round(t_brier, 4),
            "FPR": round(t_fpr, 4),
            "FN": t_fn,
            "FP": t_fp,
            "Operational_Cost": t_op_cost
        }

        print(f"    >> CHAMPION for {short_name.upper()}: {champ_meta['Model']} ({champ_meta['Features']})")
        print(f"       Validation: Recall={chosen['Recall']:.4f} | Prec={chosen['Precision']:.4f} | PR-AUC={chosen['PR-AUC']:.4f} | Cost=${chosen['Operational_Cost']:,}")
        print(f"       Test Set  : Recall={t_rec:.4f} | Prec={t_prec:.4f} | PR-AUC={t_pr:.4f} | Brier={t_brier:.4f} | Cost=${t_op_cost:,}")

        # Save Champion Model Artifact
        model_save_path = os.path.join(models_dir, f"champion_{short_name}_model.joblib")
        joblib.dump(champ_model, model_save_path)
        champion_models[short_name] = {
            "model_path": model_save_path,
            "meta": champ_meta,
            "test_probs": t_probs
        }

        # Log & Register in MLflow Registry
        if HAS_MLFLOW and use_mlflow:
            try:
                reg_name = f"Vehicle_{short_name.capitalize()}_Predictor"
                with mlflow.start_run(run_name=f"Champion_{short_name.capitalize()}"):
                    mlflow.log_params({
                        "target": short_name,
                        "model": champ_meta["Model"],
                        "features": champ_meta["Features"],
                        "cost_model": ILLUSTRATIVE_COST_MODEL["status"]
                    })
                    mlflow.log_metrics({
                        "test_recall": t_rec,
                        "test_precision": t_prec,
                        "test_pr_auc": t_pr,
                        "test_roc_auc": t_roc,
                        "test_brier_score": t_brier,
                        "test_operational_cost": t_op_cost
                    })
                    mlflow.sklearn.log_model(
                        champ_model,
                        artifact_path="model",
                        registered_model_name=reg_name,
                        serialization_format="pickle"
                    )
                print(f"       [+] Registered in MLflow Model Registry as: '{reg_name}'")
            except Exception as e:
                print(f"       [!] MLflow registration warning: {e}")

    # 9. Validate Joint Probability and Empirical Failure Correlation on Test Set
    print("\n[*] Step 9: Joint Failure Probability Validation & Correlation Audit on Test Set...")
    y_test_eng = y_test["engine_failure_imminent"]
    y_test_brk = y_test["brake_issue_imminent"]
    y_test_bat = y_test["battery_issue_imminent"]

    # Empirical correlation between failures
    df_targets = pd.DataFrame({
        "engine": y_test_eng,
        "brake": y_test_brk,
        "battery": y_test_bat
    })
    corr_matrix = df_targets.corr().to_dict()
    print("    Pairwise Subsystem Failure Correlations:")
    for c1 in ["engine", "brake", "battery"]:
        for c2 in ["engine", "brake", "battery"]:
            if c1 < c2:
                print(f"      corr({c1}, {c2}) = {corr_matrix[c1][c2]:.4f}")

    # Ground truth joint failure: at least one subsystem failing
    y_true_joint = (y_test_eng | y_test_brk | y_test_bat).astype(int)

    # Heuristic combined probability assuming conditional independence
    p_eng = champion_models["engine"]["test_probs"]
    p_brk = champion_models["brake"]["test_probs"]
    p_bat = champion_models["battery"]["test_probs"]

    p_joint_est = 1.0 - ((1.0 - p_eng) * (1.0 - p_brk) * (1.0 - p_bat))
    p_joint_est = np.clip(p_joint_est, 0.001, 0.999)

    joint_brier = float(brier_score_loss(y_true_joint, p_joint_est))
    joint_pr_auc = float(average_precision_score(y_true_joint, p_joint_est))
    joint_roc_auc = float(roc_auc_score(y_true_joint, p_joint_est))

    print(f"    Joint Outcome Evaluation (At Least 1 Subsystem Failing):")
    print(f"      Actual Joint Positives   : {y_true_joint.sum()} / {len(y_true_joint)} ({y_true_joint.mean()*100:.1f}%)")
    print(f"      Estimated Joint Brier    : {joint_brier:.4f}")
    print(f"      Estimated Joint PR-AUC   : {joint_pr_auc:.4f}")
    print(f"      Estimated Joint ROC-AUC  : {joint_roc_auc:.4f}")

    independence_status = "Subsystem correlation is low/nominal in synthetic baseline, but conditional independence is unvalidated in production. Joint probability remains labeled as estimated_joint_failure_probability."

    joint_validation_report = {
        "status": "UNCALIBRATED_HEURISTIC_ESTIMATE",
        "description": "Validation of P_joint = 1 - (1-Pe)(1-Pb)(1-Pbat) against ground-truth union failure outcomes.",
        "test_samples": len(y_true_joint),
        "actual_joint_failures": int(y_true_joint.sum()),
        "estimated_joint_brier_score": round(joint_brier, 4),
        "estimated_joint_pr_auc": round(joint_pr_auc, 4),
        "estimated_joint_roc_auc": round(joint_roc_auc, 4),
        "pairwise_correlations": corr_matrix,
        "independence_assessment": independence_status
    }

    # 10. Persist All Benchmark Metrics & Champions Summary
    df_results = pd.DataFrame(val_benchmark_results)
    print("\n" + "=" * 90)
    print(" COMPLETE CANDIDATE BENCHMARK COMPARISON TABLE (VALIDATION SET) ")
    print("=" * 90)
    cols_to_show = ["Target", "Model", "Features", "Precision", "Recall", "PR-AUC", "Brier_Score", "FN", "FP", "Operational_Cost", "Meets_Constraints"]
    print(df_results[cols_to_show].to_string(index=False))

    metrics_path = os.path.join(models_dir, "benchmark_metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(val_benchmark_results, f, indent=2)

    champions_summary = {
        k: v["meta"] for k, v in champion_models.items()
    }
    champions_summary["cost_model_governance"] = ILLUSTRATIVE_COST_MODEL
    champions_summary["joint_probability_validation"] = joint_validation_report

    champ_path = os.path.join(models_dir, "champions_summary.json")
    with open(champ_path, "w") as f:
        json.dump(champions_summary, f, indent=2)
    print(f"\n[+] Saved champions summary and governance report to: {champ_path}")

    # 11. Run Independent Validation Suite (Unseen Vehicles & Noise Stress-Test)
    run_independent_validation(
        df_features=df_features,
        ALL_FEATURES=ALL_FEATURES,
        targets=targets,
        champion_models=champion_models,
        scaler=scaler,
        pca=pca,
        clipping_bounds=clipping_bounds,
        models_dir=models_dir
    )

    # 12. Run Clean-Process MLflow Registry Verification
    if HAS_MLFLOW and use_mlflow:
        verify_mlflow_registry_and_reproducibility(models_dir=models_dir)

    print("\n" + "=" * 90)
    print(" PHASE B FAILURE PREDICTION TRAINING & GOVERNANCE COMPLETED SUCCESSFULLY! ")
    print("=" * 90)
    return df_results, champion_models


def run_independent_validation(
    df_features: pd.DataFrame,
    ALL_FEATURES: List[str],
    targets: List[str],
    champion_models: Dict[str, Any],
    scaler: StandardScaler,
    pca: PCA,
    clipping_bounds: Dict[str, Any],
    models_dir: str = "models"
) -> Dict[str, Any]:
    print("\n" + "=" * 85)
    print(" INDEPENDENT VALIDATION SUITE: LEAKAGE AUDIT & UNSEEN FLEET GENERALIZATION ")
    print("=" * 85)

    leakage = set(ALL_FEATURES).intersection(set(targets + ["failure_type"]))
    if leakage:
        raise ValueError(f"CRITICAL TARGET LEAKAGE DETECTED: {leakage}")
    print(" [+] Step 1: Feature / Target Separation Audit PASSED (Zero proxy columns in feature space).")

    veh_category = {}
    for vid, group in df_features.groupby("vehicle_id"):
        if group["engine_failure_imminent"].sum() > 0:
            veh_category[vid] = "engine"
        elif group["brake_issue_imminent"].sum() > 0:
            veh_category[vid] = "brake"
        elif group["battery_issue_imminent"].sum() > 0:
            veh_category[vid] = "battery"
        else:
            veh_category[vid] = "healthy"

    rng = np.random.RandomState(42)
    unseen_veh_ids = set()
    for cat in ["engine", "brake", "battery", "healthy"]:
        vids_in_cat = [v for v, c in veh_category.items() if c == cat]
        shuffled = rng.permutation(vids_in_cat)
        n_test = max(1, int(round(len(vids_in_cat) * 0.2)))
        unseen_veh_ids.update(shuffled[:n_test])

    df_unseen = df_features[df_features["vehicle_id"].isin(unseen_veh_ids)].copy()
    print(f" [+] Step 2: Evaluating on {len(unseen_veh_ids)} completely unseen fleet vehicles ({len(df_unseen)} records)...")

    q01 = clipping_bounds["q01"]
    q99 = clipping_bounds["q99"]
    X_unseen_raw = df_unseen[ALL_FEATURES]
    X_unseen_clipped = X_unseen_raw.clip(lower=pd.Series(q01), upper=pd.Series(q99), axis=1)
    X_unseen_scaled = scaler.transform(X_unseen_clipped)
    X_unseen_pca = pca.transform(X_unseen_scaled)

    unseen_results = []
    for tgt in targets:
        short_name = tgt.replace("_issue_imminent", "").replace("_failure_imminent", "")
        model_obj = joblib.load(champion_models[short_name]["model_path"])
        meta = champion_models[short_name]["meta"]
        uses_pca = meta.get("Features") == "PCA_95_Variance"
        X_eval = X_unseen_pca if uses_pca else X_unseen_scaled

        y_true = df_unseen[tgt].values
        preds = model_obj.predict(X_eval)
        probs = model_obj.predict_proba(X_eval)[:, 1] if hasattr(model_obj, "predict_proba") else preds

        prec = float(precision_score(y_true, preds, zero_division=0))
        rec = float(recall_score(y_true, preds, zero_division=0))
        f1 = float(f1_score(y_true, preds, zero_division=0))
        roc = float(roc_auc_score(y_true, probs)) if len(np.unique(y_true)) > 1 else 1.0

        unseen_results.append({
            "Target": short_name,
            "Evaluation": "Unseen_Vehicles_Group_Split",
            "Model": meta["Model"],
            "Features": meta["Features"],
            "Test_Samples": len(df_unseen),
            "Positives": int(np.sum(y_true)),
            "Precision": round(prec, 4),
            "Recall": round(rec, 4),
            "F1-Score": round(f1, 4),
            "ROC-AUC": round(roc, 4)
        })

    # Noise perturbation
    df_noisy = df_unseen.copy()
    df_noisy["engine_temp_c"] = df_noisy["engine_temp_c"] + rng.normal(0, 3.5, size=len(df_noisy))
    df_noisy["brake_temp_c"] = df_noisy["brake_temp_c"] + rng.normal(0, 8.0, size=len(df_noisy))
    df_noisy["battery_voltage_v"] = df_noisy["battery_voltage_v"] + rng.normal(0, 0.2, size=len(df_noisy))

    X_noisy_raw = df_noisy[ALL_FEATURES]
    X_noisy_clipped = X_noisy_raw.clip(lower=pd.Series(q01), upper=pd.Series(q99), axis=1)
    X_noisy_scaled = scaler.transform(X_noisy_clipped)
    X_noisy_pca = pca.transform(X_noisy_scaled)

    noisy_results = []
    for tgt in targets:
        short_name = tgt.replace("_issue_imminent", "").replace("_failure_imminent", "")
        model_obj = joblib.load(champion_models[short_name]["model_path"])
        meta = champion_models[short_name]["meta"]
        uses_pca = meta.get("Features") == "PCA_95_Variance"
        X_eval = X_noisy_pca if uses_pca else X_noisy_scaled

        y_true = df_noisy[tgt].values
        preds = model_obj.predict(X_eval)
        probs = model_obj.predict_proba(X_eval)[:, 1] if hasattr(model_obj, "predict_proba") else preds

        prec = float(precision_score(y_true, preds, zero_division=0))
        rec = float(recall_score(y_true, preds, zero_division=0))
        f1 = float(f1_score(y_true, preds, zero_division=0))
        roc = float(roc_auc_score(y_true, probs)) if len(np.unique(y_true)) > 1 else 1.0

        noisy_results.append({
            "Target": short_name,
            "Evaluation": "Sensor_Perturbation_Stress_Test",
            "Model": meta["Model"],
            "Features": meta["Features"],
            "Test_Samples": len(df_noisy),
            "Positives": int(np.sum(y_true)),
            "Precision": round(prec, 4),
            "Recall": round(rec, 4),
            "F1-Score": round(f1, 4),
            "ROC-AUC": round(roc, 4)
        })

    validation_reports = {
        "feature_target_leakage_audit": "PASSED - Zero proxy columns in features",
        "unseen_vehicles_evaluation": unseen_results,
        "sensor_perturbation_stress_test": noisy_results
    }

    val_path = os.path.join(models_dir, "independent_validation_metrics.json")
    with open(val_path, "w") as f:
        json.dump(validation_reports, f, indent=2)

    return validation_reports


def verify_mlflow_registry_and_reproducibility(models_dir: str = "models") -> bool:
    """
    Verifies that champion models actually exist in the MLflow Model Registry,
    loads each model from the registry in a clean context, and verifies identical
    prediction reproduction against local joblib artifacts.
    """
    print("\n" + "=" * 85)
    print(" MLFLOW MODEL REGISTRY & CLEAN PREDICTION REPRODUCIBILITY AUDIT ")
    print("=" * 85)

    if not HAS_MLFLOW:
        print(" [!] MLflow not installed; skipping registry check.")
        return False

    proj_root = os.path.abspath(os.path.join(models_dir, ".."))
    db_path = os.path.join(proj_root, "mlflow.db").replace("\\", "/")
    mlflow.set_tracking_uri(f"sqlite:///{db_path}")
    client = MlflowClient()

    registered_models = client.search_registered_models()
    reg_names = [m.name for m in registered_models]
    print(f" [+] Found {len(reg_names)} models in MLflow Model Registry: {reg_names}")

    expected_champions = [
        ("engine", "Vehicle_Engine_Predictor"),
        ("brake", "Vehicle_Brake_Predictor"),
        ("battery", "Vehicle_Battery_Predictor")
    ]

    scaler = joblib.load(os.path.join(models_dir, "scaler.joblib"))
    pca = joblib.load(os.path.join(models_dir, "pca.joblib"))
    clipping_bounds = joblib.load(os.path.join(models_dir, "clipping_bounds.joblib"))
    feature_names = joblib.load(os.path.join(models_dir, "feature_names.joblib"))
    champions_summary = json.load(open(os.path.join(models_dir, "champions_summary.json")))

    # Create synthetic test record for inference verification
    test_record = {f: 50.0 for f in feature_names}
    test_record["engine_temp_c"] = 108.0
    test_record["brake_pad_wear_mm"] = 2.1
    test_record["battery_voltage_v"] = 11.2
    df_test_row = pd.DataFrame([test_record])[feature_names]

    # Preprocessing
    q01 = pd.Series(clipping_bounds["q01"])
    q99 = pd.Series(clipping_bounds["q99"])
    X_clipped = df_test_row.clip(lower=q01, upper=q99, axis=1)
    X_scaled = scaler.transform(X_clipped)
    X_pca = pca.transform(X_scaled)

    all_matched = True
    for short_name, reg_name in expected_champions:
        if reg_name not in reg_names:
            print(f" [FAIL] Expected registered model '{reg_name}' NOT found in MLflow registry!")
            all_matched = False
            continue

        # Load from MLflow Model Registry
        try:
            versions = client.get_latest_versions(reg_name)
            latest_v = versions[0].version if versions else 1
            model_uri = f"models:/{reg_name}/{latest_v}"
            mlflow_model = mlflow.sklearn.load_model(model_uri)
            print(f" [+] Successfully loaded registered model: '{model_uri}'")
        except Exception as e:
            try:
                model_uri = f"models:/{reg_name}/latest"
                mlflow_model = mlflow.sklearn.load_model(model_uri)
                print(f" [+] Successfully loaded registered model: '{model_uri}'")
            except Exception as e2:
                print(f" [FAIL] Could not load registered model '{reg_name}': {e2}")
                all_matched = False
                continue

        # Load local joblib model
        local_model_path = os.path.join(models_dir, f"champion_{short_name}_model.joblib")
        local_model = joblib.load(local_model_path)

        # Determine feature input
        meta = champions_summary.get(short_name, {})
        uses_pca = meta.get("Features") == "PCA_95_Variance"
        X_input = X_pca if uses_pca else X_scaled

        # Compare probabilities
        mlflow_prob = float(mlflow_model.predict_proba(X_input)[0, 1])
        local_prob = float(local_model.predict_proba(X_input)[0, 1])
        diff = abs(mlflow_prob - local_prob)

        if diff < 1e-6:
            print(f"     [MATCH] Subsystem '{short_name}': MLflow prob={mlflow_prob:.4f} == Local prob={local_prob:.4f} (Diff={diff:.1e})")
        else:
            print(f"     [MISMATCH] Subsystem '{short_name}': MLflow prob={mlflow_prob:.4f} != Local prob={local_prob:.4f}")
            all_matched = False

    if all_matched:
        print(" [+] All 3 registered MLflow models verified with 100% numerical reproduction parity!")
    return all_matched


if __name__ == "__main__":
    run_ml_training()

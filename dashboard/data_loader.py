"""
AI-Powered Connected Vehicle Analytics Platform (BDA)
Dashboard Data Loading and Real-Time Stream Ingestion Engine
"""

import os
import sys
import time
import glob
import json
import datetime
import pandas as pd
import numpy as np

# Ensure project root is in sys.path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from data_generator.generate_telematics import generate_vehicle_master, VehicleState
from ml_pipeline.predictor import get_predictor

def emit_realtime_batch(records_per_batch: int = 20, inject_faults: bool = True):
    """
    Generates and persists a real-time streaming JSON micro-batch into data/landing/.
    If inject_faults=True, deliberately includes active critical component failures
    so the live monitoring dashboard immediately demonstrates early-warning triggers.
    """
    master_path = "data/vehicle_master.csv"
    if os.path.exists(master_path):
        df_m = pd.read_csv(master_path)
    else:
        df_m = generate_vehicle_master(50)
        os.makedirs("data", exist_ok=True)
        df_m.to_csv(master_path, index=False)
        
    master_rows = df_m.to_dict(orient="records")
    fleet = [VehicleState(row) for row in master_rows]
    
    # Inject deliberate faults into vehicles for live demo impact (both Critical and Warning)
    if inject_faults and len(fleet) >= 6:
        # Critical vehicles (severe degradation >= 0.70)
        fleet[0].failure_mode = "Engine_Overheat"
        fleet[0].degradation = 0.85
        fleet[1].failure_mode = "Brake_Pad_Worn"
        fleet[1].degradation = 0.90
        fleet[2].failure_mode = "Battery_Degraded"
        fleet[2].degradation = 0.80

        # Warning vehicles (early wear / inspection range 0.32 - 0.45)
        fleet[3].failure_mode = "Engine_Overheat"
        fleet[3].degradation = 0.36
        fleet[4].failure_mode = "Brake_Pad_Worn"
        fleet[4].degradation = 0.34
        fleet[5].failure_mode = "Battery_Degraded"
        fleet[5].degradation = 0.35

    os.makedirs("data/landing", exist_ok=True)
    batch_records = []
    now_iso = datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
    
    for i in range(records_per_batch):
        # Pick injected vehicles first (3 critical, 3 warning), then random healthy
        v = fleet[i % 6] if (inject_faults and i < 6) else np.random.choice(fleet)
        record, _ = v.step(delta_seconds=3)
        record["timestamp"] = now_iso
        batch_records.append(record)
        
    fname = f"stream_{int(time.time())}_batch_{len(os.listdir('data/landing'))+1:05d}.json"
    fpath = os.path.join("data/landing", fname)
    with open(fpath, "w", encoding="utf-8") as f:
        for rec in batch_records:
            f.write(json.dumps(rec) + "\n")
            
    return fname, len(batch_records), now_iso

def load_fleet_data():
    """
    Ingests all available streaming micro-batches from data/landing/,
    deduplicates to the latest telemetry state per vehicle, and evaluates
    ML failure probabilities and physical safety alert overrides.
    """
    master_path = "data/vehicle_master.csv"
    df_master = pd.read_csv(master_path) if os.path.exists(master_path) else pd.DataFrame()

    landing_files = glob.glob("data/landing/*.json")
    records = []
    latest_mtime = 0
    if landing_files:
        for fpath in landing_files:
            m = os.path.getmtime(fpath)
            if m > latest_mtime:
                latest_mtime = m
            with open(fpath, "r", encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        records.append(json.loads(line))
        df_stream = pd.DataFrame(records)
    else:
        hist_path = "data/historical_labeled_telematics.parquet"
        df_stream = pd.read_parquet(hist_path).tail(500) if os.path.exists(hist_path) else pd.DataFrame()

    if not df_stream.empty and not df_master.empty:
        df_merged = df_stream.merge(df_master, on="vehicle_id", how="left")
        if "timestamp" in df_merged.columns:
            df_merged["timestamp"] = pd.to_datetime(df_merged["timestamp"])
            df_latest = df_merged.sort_values("timestamp").groupby("vehicle_id").last().reset_index()
        else:
            df_latest = df_merged.groupby("vehicle_id").last().reset_index()
    else:
        df_latest = pd.DataFrame()

    if not df_latest.empty:
        predictor = get_predictor()
        preds = []
        for _, row in df_latest.iterrows():
            res = predictor.predict_record(row.to_dict())
            preds.append(res)
            
        df_latest["failure_probability"] = [p["failure_probability"] if p["failure_probability"] is not None else 0.0 for p in preds]
        df_latest["status"] = [p["status"] for p in preds]
        df_latest["predicted_failure_type"] = [p["predicted_failure_type"] for p in preds]
        df_latest["engine_risk_score"] = [(p["subsystem_probabilities"]["engine"] * 100) if (p.get("subsystem_probabilities") and p["subsystem_probabilities"].get("engine") is not None) else 0.0 for p in preds]
        df_latest["brake_risk_score"] = [(p["subsystem_probabilities"]["brake"] * 100) if (p.get("subsystem_probabilities") and p["subsystem_probabilities"].get("brake") is not None) else 0.0 for p in preds]
        df_latest["battery_risk_score"] = [(p["subsystem_probabilities"]["battery"] * 100) if (p.get("subsystem_probabilities") and p["subsystem_probabilities"].get("battery") is not None) else 0.0 for p in preds]
        df_latest["overall_risk_score"] = [(p["failure_probability"] * 100) if p["failure_probability"] is not None else 0.0 for p in preds]
        df_latest["prediction_valid"] = [p.get("prediction_valid", True) for p in preds]
        df_latest["ml_status"] = [p.get("ml_status", "AVAILABLE") for p in preds]
        df_latest["safety_alerts"] = [p.get("safety_alerts", []) for p in preds]

    return df_latest, len(records), latest_mtime

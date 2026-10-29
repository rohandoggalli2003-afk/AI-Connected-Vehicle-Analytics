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

# Resolve data paths from the repository root, not the shell's current directory.
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
LANDING_DIR = os.path.join(DATA_DIR, "landing")
MASTER_PATH = os.path.join(DATA_DIR, "vehicle_master.csv")
HISTORICAL_CSV_PATH = os.path.join(DATA_DIR, "historical_telematics_with_missing_values.csv")
HISTORICAL_PARQUET_PATH = os.path.join(DATA_DIR, "historical_labeled_telematics.parquet")

# Persistent vehicle state for continuous realistic streaming simulation
_persistent_fleet = None

def get_simulation_fleet(inject_faults: bool = True):
    global _persistent_fleet
    if _persistent_fleet is None:
        master_path = MASTER_PATH
        if os.path.exists(master_path):
            df_m = pd.read_csv(master_path)
        else:
            df_m = generate_vehicle_master(50)
            os.makedirs(DATA_DIR, exist_ok=True)
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
            
        _persistent_fleet = fleet
    return _persistent_fleet

def emit_realtime_batch(records_per_batch: int = 20, inject_faults: bool = True):
    """
    Generates and persists a real-time streaming JSON micro-batch into data/landing/.
    If inject_faults=True, deliberately includes active critical component failures
    so the live monitoring dashboard immediately demonstrates early-warning triggers.
    """
    fleet = get_simulation_fleet(inject_faults=inject_faults)
    os.makedirs(LANDING_DIR, exist_ok=True)
    batch_records = []
    now_iso = datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
    
    for i in range(records_per_batch):
        # Pick injected vehicles first (3 critical, 3 warning), then random healthy
        v = fleet[i % 6] if (inject_faults and i < 6) else np.random.choice(fleet)
        record, _ = v.step(delta_seconds=3)
        record["timestamp"] = now_iso
        batch_records.append(record)
        
    fname = f"stream_{time.time_ns()}_batch_{len(os.listdir(LANDING_DIR))+1:05d}.json"
    fpath = os.path.join(LANDING_DIR, fname)
    with open(fpath, "w", encoding="utf-8") as f:
        for rec in batch_records:
            f.write(json.dumps(rec) + "\n")
            
    return fname, len(batch_records), now_iso

# Global incremental cache to make live auto-refresh ultra fast (< 0.2s)
_fleet_cache = {
    "file_mtimes": {},
    "total_pings": 0,
    "latest_mtime": 0,
    "latest_records": {},     # vid -> record dict
    "predictions": {},        # vid -> prediction dict
    "cached_df": None
}

def reset_fleet_cache():
    """Resets cache and forces full re-ingestion from data/landing and vehicle_master.csv."""
    global _fleet_cache, _persistent_fleet
    _persistent_fleet = None
    _fleet_cache = {
        "file_mtimes": {},
        "total_pings": 0,
        "latest_mtime": 0,
        "latest_records": {},
        "predictions": {},
        "cached_df": None
    }

def load_fleet_data(force_reload: bool = False):
    """
    Ingests all available streaming micro-batches from data/landing/,
    deduplicates to the latest telemetry state per vehicle, and evaluates
    ML failure probabilities and physical safety alert overrides.
    Uses an incremental in-memory cache for ultra-responsive live dashboard auto-refresh.
    """
    global _fleet_cache

    if force_reload:
        reset_fleet_cache()

    master_path = MASTER_PATH
    df_master = pd.read_csv(master_path) if os.path.exists(master_path) else pd.DataFrame()

    landing_files = glob.glob(os.path.join(LANDING_DIR, "*.json"))

    # If no landing files exist, fallback to historical dataset
    if not landing_files:
        hist_path = HISTORICAL_CSV_PATH
        if os.path.exists(hist_path):
            df_stream = pd.read_csv(hist_path).tail(500)
        else:
            hist_pq = HISTORICAL_PARQUET_PATH
            df_stream = pd.read_parquet(hist_pq).tail(500) if os.path.exists(hist_pq) else pd.DataFrame()

        if not df_stream.empty and not df_master.empty:
            df_merged = df_stream.merge(df_master, on="vehicle_id", how="left")
            df_latest = df_merged.sort_values("timestamp").groupby("vehicle_id").last().reset_index()
        else:
            df_latest = pd.DataFrame()
            
        if not df_latest.empty:
            predictor = get_predictor()
            preds = [predictor.predict_record(row.to_dict()) for _, row in df_latest.iterrows()]
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
            df_latest["profile"] = [p.get("profile") or row.get("profile", "A") for p, (_, row) in zip(preds, df_latest.iterrows())]
            df_latest["electrical_system"] = [p.get("electrical_system") or row.get("electrical_system", "12V") for p, (_, row) in zip(preds, df_latest.iterrows())]
            df_latest["brake_configuration"] = [p.get("brake_configuration") or row.get("brake_configuration", "Hydraulic Disc") for p, (_, row) in zip(preds, df_latest.iterrows())]
            df_latest["configuration_status"] = [p.get("configuration_status", "VALID") for p in preds]
            df_latest["configuration_flags"] = [p.get("configuration_flags", []) for p in preds]
            df_latest["data_quality_status"] = [p.get("data_quality_status", "HIGH") for p in preds]
            df_latest["imputed_sensors"] = [p.get("imputed_sensors", []) for p in preds]
            df_latest["imputed_count"] = [p.get("imputed_count", 0) for p in preds]
            df_latest["sensor_errors"] = [p.get("sensor_errors", []) for p in preds]
            df_latest["data_quality_warning"] = [p.get("data_quality_warning", "") for p in preds]
            
            # Geographic coordinate resolution
            df_latest["latitude"] = pd.to_numeric(df_latest.get("gps_latitude", df_latest.get("base_lat", 0.0)), errors="coerce").fillna(0.0)
            df_latest["longitude"] = pd.to_numeric(df_latest.get("gps_longitude", df_latest.get("base_lon", 0.0)), errors="coerce").fillna(0.0)
            df_latest["map_color"] = df_latest["status"].map({"CRITICAL": "#ef4444", "WARNING": "#f59e0b", "HEALTHY": "#10b981"}).fillna("#10b981")
            df_latest["map_color_rgb"] = df_latest["status"].map({"CRITICAL": [239, 68, 68, 220], "WARNING": [245, 158, 11, 200], "HEALTHY": [16, 185, 129, 180]})
            df_latest["map_radius"] = df_latest["status"].map({"CRITICAL": 35000, "WARNING": 25000, "HEALTHY": 18000}).fillna(18000)
            df_latest["map_size"] = df_latest["status"].map({"CRITICAL": 35, "WARNING": 25, "HEALTHY": 15}).fillna(15)

        return df_latest, len(df_stream), 0

    # Fast detection of new or modified landing files
    new_or_modified = []
    current_max_mtime = 0
    for fpath in landing_files:
        try:
            m = os.path.getmtime(fpath)
            if m > current_max_mtime:
                current_max_mtime = m
            if fpath not in _fleet_cache["file_mtimes"] or _fleet_cache["file_mtimes"][fpath] != m:
                new_or_modified.append((fpath, m))
        except OSError:
            continue

    # Fast-path cache hit: no files changed, return cached DataFrame in < 1ms
    if not new_or_modified and _fleet_cache["cached_df"] is not None and not _fleet_cache["cached_df"].empty:
        return _fleet_cache["cached_df"].copy(), _fleet_cache["total_pings"], _fleet_cache["latest_mtime"]

    # Ingest new/modified JSON files incrementally with validation
    dirty_vehicles = set()
    for fpath, m in new_or_modified:
        _fleet_cache["file_mtimes"][fpath] = m
        try:
            with open(fpath, "r", encoding="utf-8") as f:
                for line in f:
                    if not line.strip():
                        continue
                    try:
                        rec = json.loads(line)
                    except (json.JSONDecodeError, TypeError):
                        # Ignore malformed lines without stopping ingestion of the batch.
                        continue
                    if not isinstance(rec, dict):
                        continue
                    _fleet_cache["total_pings"] += 1
                    vid = rec.get("vehicle_id")
                    if not isinstance(vid, str) or not vid.strip():
                        continue

                    prev = _fleet_cache["latest_records"].get(vid)
                    rec_ts = pd.to_datetime(rec.get("timestamp"), utc=True, errors="coerce")
                    prev_ts = pd.to_datetime(prev.get("timestamp"), utc=True, errors="coerce") if prev else pd.NaT
                    # Accept the first record or a record with a valid timestamp that is
                    # at least as recent as the previously retained record. If timestamps
                    # are absent/invalid, retain the first valid record rather than guessing.
                    if prev is None or (pd.notna(rec_ts) and (pd.isna(prev_ts) or rec_ts >= prev_ts)):
                        _fleet_cache["latest_records"][vid] = rec
                        dirty_vehicles.add(vid)
        except OSError:
            # A file may be moved/deleted while the dashboard is refreshing.
            continue

    _fleet_cache["latest_mtime"] = current_max_mtime

    # Run trained-model inference for each vehicle whose latest telemetry changed.
    # This is prediction on incoming data; it does not retrain the models.
    predictor = get_predictor()
    for vid in dirty_vehicles:
        rec = _fleet_cache["latest_records"][vid]
        # Always synchronize authoritative metadata from vehicle_master
        if not df_master.empty:
            m_row = df_master[df_master["vehicle_id"] == vid]
            if not m_row.empty:
                m_dict = m_row.iloc[0].to_dict()
                for k in ["brand", "model", "vehicle_year", "fuel_type", "base_hub", "base_lat", "base_lon", "profile", "electrical_system", "brake_configuration"]:
                    if k in m_dict:
                        rec[k] = m_dict[k]
        _fleet_cache["predictions"][vid] = predictor.predict_record(rec)

    # Build consolidated DataFrame
    rows = []
    for vid, rec in _fleet_cache["latest_records"].items():
        row = dict(rec)
        # Ensure latest vehicle_master data is attached to all output rows
        if not df_master.empty:
            m_row = df_master[df_master["vehicle_id"] == vid]
            if not m_row.empty:
                m_dict = m_row.iloc[0].to_dict()
                for k in ["brand", "model", "vehicle_year", "fuel_type", "base_hub", "base_lat", "base_lon", "profile", "electrical_system", "brake_configuration"]:
                    if k in m_dict:
                        row[k] = m_dict[k]

        p = _fleet_cache["predictions"].get(vid, {})
        row["failure_probability"] = p.get("failure_probability") if p.get("failure_probability") is not None else 0.0
        row["status"] = p.get("status", "HEALTHY")
        row["predicted_failure_type"] = p.get("predicted_failure_type", "Healthy")
        sp = p.get("subsystem_probabilities") or {}
        row["engine_risk_score"] = (sp.get("engine") * 100) if sp.get("engine") is not None else 0.0
        row["brake_risk_score"] = (sp.get("brake") * 100) if sp.get("brake") is not None else 0.0
        row["battery_risk_score"] = (sp.get("battery") * 100) if sp.get("battery") is not None else 0.0
        row["overall_risk_score"] = (p.get("failure_probability") * 100) if p.get("failure_probability") is not None else 0.0
        row["prediction_valid"] = p.get("prediction_valid", True)
        row["ml_status"] = p.get("ml_status", "AVAILABLE")
        row["safety_alerts"] = p.get("safety_alerts", [])
        row["profile"] = p.get("profile") or row.get("profile", "A")
        row["electrical_system"] = p.get("electrical_system") or row.get("electrical_system", "12V")
        row["brake_configuration"] = p.get("brake_configuration") or row.get("brake_configuration", "Hydraulic Disc")
        row["configuration_status"] = p.get("configuration_status", "VALID")
        row["configuration_flags"] = p.get("configuration_flags", [])
        row["data_quality_status"] = p.get("data_quality_status", "HIGH")
        row["imputed_sensors"] = p.get("imputed_sensors", [])
        row["imputed_count"] = p.get("imputed_count", 0)
        row["sensor_errors"] = p.get("sensor_errors", [])
        row["data_quality_warning"] = p.get("data_quality_warning", "")

        # Geographic coordinates for live interactive map
        try:
            lat = float(rec.get("gps_latitude", 0.0))
        except (ValueError, TypeError):
            lat = 0.0
        try:
            lon = float(rec.get("gps_longitude", 0.0))
        except (ValueError, TypeError):
            lon = 0.0

        if (lat == 0.0 or pd.isna(lat)) and "base_lat" in row:
            try:
                lat = float(row.get("base_lat", 0.0))
            except (ValueError, TypeError):
                lat = 0.0
        if (lon == 0.0 or pd.isna(lon)) and "base_lon" in row:
            try:
                lon = float(row.get("base_lon", 0.0))
            except (ValueError, TypeError):
                lon = 0.0

        row["latitude"] = lat
        row["longitude"] = lon
        row["base_hub"] = row.get("base_hub", "Fleet Hub")

        # Visual styling attributes for Streamlit and PyDeck maps
        stat = row["status"]
        if stat == "CRITICAL":
            row["map_color"] = "#ef4444"
            row["map_color_rgb"] = [239, 68, 68, 220]
            row["map_radius"] = 35000
            row["map_size"] = 35
        elif stat == "WARNING":
            row["map_color"] = "#f59e0b"
            row["map_color_rgb"] = [245, 158, 11, 200]
            row["map_radius"] = 25000
            row["map_size"] = 25
        else:
            row["map_color"] = "#10b981"
            row["map_color_rgb"] = [16, 185, 129, 180]
            row["map_radius"] = 18000
            row["map_size"] = 15

        rows.append(row)

    df_latest = pd.DataFrame(rows)
    _fleet_cache["cached_df"] = df_latest
    return df_latest.copy(), _fleet_cache["total_pings"], _fleet_cache["latest_mtime"]

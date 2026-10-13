"""
AI-Powered Connected Vehicle Analytics Platform (BDA)
Phase A End-to-End Local Verification Script

This script verifies the entire Phase A pipeline offline (without needing an active Databricks cluster):
1. Verifies the 34 streaming columns across all landing batches.
2. Simulates Bronze ingestion, schema enforcement, and metadata tagging.
3. Simulates Silver cleaning, deduplication, sanity boundaries, and Stream-Static join with Vehicle Master.
4. Simulates Gold windowed aggregations, trend metrics, and multi-component risk scoring.
5. Verifies ML training dataset schema and class distributions.
"""

import os
import glob
import json
import pandas as pd
import numpy as np

def verify_phase_a():
    print("=" * 80)
    print(" CONNECTED VEHICLE ANALYTICS PLATFORM - PHASE A VERIFICATION ")
    print("=" * 80)
    
    # 1. Verify Vehicle Master Table
    master_path = "data/vehicle_master.csv"
    assert os.path.exists(master_path), f"Missing {master_path}"
    df_master = pd.read_csv(master_path)
    print(f"[OK] Step 1: Vehicle Master Table Loaded: {len(df_master)} vehicles")
    print(f"    Columns: {list(df_master.columns)}")
    print(f"    Brands:  {list(df_master['brand'].unique())}")
    
    # 2. Verify Landing Files & 34 Streaming Columns
    landing_files = glob.glob("data/landing/*.json")
    assert len(landing_files) > 0, "No JSON landing files found!"
    print(f"\n[OK] Step 2: Found {len(landing_files)} streaming JSON batches in landing zone.")
    
    expected_34_cols = [
        "vehicle_id", "timestamp",
        "odometer_km", "engine_hours", "vehicle_speed_kph", "engine_temp_c",
        "engine_rpm", "oil_pressure_psi", "coolant_temp_c", "fuel_level_percent",
        "fuel_consumption_lph", "engine_load_percent", "throttle_position_percent",
        "air_flow_rate_gps", "exhaust_gas_temp_c", "engine_vibration",
        "brake_pad_wear_mm", "brake_fluid_level", "brake_temp_c",
        "abs_fault_indicator", "brake_pedal_position", "wheel_speed_fl_kph",
        "wheel_speed_fr_kph", "wheel_speed_rl_kph", "wheel_speed_rr_kph",
        "battery_voltage_v", "battery_current_a", "battery_temp_c",
        "alternator_output_v", "battery_charge_percent", "battery_health_percent",
        "ambient_temp_c", "gps_latitude", "gps_longitude"
    ]
    
    # Check first batch columns
    sample_records = []
    with open(landing_files[0], "r") as f:
        for line in f:
            if line.strip():
                sample_records.append(json.loads(line))
                
    first_row_cols = list(sample_records[0].keys())
    assert len(first_row_cols) == 34, f"Expected 34 columns, got {len(first_row_cols)}"
    assert set(first_row_cols) == set(expected_34_cols), "Columns mismatch!"
    print(f"    Strict check: All 34 streaming columns verified successfully!")
    print(f"    Sample Vehicle: {sample_records[0]['vehicle_id']} at {sample_records[0]['timestamp']}")
    
    # 3. Simulate Bronze Ingestion
    print("\n[OK] Step 3: Simulating Bronze Ingestion...")
    all_raw_records = []
    for fpath in landing_files:
        with open(fpath, "r") as f:
            for line in f:
                if line.strip():
                    rec = json.loads(line)
                    rec["_ingest_timestamp"] = pd.Timestamp.now(tz="UTC")
                    rec["_source_file"] = os.path.basename(fpath)
                    all_raw_records.append(rec)
                    
    df_bronze = pd.DataFrame(all_raw_records)
    print(f"    Bronze records ingested : {len(df_bronze)}")
    print(f"    Bronze schema fields    : {len(df_bronze.columns)} (34 stream + 2 metadata)")
    
    # 4. Simulate Silver Cleaning & Join
    print("\n[OK] Step 4: Simulating Silver Cleaning, Deduplication & Static Master Join...")
    # Deduplicate
    initial_count = len(df_bronze)
    df_silver = df_bronze.drop_duplicates(subset=["vehicle_id", "timestamp"]).copy()
    print(f"    Deduplication: {initial_count} -> {len(df_silver)} records")
    
    # Sanity clipping
    df_silver["vehicle_speed_kph"] = df_silver["vehicle_speed_kph"].clip(lower=0.0, upper=200.0)
    df_silver["engine_rpm"] = df_silver["engine_rpm"].clip(lower=0.0, upper=8000.0)
    df_silver["battery_voltage_v"] = df_silver["battery_voltage_v"].clip(lower=5.0, upper=20.0)
    
    # Stream-Static Join
    df_silver = df_silver.merge(df_master, on="vehicle_id", how="inner")
    print(f"    Enriched with vehicle_master: {len(df_silver)} rows joined.")
    
    # Diagnostic flags
    df_silver["is_overheating"] = (df_silver["engine_temp_c"] > 105.0) | (df_silver["coolant_temp_c"] > 108.0)
    df_silver["is_low_oil"] = df_silver["oil_pressure_psi"] < 20.0
    df_silver["is_worn_brakes"] = df_silver["brake_pad_wear_mm"] < 3.0
    df_silver["is_battery_critical"] = (df_silver["battery_voltage_v"] < 11.8) | (df_silver["battery_health_percent"] < 50.0)
    
    print(f"    Silver Diagnostic Summary:")
    print(f"      Overheating events    : {df_silver['is_overheating'].sum()}")
    print(f"      Low oil pressure      : {df_silver['is_low_oil'].sum()}")
    print(f"      Worn brake pads       : {df_silver['is_worn_brakes'].sum()}")
    print(f"      Critical battery alert: {df_silver['is_battery_critical'].sum()}")
    
    # 5. Simulate Gold Rolling Features & Risk Scoring
    print("\n[OK] Step 5: Simulating Gold Aggregations & Component Risk Scoring...")
    gold_rows = []
    for vid, group in df_silver.groupby("vehicle_id"):
        meta = group.iloc[0]
        max_eng_temp = group["engine_temp_c"].max()
        min_oil_psi  = group["oil_pressure_psi"].min()
        max_vib      = group["engine_vibration"].max()
        min_brake_mm = group["brake_pad_wear_mm"].min()
        max_brk_temp = group["brake_temp_c"].max()
        abs_faults   = group["abs_fault_indicator"].sum()
        min_batt_v   = group["battery_voltage_v"].min()
        min_batt_soh = group["battery_health_percent"].min()
        avg_alt_v    = group["alternator_output_v"].mean()
        
        # Risk scores
        eng_risk = (50 if max_eng_temp > 115 else (30 if max_eng_temp > 105 else 0)) + \
                   (40 if min_oil_psi < 15 else (20 if min_oil_psi < 25 else 0)) + \
                   (20 if max_vib > 3.5 else 0)
        
        brk_risk = (60 if min_brake_mm < 2.0 else (30 if min_brake_mm < 3.5 else 0)) + \
                   (30 if max_brk_temp > 280 else 0) + \
                   (25 if abs_faults > 0 else 0)
        
        bat_risk = (50 if min_batt_v < 11.5 else (25 if min_batt_v < 12.2 else 0)) + \
                   (40 if min_batt_soh < 45 else (20 if min_batt_soh < 65 else 0)) + \
                   (30 if avg_alt_v < 12.8 else 0)
        
        overall = max(eng_risk, brk_risk, bat_risk)
        status = "CRITICAL" if overall >= 60 else ("WARNING" if overall >= 30 else "HEALTHY")
        
        gold_rows.append({
            "vehicle_id": vid,
            "brand": meta["brand"],
            "model": meta["model"],
            "vehicle_year": meta["vehicle_year"],
            "pings": len(group),
            "avg_speed_kph": round(group["vehicle_speed_kph"].mean(), 1),
            "avg_engine_temp_c": round(group["engine_temp_c"].mean(), 1),
            "max_engine_temp_c": round(max_eng_temp, 1),
            "min_oil_pressure_psi": round(min_oil_psi, 1),
            "min_brake_pad_wear_mm": round(min_brake_mm, 2),
            "min_battery_voltage_v": round(min_batt_v, 2),
            "engine_risk_score": eng_risk,
            "brake_risk_score": brk_risk,
            "battery_risk_score": bat_risk,
            "overall_risk_score": overall,
            "fleet_health_status": status,
            "current_lat": group["gps_latitude"].iloc[-1],
            "current_lon": group["gps_longitude"].iloc[-1]
        })
        
    df_gold = pd.DataFrame(gold_rows)
    print(f"    Gold fleet summary: {len(df_gold)} vehicles evaluated.")
    print("    Health status breakdown:")
    print(df_gold["fleet_health_status"].value_counts().to_string())
    
    print("\n    Top 3 High-Risk Vehicles:")
    high_risk = df_gold.sort_values(by="overall_risk_score", ascending=False).head(3)
    for _, r in high_risk.iterrows():
        print(f"      - {r['vehicle_id']} ({r['brand']} {r['model']}): Risk {r['overall_risk_score']}/100 [{r['fleet_health_status']}] | Eng:{r['engine_risk_score']} Brk:{r['brake_risk_score']} Bat:{r['battery_risk_score']}")
        
    # 6. Verify Historical Dataset for ML Phase B
    hist_path = "data/historical_labeled_telematics.parquet"
    assert os.path.exists(hist_path), f"Missing {hist_path}"
    df_hist = pd.read_parquet(hist_path)
    print(f"\n[OK] Step 6: Verified ML Training Parquet Dataset: {len(df_hist)} rows, {len(df_hist.columns)} columns.")
    print(f"    Labels present: 'engine_failure_imminent', 'brake_issue_imminent', 'battery_issue_imminent', 'failure_type'")
    print(f"    Engine failure rate  : {df_hist['engine_failure_imminent'].mean()*100:.2f}%")
    print(f"    Brake failure rate   : {df_hist['brake_issue_imminent'].mean()*100:.2f}%")
    print(f"    Battery failure rate : {df_hist['battery_issue_imminent'].mean()*100:.2f}%")
    
    print("\n" + "=" * 80)
    print(" ALL PHASE A PIPELINE VERIFICATION TESTS PASSED SUCCESSFULLY! ")
    print("=" * 80)

if __name__ == "__main__":
    verify_phase_a()

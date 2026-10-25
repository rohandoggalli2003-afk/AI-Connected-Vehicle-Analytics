"""
AI-Powered Connected Vehicle Analytics Platform (BDA)
Vehicle Configuration & Health State Verification Suite

Validates:
1. All 50 vehicle mappings (IDs, profiles, electrical systems, brake configurations)
2. Configuration loading (JSON profiles A-D, OEM heuristic overrides like XUV700)
3. Nominal / Warning / Critical sensor telemetry readings across 12V and 24V architectures
4. Missing & Stale telemetry data quality auditing
5. Unknown / Unsupported vehicle configurations (threshold suspension & warning flagging)
"""

import os
import sys
import json
import datetime
import pandas as pd
import numpy as np

# Ensure project root is in sys.path
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from ml_pipeline.predictor import get_predictor, UnifiedVehiclePredictor
from ml_pipeline.data_cleaner import LiveCausalVehicleBuffer, SUPPORTED_ELECTRICAL_SYSTEMS, SUPPORTED_BRAKE_CONFIGURATIONS


def run_vehicle_configuration_tests():
    print("=" * 85)
    print(" VEHICLE CONFIGURATION & TELEMETRY THRESHOLD VERIFICATION SUITE ")
    print("=" * 85)
    
    total_tests = 0
    passed_tests = 0
    
    # -------------------------------------------------------------------------
    # TEST SUITE 1: ALL 50 VEHICLE MAPPINGS VERIFICATION
    # -------------------------------------------------------------------------
    print("\n[TEST GROUP 1] Verifying all 50 Vehicle Mappings...")
    total_tests += 1
    
    csv_path = os.path.join(PROJECT_ROOT, "data", "vehicle_master.csv")
    assert os.path.exists(csv_path), f"Missing {csv_path}"
    df_master = pd.read_csv(csv_path)
    
    assert len(df_master) == 50, f"Expected 50 vehicles in vehicle_master.csv, got {len(df_master)}"
    
    expected_vids = [f"VH-{1001 + i}" for i in range(50)]
    actual_vids = list(df_master["vehicle_id"])
    assert actual_vids == expected_vids, "Vehicle ID sequence mismatch!"
    
    required_cols = [
        "vehicle_id", "brand", "vehicle_year", "model", "fuel_type",
        "base_hub", "base_lat", "base_lon", "profile", "electrical_system", "brake_configuration"
    ]
    for col in required_cols:
        assert col in df_master.columns, f"Missing required column '{col}' in vehicle_master.csv"
        assert df_master[col].isna().sum() == 0, f"Found null values in column '{col}'"

    # Verify distribution
    profile_counts = df_master["profile"].value_counts().to_dict()
    elec_counts = df_master["electrical_system"].value_counts().to_dict()
    brake_counts = df_master["brake_configuration"].value_counts().to_dict()
    
    print(f" [+] Verified 50 vehicles (VH-1001 to VH-1050):")
    print(f"     - Profiles: {profile_counts}")
    print(f"     - Electrical Systems: {elec_counts}")
    print(f"     - Brake Configurations: {brake_counts}")
    
    assert elec_counts.get("24V", 0) == 8, f"Expected exactly 8 heavy commercial 24V trucks (Profile D), got {elec_counts.get('24V')}"
    assert elec_counts.get("12V", 0) == 42, f"Expected 42 12V vehicles, got {elec_counts.get('12V')}"
    
    passed_tests += 1
    print(" [PASS] Test 1: All 50 vehicle static records, IDs, and column specifications verified.")

    # -------------------------------------------------------------------------
    # TEST SUITE 2: CONFIGURATION LOADING & OEM HEURISTICS
    # -------------------------------------------------------------------------
    print("\n[TEST GROUP 2] Verifying Configuration Loading & Overrides...")
    total_tests += 1
    
    json_path = os.path.join(PROJECT_ROOT, "config", "vehicle_thresholds.json")
    assert os.path.exists(json_path), f"Missing {json_path}"
    with open(json_path, "r", encoding="utf-8") as f:
        thresh_cfg = json.load(f)
        
    assert set(thresh_cfg["supported_electrical_systems"]) == {"12V", "24V"}
    assert set(thresh_cfg["supported_brake_configurations"]) == {
        "Hydraulic Disc", "Hydraulic Disc/Drum", "Pneumatic/Hydraulic Disc", "Pneumatic S-Cam/Disc"
    }
    assert set(thresh_cfg["profiles"].keys()) == {"A", "B", "C", "D"}
    assert len(thresh_cfg["vehicles"]) == 50
    
    # Verify alignment between CSV and JSON for all 50 vehicles
    json_veh_map = {v["vehicle_id"]: v for v in thresh_cfg["vehicles"]}
    for _, row in df_master.iterrows():
        vid = row["vehicle_id"]
        assert vid in json_veh_map, f"Vehicle {vid} missing in vehicle_thresholds.json"
        jv = json_veh_map[vid]
        assert jv["profile"] == row["profile"], f"Profile mismatch for {vid}"
        assert jv["electrical_system"] == row["electrical_system"], f"Electrical mismatch for {vid}"
        assert jv["brake_configuration"] == row["brake_configuration"], f"Brake mismatch for {vid}"

    # Verify Mahindra XUV700 documented manual heuristic (113C warn / 115C crit)
    xuv = json_veh_map["VH-1002"]
    assert xuv["engine_warning_temp_c"] == 113.0 and xuv["engine_critical_temp_c"] == 115.0
    assert xuv["threshold_status"] == "documented_manual_heuristic"
    print(f" [+] Verified Mahindra XUV700 (VH-1002) heuristic thresholds: 113°C Warning / 115°C Critical.")

    # Verify Profile D heavy commercial trucks (24V, 23.6V warn / 22.0V crit, 6.4mm warn / 3.2mm crit)
    for vid in ["VH-1006", "VH-1015", "VH-1023", "VH-1027", "VH-1034", "VH-1036", "VH-1042", "VH-1050"]:
        truck = json_veh_map[vid]
        assert truck["profile"] == "D"
        assert truck["electrical_system"] == "24V"
        assert truck["battery_warning_voltage_v"] == 23.6 and truck["battery_critical_voltage_v"] == 22.0
        assert truck["brake_warning_thickness_mm"] == 6.4 and truck["brake_critical_thickness_mm"] == 3.2
    print(f" [+] Verified all 8 Profile D Heavy Commercial Trucks configured for 24V architecture.")

    passed_tests += 1
    print(" [PASS] Test 2: Configuration loading, profile specifications, and vehicle alignment verified.")

    # -------------------------------------------------------------------------
    # TEST SUITE 3: NOMINAL SENSOR TELEMETRY READINGS (12V & 24V)
    # -------------------------------------------------------------------------
    print("\n[TEST GROUP 3] Testing Nominal (Healthy) Telemetry Readings...")
    total_tests += 1
    
    predictor = get_predictor()
    
    # Test nominal 12V passenger vehicle (VH-1001 Tata Nexon)
    rec_12v_nom = {
        "vehicle_id": "VH-1001",
        "engine_temp_c": 88.0,
        "oil_pressure_psi": 45.0,
        "brake_pad_wear_mm": 9.5,
        "battery_voltage_v": 13.7,
        "alternator_output_v": 14.1,
        "battery_health_percent": 95.0
    }
    res_12v_nom = predictor.predict_record(rec_12v_nom)
    assert res_12v_nom["status"] == "HEALTHY", f"Expected HEALTHY, got {res_12v_nom['status']}"
    assert len(res_12v_nom["safety_alerts"]) == 0, f"Expected 0 alerts on nominal 12V, got {res_12v_nom['safety_alerts']}"
    assert res_12v_nom["electrical_system"] == "12V"
    
    # Test nominal 24V commercial truck (VH-1006 BharatBenz 3528C)
    rec_24v_nom = {
        "vehicle_id": "VH-1006",
        "engine_temp_c": 88.0,
        "oil_pressure_psi": 48.0,
        "brake_pad_wear_mm": 16.5,
        "battery_voltage_v": 27.4,
        "alternator_output_v": 28.2,
        "battery_health_percent": 94.0
    }
    res_24v_nom = predictor.predict_record(rec_24v_nom)
    assert res_24v_nom["status"] == "HEALTHY", f"Expected HEALTHY, got {res_24v_nom['status']}"
    assert len(res_24v_nom["safety_alerts"]) == 0, f"Expected 0 alerts on nominal 24V, got {res_24v_nom['safety_alerts']}"
    assert res_24v_nom["electrical_system"] == "24V"
    
    passed_tests += 1
    print(" [PASS] Test 3: Nominal telemetry evaluations verified without false alerts.")

    # -------------------------------------------------------------------------
    # TEST SUITE 4: WARNING TELEMETRY READINGS & COMPARISON LOGIC
    # -------------------------------------------------------------------------
    print("\n[TEST GROUP 4] Testing Warning Threshold Telemetry & Comparison Logic...")
    total_tests += 1
    
    # 4a. High Engine Temperature Warning (eng_temp >= 105.0)
    rec_eng_w = {"vehicle_id": "VH-1001", "engine_temp_c": 107.0, "oil_pressure_psi": 45.0, "brake_pad_wear_mm": 8.0, "battery_voltage_v": 13.7}
    res_eng_w = predictor.predict_record(rec_eng_w)
    assert any("Engine Temperature Warning" in a for a in res_eng_w["safety_alerts"]), "Engine temp warning failed!"
    assert res_eng_w["subsystem_probabilities"]["engine"] >= 0.45

    # 4b. Mahindra XUV700 113.0C Warning Heuristic Check
    # At 108.0C, XUV700 should NOT trigger engine warning (threshold is 113C)
    rec_xuv_sub = {"vehicle_id": "VH-1002", "engine_temp_c": 108.0, "oil_pressure_psi": 45.0, "brake_pad_wear_mm": 8.0, "battery_voltage_v": 13.7}
    res_xuv_sub = predictor.predict_record(rec_xuv_sub)
    assert not any("Engine Temperature Warning" in a for a in res_xuv_sub["safety_alerts"]), "XUV700 falsely warned below 113°C!"
    
    # At 113.5C, XUV700 triggers Engine Temperature Warning
    rec_xuv_w = {"vehicle_id": "VH-1002", "engine_temp_c": 113.5, "oil_pressure_psi": 45.0, "brake_pad_wear_mm": 8.0, "battery_voltage_v": 13.7}
    res_xuv_w = predictor.predict_record(rec_xuv_w)
    assert any("Engine Temperature Warning" in a for a in res_xuv_w["safety_alerts"]), "XUV700 failed to warn at 113.5°C!"

    # 4c. Low Brake Pad Warning (brake_pad <= warn_thickness)
    # Profile A (VH-1001, warn=3.2mm): 3.0mm triggers warning
    rec_brk_a_w = {"vehicle_id": "VH-1001", "engine_temp_c": 88.0, "oil_pressure_psi": 45.0, "brake_pad_wear_mm": 3.0, "battery_voltage_v": 13.7}
    res_brk_a_w = predictor.predict_record(rec_brk_a_w)
    assert any("Worn Brake Pads Warning" in a for a in res_brk_a_w["safety_alerts"]), "Profile A brake pad warning failed!"
    
    # Profile D (VH-1006, warn=6.4mm): 5.5mm triggers warning
    rec_brk_d_w = {"vehicle_id": "VH-1006", "engine_temp_c": 88.0, "oil_pressure_psi": 45.0, "brake_pad_wear_mm": 5.5, "battery_voltage_v": 27.4}
    res_brk_d_w = predictor.predict_record(rec_brk_d_w)
    assert any("Worn Brake Pads Warning" in a for a in res_brk_d_w["safety_alerts"]), "Profile D brake pad warning failed!"

    # 4d. Low Battery Voltage Warning (batt_v <= warn_voltage)
    # 12V System (VH-1001, warn=11.8V): 11.5V triggers warning
    rec_bat_12_w = {"vehicle_id": "VH-1001", "engine_temp_c": 88.0, "oil_pressure_psi": 45.0, "brake_pad_wear_mm": 8.0, "battery_voltage_v": 11.5}
    res_bat_12_w = predictor.predict_record(rec_bat_12_w)
    assert any("Battery Voltage Warning" in a for a in res_bat_12_w["safety_alerts"]), "12V battery warning failed!"
    
    # 24V System (VH-1006, warn=23.6V): 23.0V triggers warning
    rec_bat_24_w = {"vehicle_id": "VH-1006", "engine_temp_c": 88.0, "oil_pressure_psi": 45.0, "brake_pad_wear_mm": 15.0, "battery_voltage_v": 23.0}
    res_bat_24_w = predictor.predict_record(rec_bat_24_w)
    assert any("Battery Voltage Warning" in a for a in res_bat_24_w["safety_alerts"]), "24V battery warning failed!"

    # 4e. Low Oil Pressure Warning (oil_psi <= 22.0)
    rec_oil_w = {"vehicle_id": "VH-1001", "engine_temp_c": 88.0, "oil_pressure_psi": 21.0, "brake_pad_wear_mm": 8.0, "battery_voltage_v": 13.7}
    res_oil_w = predictor.predict_record(rec_oil_w)
    assert any("Low Oil Pressure Warning" in a for a in res_oil_w["safety_alerts"]), "Oil pressure warning failed!"

    passed_tests += 1
    print(" [PASS] Test 4: Warning thresholds (engine, brake, 12V/24V battery, oil) verified with exact comparison logic.")

    # -------------------------------------------------------------------------
    # TEST SUITE 5: CRITICAL TELEMETRY READINGS & OVERRIDE LOGIC
    # -------------------------------------------------------------------------
    print("\n[TEST GROUP 5] Testing Critical Fault Threshold Telemetry & Overrides...")
    total_tests += 1
    
    # 5a. High Engine Temperature Critical (eng_temp >= 112.0)
    rec_eng_c = {"vehicle_id": "VH-1001", "engine_temp_c": 113.5, "oil_pressure_psi": 45.0, "brake_pad_wear_mm": 8.0, "battery_voltage_v": 13.7}
    res_eng_c = predictor.predict_record(rec_eng_c)
    assert res_eng_c["status"] == "CRITICAL"
    assert any("Engine Overheating" in a for a in res_eng_c["safety_alerts"]), "Engine overheating critical alert failed!"
    assert res_eng_c["subsystem_probabilities"]["engine"] >= 0.90

    # 5b. Mahindra XUV700 115.0C Critical Heuristic Check
    rec_xuv_c = {"vehicle_id": "VH-1002", "engine_temp_c": 115.8, "oil_pressure_psi": 45.0, "brake_pad_wear_mm": 8.0, "battery_voltage_v": 13.7}
    res_xuv_c = predictor.predict_record(rec_xuv_c)
    assert res_xuv_c["status"] == "CRITICAL"
    assert any("Engine Overheating" in a for a in res_xuv_c["safety_alerts"]), "XUV700 failed to alert overheating at 115.8°C!"

    # 5c. Low Brake Pad Critical (brake_pad <= crit_thickness)
    # Profile A (crit=2.2mm): 1.9mm triggers critical
    rec_brk_a_c = {"vehicle_id": "VH-1001", "engine_temp_c": 88.0, "oil_pressure_psi": 45.0, "brake_pad_wear_mm": 1.9, "battery_voltage_v": 13.7}
    res_brk_a_c = predictor.predict_record(rec_brk_a_c)
    assert res_brk_a_c["status"] == "CRITICAL"
    assert any("Worn Brake Pads Critical" in a for a in res_brk_a_c["safety_alerts"]), "Profile A brake pad critical failed!"
    
    # Profile D (crit=3.2mm): 2.8mm triggers critical
    rec_brk_d_c = {"vehicle_id": "VH-1006", "engine_temp_c": 88.0, "oil_pressure_psi": 45.0, "brake_pad_wear_mm": 2.8, "battery_voltage_v": 27.4}
    res_brk_d_c = predictor.predict_record(rec_brk_d_c)
    assert res_brk_d_c["status"] == "CRITICAL"
    assert any("Worn Brake Pads Critical" in a for a in res_brk_d_c["safety_alerts"]), "Profile D brake pad critical failed!"

    # 5d. Low Battery Voltage Critical (batt_v <= crit_voltage)
    # 12V System (crit=11.0V): 10.6V triggers critical
    rec_bat_12_c = {"vehicle_id": "VH-1001", "engine_temp_c": 88.0, "oil_pressure_psi": 45.0, "brake_pad_wear_mm": 8.0, "battery_voltage_v": 10.6}
    res_bat_12_c = predictor.predict_record(rec_bat_12_c)
    assert res_bat_12_c["status"] == "CRITICAL"
    assert any("Critically Low Battery Voltage" in a for a in res_bat_12_c["safety_alerts"]), "12V critical voltage failed!"
    
    # 24V System (crit=22.0V): 21.2V triggers critical
    rec_bat_24_c = {"vehicle_id": "VH-1006", "engine_temp_c": 88.0, "oil_pressure_psi": 45.0, "brake_pad_wear_mm": 15.0, "battery_voltage_v": 21.2}
    res_bat_24_c = predictor.predict_record(rec_bat_24_c)
    assert res_bat_24_c["status"] == "CRITICAL"
    assert any("Critically Low Battery Voltage" in a for a in res_bat_24_c["safety_alerts"]), "24V critical voltage failed!"

    # 5e. Low Oil Pressure Critical (oil_psi <= 15.0)
    rec_oil_c = {"vehicle_id": "VH-1001", "engine_temp_c": 88.0, "oil_pressure_psi": 12.0, "brake_pad_wear_mm": 8.0, "battery_voltage_v": 13.7}
    res_oil_c = predictor.predict_record(rec_oil_c)
    assert res_oil_c["status"] == "CRITICAL"
    assert any("Low Oil Pressure Critical" in a for a in res_oil_c["safety_alerts"]), "Critical oil pressure alert failed!"

    passed_tests += 1
    print(" [PASS] Test 5: Critical alert thresholds and safety net overrides verified.")

    # -------------------------------------------------------------------------
    # TEST SUITE 6: MISSING & STALE READINGS AUDIT
    # -------------------------------------------------------------------------
    print("\n[TEST GROUP 6] Testing Missing Sensor Imputation & Stale Readings...")
    total_tests += 1
    
    buffer = LiveCausalVehicleBuffer(window=3, max_age_seconds=600)
    buffer.clear()
    
    # 6a. Missing Mandatory Sensor on First Ping (unresolved -> UNAVAILABLE)
    bad_ping = {"vehicle_id": "VH-MISSING", "oil_pressure_psi": 45.0, "brake_pad_wear_mm": 8.0, "battery_voltage_v": 13.7} # missing engine_temp_c
    cleaned_bad, dq_bad = buffer.process_telemetry(bad_ping)
    assert dq_bad["status"] == "UNAVAILABLE", f"Expected UNAVAILABLE, got {dq_bad['status']}"
    assert "engine_temp_c" in dq_bad["warning_msg"], "Missing engine_temp_c not reported!"
    
    # 6b. Causal Rolling Imputation from History
    buffer.clear("VH-CAUSAL")
    now_dt = datetime.datetime.now(datetime.timezone.utc)
    t0 = (now_dt - datetime.timedelta(seconds=20)).isoformat()
    t1 = (now_dt - datetime.timedelta(seconds=10)).isoformat()
    t2 = now_dt.isoformat()
    
    ping1 = {"vehicle_id": "VH-CAUSAL", "timestamp": t0, "engine_temp_c": 86.0, "oil_pressure_psi": 45.0, "brake_pad_wear_mm": 8.0, "battery_voltage_v": 13.7}
    ping2 = {"vehicle_id": "VH-CAUSAL", "timestamp": t1, "engine_temp_c": 88.0, "oil_pressure_psi": 45.0, "brake_pad_wear_mm": 8.0, "battery_voltage_v": 13.7}
    buffer.process_telemetry(ping1)
    buffer.process_telemetry(ping2)
    
    # Ping 3 has missing engine_temp_c: should be causally imputed from history median (87.0C)
    ping3 = {"vehicle_id": "VH-CAUSAL", "timestamp": t2, "engine_temp_c": None, "oil_pressure_psi": 45.0, "brake_pad_wear_mm": 8.0, "battery_voltage_v": 13.7}
    cleaned3, dq3 = buffer.process_telemetry(ping3)
    assert cleaned3["engine_temp_c"] == 87.0, f"Expected causal median 87.0, got {cleaned3['engine_temp_c']}"
    assert "engine_temp_c" in dq3["imputed_sensors"], "engine_temp_c missing from imputed list"
    assert dq3["status"] == "DEGRADED"

    # 6c. Stale Sensor Reading (> 600 seconds old)
    stale_ts = (now_dt - datetime.timedelta(seconds=900)).isoformat()
    ping_stale = {"vehicle_id": "VH-CAUSAL", "timestamp": stale_ts, "engine_temp_c": 88.0, "oil_pressure_psi": 45.0, "brake_pad_wear_mm": 8.0, "battery_voltage_v": 13.7}
    cleaned_stale, dq_stale = buffer.process_telemetry(ping_stale)
    assert dq_stale["is_stale"] is True, "Stale reading not flagged!"
    assert dq_stale["status"] == "STALE", f"Expected STALE status, got {dq_stale['status']}"

    passed_tests += 1
    print(" [PASS] Test 6: Missing mandatory detection, causal rolling imputation, and staleness detection verified.")

    # -------------------------------------------------------------------------
    # TEST SUITE 7: UNKNOWN & UNSUPPORTED VEHICLE CONFIGURATIONS
    # -------------------------------------------------------------------------
    print("\n[TEST GROUP 7] Testing Unknown Vehicle Configuration Handling...")
    total_tests += 1
    
    # 7a. Unsupported Electrical System (48V)
    rec_unk_elec = {
        "vehicle_id": "VH-UNK-48V",
        "electrical_system": "48V",
        "engine_temp_c": 90.0,
        "oil_pressure_psi": 45.0,
        "brake_pad_wear_mm": 8.0,
        "battery_voltage_v": 11.2  # 11.2V on 12V would trigger warning, but 48V is unsupported
    }
    res_unk_elec = predictor.predict_record(rec_unk_elec)
    assert res_unk_elec["configuration_status"] == "UNKNOWN_CONFIGURATION", "Failed to set UNKNOWN_CONFIGURATION status!"
    assert any("Unknown electrical system configuration" in a for a in res_unk_elec["safety_alerts"]), "Missing unknown electrical alert!"
    # Crucial requirement: 12V/24V thresholds suspended, do not trigger false voltage alarm
    assert not any("Battery Voltage Warning" in a for a in res_unk_elec["safety_alerts"]), "12V threshold falsely applied to 48V system!"

    # 7b. Unsupported Brake Configuration ("Ceramic Regenerative")
    rec_unk_brake = {
        "vehicle_id": "VH-UNK-BRK",
        "brake_configuration": "Ceramic Regenerative",
        "engine_temp_c": 90.0,
        "oil_pressure_psi": 45.0,
        "brake_pad_wear_mm": 1.2,  # 1.2mm on standard brakes is critical, but unknown config suspended
        "battery_voltage_v": 13.8
    }
    res_unk_brake = predictor.predict_record(rec_unk_brake)
    assert res_unk_brake["configuration_status"] == "UNKNOWN_CONFIGURATION", "Failed to set UNKNOWN_CONFIGURATION status!"
    assert any("Unknown brake configuration" in a for a in res_unk_brake["safety_alerts"]), "Missing unknown brake alert!"
    # Crucial requirement: Brake wear thresholds suspended
    assert not any("Worn Brake Pads" in a for a in res_unk_brake["safety_alerts"]), "Brake wear threshold falsely applied to unknown brake config!"

    # 7c. Unregistered Vehicle ID without metadata
    rec_unregistered = {
        "vehicle_id": "VH-UNKNOWN-EXTERNAL",
        "engine_temp_c": 90.0,
        "oil_pressure_psi": 45.0,
        "brake_pad_wear_mm": 8.0,
        "battery_voltage_v": 13.8
    }
    res_unregistered = predictor.predict_record(rec_unregistered)
    assert res_unregistered["configuration_status"] == "UNKNOWN_CONFIGURATION"
    assert any("Missing electrical system configuration" in a or "Unknown electrical system" in a for a in res_unregistered["safety_alerts"])

    passed_tests += 1
    print(" [PASS] Test 7: Unknown electrical and brake configurations properly flagged and thresholds suspended.")

    # -------------------------------------------------------------------------
    # SUMMARY
    # -------------------------------------------------------------------------
    print("\n" + "=" * 85)
    print(f" ALL {passed_tests}/{total_tests} CONFIGURATION & TELEMETRY TEST GROUPS PASSED WITH 100% SUCCESS! ")
    print("=" * 85)
    return True


if __name__ == "__main__":
    success = run_vehicle_configuration_tests()
    if not success:
        sys.exit(1)

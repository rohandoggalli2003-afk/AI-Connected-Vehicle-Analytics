"""
AI-Powered Connected Vehicle Analytics Platform (BDA)
Phase C: AI-Based Maintenance Report Generator

Transforms real-time model failure probabilities and sensor telemetry
into structured, human-readable maintenance diagnostic work orders.

Pipeline Output Columns:
- failure_probability
- predicted_failure_type
- maintenance_report
"""

import os
import json
import joblib
import sys
import pandas as pd
from typing import Dict, List, Any

# Ensure project root is in sys.path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# Root cause diagnostic recommendations
DIAGNOSTIC_ACTIONS = {
    "Engine_Overheat": {
        "subsystem": "Powertrain / Cooling",
        "severity": "CRITICAL",
        "action": "Inspect radiator coolant flow, check water pump impeller, verify thermostat valve opening, and pressure-test cylinder head gasket.",
        "parts": ["Engine Coolant 50/50", "Thermostat Assembly", "Water Pump Gasket"],
        "estimated_downtime_hours": 3.5
    },
    "Low_Oil_Pressure": {
        "subsystem": "Powertrain / Lubrication",
        "severity": "CRITICAL",
        "action": "Immediate engine shutdown required. Inspect oil pan level, check oil pump pickup screen for sludge, verify oil pressure sensor harness, inspect main bearings.",
        "parts": ["Synthetic Engine Oil 5W-30", "OEM Oil Filter", "Oil Pressure Sensor"],
        "estimated_downtime_hours": 2.5
    },
    "Brake_Overheat": {
        "subsystem": "Chassis / Braking & Friction",
        "severity": "CRITICAL",
        "action": "Severe brake rotor thermal surge detected (>240 deg C). Inspect caliper sliding pins for seizure, check brake hose restriction, replace glazed friction pads and warped brake discs.",
        "parts": ["Vented Brake Rotors (Pair)", "Heavy-Duty Brake Caliper Assembly", "High-Temp Ceramic Brake Pads", "DOT-4 Synthetic Brake Fluid"],
        "estimated_downtime_hours": 3.0
    },
    "Brake_Pad_Worn": {
        "subsystem": "Chassis / Braking",
        "severity": "HIGH",
        "action": "Brake pad thickness below 3.0mm critical threshold. Replace front and rear friction pads, machine or replace brake rotors, bleed hydraulic fluid lines.",
        "parts": ["Ceramic Brake Pad Set", "Vented Brake Rotors", "DOT-4 Brake Fluid"],
        "estimated_downtime_hours": 2.0
    },
    "ABS_Malfunction": {
        "subsystem": "Chassis / Safety Electronics",
        "severity": "HIGH",
        "action": "ABS reluctor ring or wheel speed sensor discrepancy detected. Check sensor tone rings for debris, verify wheel speed sensor wire harness, run ECU ABS diagnostic scan.",
        "parts": ["ABS Wheel Speed Sensor", "Sensor Wiring Harness"],
        "estimated_downtime_hours": 1.5
    },
    "Battery_Degraded": {
        "subsystem": "Electrical / Energy Storage",
        "severity": "MEDIUM",
        "action": "Battery State-of-Health (SOH) dropped below operating safety margin. Perform 12V conductance battery test, clean terminal corrosion, replace 12V AGM auxiliary/starter battery.",
        "parts": ["12V 80Ah AGM Battery", "Terminal Anti-Corrosion Spray"],
        "estimated_downtime_hours": 0.8
    },
    "Alternator_Failure": {
        "subsystem": "Electrical / Charging System",
        "severity": "HIGH",
        "action": "Alternator voltage regulation failure detected (<12.8V charging output). Inspect serpentine belt tensioner, test alternator diode bridge rectifier, replace alternator.",
        "parts": ["High-Output Alternator 160A", "Serpentine Drive Belt"],
        "estimated_downtime_hours": 2.0
    },
    "Healthy": {
        "subsystem": "General Inspection",
        "severity": "LOW",
        "action": "All subsystems operating within nominal OEM tolerances. Continue routine scheduled maintenance interval.",
        "parts": ["None"],
        "estimated_downtime_hours": 0.0
    }
}

from ml_pipeline.predictor import get_predictor, UnifiedVehiclePredictor

class MaintenanceReportGenerator:
    """
    Phase C: Rule-Based Expert AI Diagnostic Engine & Predictive Work Order Generator.
    
    Combines calibrated ML inference with physical sensor safety checks (diagnostic catalog,
    OEM parts requisition, and service downtime estimation) to generate deterministic,
    mechanic-ready fleet work orders.
    """
    def __init__(self, models_dir: str = "models"):
        self.models_dir = models_dir
        self.predictor = get_predictor()

    def predict_vehicle_risk(self, telemetry_row: Dict[str, Any]) -> Dict[str, Any]:
        """Calculates unified failure probabilities with safety sensor overrides."""
        if "failure_probability" in telemetry_row and "predicted_failure_type" in telemetry_row and "subsystem_probabilities" in telemetry_row:
            probs = telemetry_row["subsystem_probabilities"]
            fail_prob = telemetry_row["failure_probability"]
            pred_type = telemetry_row["predicted_failure_type"]
            status = telemetry_row.get("status", "HEALTHY")
            valid_p = {k: v for k, v in probs.items() if v is not None} if isinstance(probs, dict) else {}
            max_comp = max(valid_p, key=valid_p.get) if valid_p else "NONE"
            return {
                "failure_probability": fail_prob,
                "status": status,
                "predicted_failure_type": pred_type,
                "subsystem_probabilities": probs if isinstance(probs, dict) else {},
                "primary_at_risk_subsystem": max_comp.upper() if (fail_prob is not None and fail_prob >= 0.30) else "NONE",
                "safety_alerts": telemetry_row.get("safety_alerts", []),
                "ml_status": telemetry_row.get("ml_status", "AVAILABLE")
            }

        res = self.predictor.predict_record(telemetry_row)
        probs = res["subsystem_probabilities"]
        fail_prob = res["failure_probability"]
        pred_type = res["predicted_failure_type"]
        status = res["status"]
        valid_p = {k: v for k, v in probs.items() if v is not None} if isinstance(probs, dict) else {}
        max_comp = max(valid_p, key=valid_p.get) if valid_p else "NONE"
        return {
            "failure_probability": fail_prob,
            "status": status,
            "predicted_failure_type": pred_type,
            "subsystem_probabilities": probs,
            "primary_at_risk_subsystem": max_comp.upper() if (fail_prob is not None and fail_prob >= 0.30) else "NONE",
            "safety_alerts": res.get("safety_alerts", []),
            "ml_status": res.get("ml_status", "AVAILABLE")
        }

    def generate_report(self, vehicle_id: str, telemetry: Dict[str, Any], metadata: Dict[str, Any] = None) -> str:
        """Generates comprehensive formatted AI maintenance work order."""
        risk_info = self.predict_vehicle_risk(telemetry)
        fail_type = risk_info["predicted_failure_type"]
        fail_prob = risk_info["failure_probability"]
        status = risk_info.get("status", "HEALTHY")
        diag = DIAGNOSTIC_ACTIONS.get(fail_type, DIAGNOSTIC_ACTIONS["Healthy"])
        
        meta_dict = metadata.to_dict() if hasattr(metadata, "to_dict") else (metadata or {})
        brand = meta_dict.get("brand", "Commercial")
        model = meta_dict.get("model", "Vehicle")
        year  = meta_dict.get("vehicle_year", 2022)
        
        if fail_prob is not None:
            risk_str = "<1%" if fail_prob < 0.01 else f"{fail_prob * 100:.1f}%"
            risk_ml_str = "<1%" if fail_prob < 0.01 else f"{fail_prob * 100:.2f}%"
        else:
            risk_str = "UNAVAILABLE"
            risk_ml_str = "UNAVAILABLE (Model Error / Degraded)"

        probs = risk_info.get("subsystem_probabilities", {})
        p_eng = probs.get("engine")
        p_brk = probs.get("brake")
        p_bat = probs.get("battery")
        e_str = f"{p_eng * 100:.1f}%" if p_eng is not None else "N/A"
        b_str = f"{p_brk * 100:.1f}%" if p_brk is not None else "N/A"
        bat_str = f"{p_bat * 100:.1f}%" if p_bat is not None else "N/A"

        # Build human-readable report narrative
        report_lines = [
            f"================================================================================",
            f" AUTOMATED AI FLEET MAINTENANCE DIAGNOSTIC WORK ORDER ",
            f"================================================================================",
            f"VEHICLE IDENTIFIER      : {vehicle_id} ({year} {brand} {model})",
            f"TIMESTAMP GENERATED     : {telemetry.get('timestamp', 'LIVE')}",
            f"HEALTH RISK STATUS      : {status} (Risk Index: {risk_str})",
            f"PRIMARY FAULT DETECTED  : {fail_type.replace('_', ' ').upper()}",
            f"SUBSYSTEM CLASSIFICATION: {diag['subsystem']}",
            f"--------------------------------------------------------------------------------",
            f"TELEMETRIC SENSOR AUDIT:",
            f"  - Odometer Reading    : {telemetry.get('odometer_km', 'N/A')} km (Engine Hours: {telemetry.get('engine_hours', 'N/A')} h)",
            f"  - Vehicle Speed       : {telemetry.get('vehicle_speed_kph', 'N/A')} km/h | Engine RPM: {telemetry.get('engine_rpm', 'N/A')}",
            f"  - Engine Temperature  : {telemetry.get('engine_temp_c', 'N/A')} deg C (Coolant: {telemetry.get('coolant_temp_c', 'N/A')} deg C)",
            f"  - Oil Pressure        : {telemetry.get('oil_pressure_psi', 'N/A')} PSI | Vibration: {telemetry.get('engine_vibration', 'N/A')}",
            f"  - Brake Pad Thickness : {telemetry.get('brake_pad_wear_mm', 'N/A')} mm | Brake Temp: {telemetry.get('brake_temp_c', 'N/A')} deg C",
            f"  - Battery Voltage     : {telemetry.get('battery_voltage_v', 'N/A')} V | Alternator: {telemetry.get('alternator_output_v', 'N/A')} V",
            f"  - Battery SOH         : {telemetry.get('battery_health_percent', 'N/A')}%",
            f"  - GPS Location        : Lat {telemetry.get('gps_latitude', 'N/A')}, Lon {telemetry.get('gps_longitude', 'N/A')}",
            f"--------------------------------------------------------------------------------",
            f"PREDICTIVE ML INFERENCE & SAFETY NET OVERRIDE:",
            f"  - Failure Probability : {risk_ml_str}",
            f"  - Subsystem Breakdown : Engine={e_str}, Brake={b_str}, Battery={bat_str}",
        ]
        
        if risk_info.get("safety_alerts"):
            report_lines.append(f"  - Physical Alerts     : {', '.join(risk_info['safety_alerts'])}")
            
        report_lines.extend([
            f"--------------------------------------------------------------------------------",
            f"RECOMMENDED WORKSHOP ACTION:",
            f"  {diag['action']}",
            f"",
            f"REQUIRED REPLACEMENT PARTS:",
            f"  " + ", ".join(diag["parts"]),
            f"",
            f"ESTIMATED SERVICE DOWNTIME : {diag['estimated_downtime_hours']} Hours",
            f"================================================================================"
        ])
        
        return "\n".join(report_lines)

    def rewrite_work_order_with_llm(self, structured_report: str, llm_client=None) -> str:
        """
        Optional extension (interface only):
        Demonstrates contract for rewriting structured work orders into conversational garage dispatch memos.
        When no external LLM client is configured, returns the deterministic expert report.
        """
        if llm_client is None:
            return structured_report
        try:
            prompt = (
                "You are an expert commercial fleet master mechanic. "
                "Rewrite the following telematics diagnostic work order into a concise, professional briefing "
                "for the shop foreman. Retain all OEM part names, torque specs, and downtime estimates:\n\n"
                f"{structured_report}"
            )
            return llm_client.generate(prompt)
        except Exception:
            return structured_report

def run_report_generation_demo(output_file: str = "reports/sample_maintenance_reports.txt"):
    """Runs demonstration generating reports across a diverse spectrum of vehicle health states."""
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    generator = MaintenanceReportGenerator()
    
    # Load all available landing records
    all_records = []
    landing_dir = "data/landing"
    if os.path.exists(landing_dir):
        files = sorted([f for f in os.listdir(landing_dir) if f.endswith(".json")], reverse=True)
        for f in files[:5]:
            with open(os.path.join(landing_dir, f), "r", encoding="utf-8") as fp:
                for line in fp:
                    if line.strip():
                        all_records.append(json.loads(line))
                        
    # Load master metadata
    master_dict = {}
    if os.path.exists("data/vehicle_master.csv"):
        df_m = pd.read_csv("data/vehicle_master.csv")
        master_dict = {row["vehicle_id"]: row.to_dict() for _, row in df_m.iterrows()}
        
    # Categorize records by evaluated risk to ensure diverse sample coverage
    healthy_recs = []
    engine_recs = []
    brake_recs = []
    battery_recs = []
    warning_recs = []
    
    for rec in all_records:
        r = generator.predict_vehicle_risk(rec)
        p = r["failure_probability"]
        pt = r["predicted_failure_type"]
        if p >= 0.60:
            if ("Engine" in pt or "Oil" in pt) and not engine_recs:
                engine_recs.append(rec)
            elif ("Brake" in pt or "ABS" in pt) and not brake_recs:
                brake_recs.append(rec)
            elif ("Battery" in pt or "Alternator" in pt) and not battery_recs:
                battery_recs.append(rec)
        elif 0.25 <= p < 0.60 and not warning_recs:
            warning_recs.append(rec)
        elif p < 0.20 and not healthy_recs:
            healthy_recs.append(rec)
            
        if healthy_recs and engine_recs and brake_recs and battery_recs and warning_recs:
            break

    # Pick 5 representative vehicles covering healthy, warning, and critical subsystems
    selected = []
    if healthy_recs:
        selected.append(healthy_recs[0])
    if engine_recs:
        selected.append(engine_recs[0])
    if brake_recs:
        selected.append(brake_recs[0])
    if battery_recs:
        selected.append(battery_recs[0])
    if warning_recs:
        selected.append(warning_recs[0])
        
    # Fallback to general slice if any category was missing in landing files
    if len(selected) < 5 and all_records:
        for r in all_records:
            if r not in selected:
                selected.append(r)
            if len(selected) == 5:
                break

    print(f"[*] Generating diverse maintenance diagnostic reports for {len(selected)} fleet vehicles...")
    reports = []
    for rec in selected:
        vid = rec["vehicle_id"]
        meta = master_dict.get(vid, {})
        rep = generator.generate_report(vid, rec, meta)
        reports.append(rep)
        
    full_output = "\n\n".join(reports)
    with open(output_file, "w", encoding="utf-8") as f:
        f.write(full_output)
        
    print(f"[+] Sample AI maintenance reports generated and saved to: {output_file}")
    print("\n--- SAMPLE REPORT PREVIEWS ---")
    for r in reports[:2]:
        print(r[:400] + "\n...\n")

if __name__ == "__main__":
    run_report_generation_demo()

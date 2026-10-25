"""
AI-Powered Connected Vehicle Analytics Platform (BDA)
Unified Prediction, History Rolling Engine & Safety Override Service

Features:
1. Rolling Window Live Cache: Maintains per-vehicle 5-ping rolling statistics identically to training.
2. Missing-Data Policy: Validates mandatory physical sensors and uses documented SAE baselines for secondary features.
3. Centralized Safety Rules: Reads physical limits from config/safety_thresholds.json with OEM justifications.
4. Risk Quantification: Returns calibrated subsystem probabilities, joint failure probability, and overall_risk_score.
5. Error-Resilient Diagnostics: Exposes prediction validity and never defaults to silent healthy status on failures.
"""

import os
import json
import logging
from collections import defaultdict, deque
from typing import Dict, Any, Tuple, Optional, List

import joblib
import numpy as np
import pandas as pd

# Configure logger
logger = logging.getLogger("ml_pipeline.predictor")

# Mandatory sensors required to make a valid prediction
CORE_MANDATORY_FIELDS = [
    "engine_temp_c",
    "oil_pressure_psi",
    "brake_pad_wear_mm",
    "battery_voltage_v"
]

# Documented engineering nominal baselines for secondary features
SAFE_FEATURE_DEFAULTS = {
    "odometer_km": 50000.0,
    "engine_hours": 1500.0,
    "vehicle_speed_kph": 0.0,
    "engine_temp_c": 90.0,
    "engine_rpm": 800.0,
    "oil_pressure_psi": 45.0,
    "coolant_temp_c": 86.0,
    "fuel_level_percent": 60.0,
    "fuel_consumption_lph": 3.0,
    "engine_load_percent": 15.0,
    "throttle_position_percent": 10.0,
    "air_flow_rate_gps": 15.0,
    "exhaust_gas_temp_c": 450.0,
    "engine_vibration": 1.0,
    "brake_pad_wear_mm": 8.0,
    "brake_fluid_level": 90.0,
    "brake_temp_c": 75.0,
    "abs_fault_indicator": 0,
    "brake_pedal_position": 0.0,
    "wheel_speed_fl_kph": 0.0,
    "wheel_speed_fr_kph": 0.0,
    "wheel_speed_rl_kph": 0.0,
    "wheel_speed_rr_kph": 0.0,
    "battery_voltage_v": 13.8,
    "battery_current_a": 0.0,
    "battery_temp_c": 30.0,
    "alternator_output_v": 14.1,
    "battery_charge_percent": 90.0,
    "battery_health_percent": 95.0,
    "ambient_temp_c": 22.0,
    "gps_latitude": 37.77,
    "gps_longitude": -122.42,
    "rolling_engine_temp_mean": 90.0,
    "rolling_engine_temp_std": 0.0,
    "rolling_oil_pressure_min": 45.0,
    "rolling_brake_temp_max": 75.0,
    "rolling_battery_v_min": 13.8,
    "wheel_speed_delta": 0.0,
}

# Centralized fallback safety thresholds (loaded dynamically from config if present)
DEFAULT_SAFETY_THRESHOLDS = {
    "engine": {
        "overheat_warning_temp_c": 105.0,
        "overheat_critical_temp_c": 112.0,
        "coolant_overheat_temp_c": 108.0,
        "low_oil_warning_psi": 22.0,
        "low_oil_critical_psi": 15.0,
        "excessive_vibration_mms": 3.2
    },
    "brake": {
        "pad_wear_warning_mm": 3.2,
        "pad_wear_critical_mm": 2.2,
        "thermal_spike_c": 260.0,
        "abs_fault_indicator": 1
    },
    "battery": {
        "voltage_warning_v": 11.8,
        "voltage_critical_v": 11.0,
        "soh_warning_pct": 50.0,
        "soh_critical_pct": 40.0,
        "alternator_undervoltage_v": 12.8
    }
}


try:
    from ml_pipeline.data_cleaner import LiveCausalVehicleBuffer, CORE_MANDATORY_FIELDS, SENSOR_PHYSICAL_BOUNDS
except ImportError:
    from data_cleaner import LiveCausalVehicleBuffer, CORE_MANDATORY_FIELDS, SENSOR_PHYSICAL_BOUNDS


class UnifiedVehiclePredictor:
    """
    Singleton-style unified predictor executing calibrated ML failure prediction
    with per-vehicle live causal rolling window cache, data quality validation, and safety net overrides.
    """
    def __init__(self, models_dir: str = "models", config_dir: str = "config"):
        base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
        if not os.path.isabs(models_dir):
            if not os.path.exists(models_dir) and os.path.exists(os.path.join(base_dir, models_dir)):
                models_dir = os.path.join(base_dir, models_dir)
        if not os.path.isabs(config_dir):
            if not os.path.exists(config_dir) and os.path.exists(os.path.join(base_dir, config_dir)):
                config_dir = os.path.join(base_dir, config_dir)
        self.models_dir = models_dir
        self.config_dir = config_dir
        self.scaler = None
        self.feature_names = None
        self.clipping_bounds = None
        self.pca = None
        self.train_medians = {}
        self.champions_summary = {}
        self.models = {}
        self.safety_thresholds = DEFAULT_SAFETY_THRESHOLDS
        
        # Per-vehicle 5-ping rolling observation buffer for live feature rolling
        self.vehicle_history: Dict[str, deque] = defaultdict(lambda: deque(maxlen=5))
        
        # Causal vehicle-wise missing value and outlier auditor buffer
        self.causal_buffer = LiveCausalVehicleBuffer(window=3, max_age_seconds=600, train_medians=self.train_medians)
        
        self._load_artifacts()
        self._load_configs()

    def _load_artifacts(self):
        scaler_path = os.path.join(self.models_dir, "scaler.joblib")
        feats_path  = os.path.join(self.models_dir, "feature_names.joblib")
        bounds_path = os.path.join(self.models_dir, "clipping_bounds.joblib")
        pca_path    = os.path.join(self.models_dir, "pca.joblib")
        champ_path  = os.path.join(self.models_dir, "champions_summary.json")
        medians_path = os.path.join(self.models_dir, "train_medians.joblib")

        if os.path.exists(medians_path):
            try:
                self.train_medians = joblib.load(medians_path)
                self.causal_buffer.set_train_medians(self.train_medians)
            except Exception as e:
                logger.warning(f"Could not load train_medians.joblib: {e}")

        if os.path.exists(scaler_path) and os.path.exists(feats_path):
            self.scaler = joblib.load(scaler_path)
            self.feature_names = joblib.load(feats_path)
            
            if os.path.exists(bounds_path):
                self.clipping_bounds = joblib.load(bounds_path)
            if os.path.exists(pca_path):
                self.pca = joblib.load(pca_path)
            if os.path.exists(champ_path):
                try:
                    with open(champ_path, "r") as f:
                        self.champions_summary = json.load(f)
                except Exception as e:
                    logger.warning(f"Could not load champions_summary.json: {e}")

            for comp in ["engine", "brake", "battery"]:
                mpath = os.path.join(self.models_dir, f"champion_{comp}_model.joblib")
                if os.path.exists(mpath):
                    try:
                        self.models[comp] = joblib.load(mpath)
                    except Exception as e:
                        logger.error(f"Failed to load champion model from {mpath}: {e}")
        else:
            self.models = {}

    def _load_configs(self):
        self.vehicle_thresholds = {}
        self.supported_electrical_systems = {"12V", "24V"}
        self.supported_brake_configurations = {
            "Hydraulic Disc",
            "Hydraulic Disc/Drum",
            "Pneumatic/Hydraulic Disc",
            "Pneumatic S-Cam/Disc"
        }
        self.profiles = {}
        self.vehicle_master = {}

        # 1. Load vehicle_thresholds.json
        v_thresh_path = os.path.join(self.config_dir, "vehicle_thresholds.json")
        if os.path.exists(v_thresh_path):
            try:
                with open(v_thresh_path, "r", encoding="utf-8") as f:
                    v_data = json.load(f)
                    if "supported_electrical_systems" in v_data:
                        self.supported_electrical_systems = set(v_data["supported_electrical_systems"])
                    if "supported_brake_configurations" in v_data:
                        self.supported_brake_configurations = set(v_data["supported_brake_configurations"])
                    self.profiles = v_data.get("profiles", {})
                    for item in v_data.get("vehicles", []):
                        vid = item.get("vehicle_id")
                        if vid:
                            self.vehicle_thresholds[vid] = item
            except Exception as e:
                logger.warning(f"Could not load vehicle_thresholds.json: {e}")

        # 2. Load safety_thresholds.json
        cfg_path = os.path.join(self.config_dir, "safety_thresholds.json")
        if os.path.exists(cfg_path):
            try:
                with open(cfg_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    if "subsystems" in data:
                        self.safety_thresholds = data["subsystems"]
            except Exception as e:
                logger.warning(f"Could not load safety_thresholds.json: {e}")

        # 3. Load vehicle_master.csv cache if present
        v_master_path = os.path.join("data", "vehicle_master.csv")
        if os.path.exists(v_master_path):
            try:
                df_m = pd.read_csv(v_master_path)
                for _, r in df_m.iterrows():
                    self.vehicle_master[str(r.get("vehicle_id"))] = r.to_dict()
            except Exception as e:
                logger.warning(f"Could not load vehicle_master.csv: {e}")

    def reset_history(self, vehicle_id: Optional[str] = None):
        """Clears rolling sensor history and causal cleaner buffer for one or all vehicles."""
        if vehicle_id:
            if vehicle_id in self.vehicle_history:
                self.vehicle_history[vehicle_id].clear()
            self.causal_buffer.clear(vehicle_id)
        else:
            self.vehicle_history.clear()
            self.causal_buffer.clear()

    def predict_record(self, row: Dict[str, Any]) -> Dict[str, Any]:
        """
        Runs causal cleaning, ML failure prediction with live rolling stats calculation,
        data quality auditing, and safety net overrides.
        """
        # 1. Causal Vehicle-Wise Data Cleaning & Physical Sanity Audit
        cleaned_row, dq_report = self.causal_buffer.process_telemetry(row)

        # Check if mandatory sensors could not be resolved from history or baseline
        if dq_report["status"] == "UNAVAILABLE":
            err_msg = dq_report.get("warning_msg", "Critical mandatory fields missing without suitable history")
            logger.error(err_msg)
            return {
                "overall_risk_score": None,
                "failure_probability": None,
                "joint_failure_probability": None,
                "status": "UNAVAILABLE",
                "predicted_failure_type": "UNKNOWN",
                "ml_status": "INVALID_INPUT",
                "ml_errors": {"input_validation": err_msg},
                "prediction_valid": False,
                "subsystem_probabilities": {"engine": None, "brake": None, "battery": None},
                "subsystem_status": {"engine": "INVALID_INPUT", "brake": "INVALID_INPUT", "battery": "INVALID_INPUT"},
                "safety_alerts": [f"Input Error: {err_msg}"],
                "data_quality_report": dq_report,
                "data_quality_status": "UNAVAILABLE",
                "imputed_sensors": dq_report.get("imputed_sensors", []),
                "imputed_count": dq_report.get("imputed_count", 0),
                "sensor_errors": dq_report.get("sensor_errors", []),
                "data_quality_warning": dq_report.get("warning_msg", "")
            }

        # 2. Extract Physical Sensor Readings Safely (from causally cleaned row)
        vid = str(cleaned_row.get("vehicle_id", "default_vh"))
        eng_temp = float(cleaned_row.get("engine_temp_c", 90.0))
        cool_temp = float(cleaned_row.get("coolant_temp_c", 87.0))
        oil_psi = float(cleaned_row.get("oil_pressure_psi", 45.0))
        eng_vib = float(cleaned_row.get("engine_vibration", 1.0))
        brake_pad = float(cleaned_row.get("brake_pad_wear_mm", 8.0))
        brake_temp = float(cleaned_row.get("brake_temp_c", 95.0))
        abs_fault = int(cleaned_row.get("abs_fault_indicator", 0))
        batt_v = float(cleaned_row.get("battery_voltage_v", 13.8))
        batt_soh = float(cleaned_row.get("battery_health_percent", 95.0))
        alt_v = float(cleaned_row.get("alternator_output_v", 14.1))
        wheel_fl = float(cleaned_row.get("wheel_speed_fl_kph", 0.0))
        wheel_fr = float(cleaned_row.get("wheel_speed_fr_kph", 0.0))

        # 3. Maintain Per-Vehicle Rolling History & Calculate Window Features
        hist_entry = {
            "engine_temp_c": eng_temp,
            "oil_pressure_psi": oil_psi,
            "brake_temp_c": brake_temp,
            "battery_voltage_v": batt_v,
            "wheel_speed_fl_kph": wheel_fl,
            "wheel_speed_fr_kph": wheel_fr
        }
        self.vehicle_history[vid].append(hist_entry)
        hist_window = list(self.vehicle_history[vid])

        row_copy = dict(row)
        # Handle column aliases
        if "vehicle_speed_kph" not in row_copy and "speed_kph" in row_copy:
            row_copy["vehicle_speed_kph"] = row_copy["speed_kph"]
        if "engine_rpm" not in row_copy and "rpm" in row_copy:
            row_copy["engine_rpm"] = row_copy["rpm"]

        # Calculate live rolling statistics over the cached vehicle history window
        # (calculated identically to training rolling(5, min_periods=1))
        if "rolling_engine_temp_mean" not in row_copy:
            eng_temps = [h["engine_temp_c"] for h in hist_window]
            oil_psis  = [h["oil_pressure_psi"] for h in hist_window]
            brk_temps = [h["brake_temp_c"] for h in hist_window]
            batt_vs   = [h["battery_voltage_v"] for h in hist_window]

            row_copy["rolling_engine_temp_mean"] = float(np.mean(eng_temps))
            row_copy["rolling_engine_temp_std"]  = float(np.std(eng_temps, ddof=1)) if len(eng_temps) > 1 else 0.0
            row_copy["rolling_oil_pressure_min"] = float(np.min(oil_psis))
            row_copy["rolling_brake_temp_max"]   = float(np.max(brk_temps))
            row_copy["rolling_battery_v_min"]    = float(np.min(batt_vs))
            row_copy["wheel_speed_delta"]        = abs(wheel_fl - wheel_fr)

        # 4. Fill Non-Critical Secondary Features from Documented Baselines
        imputed_secondary = []
        feat_df = pd.DataFrame([row_copy])
        if self.feature_names is not None:
            for col in self.feature_names:
                if col not in feat_df.columns or pd.isna(feat_df[col].iloc[0]):
                    feat_df[col] = SAFE_FEATURE_DEFAULTS.get(col, 0.0)
                    imputed_secondary.append(col)

        # 5. ML Model Inference
        ml_status = "AVAILABLE"
        ml_errors = {}
        subsystem_status = {}
        subsystem_probabilities = {
            "engine": None,
            "brake": None,
            "battery": None
        }

        if self.scaler is not None and self.feature_names is not None and len(self.models) > 0:
            try:
                X_raw = feat_df[self.feature_names].astype(float)

                # Preprocessing step 1: Outlier Percentile Clipping (matching training)
                if self.clipping_bounds is not None and "q01" in self.clipping_bounds and "q99" in self.clipping_bounds:
                    q01 = pd.Series(self.clipping_bounds["q01"])
                    q99 = pd.Series(self.clipping_bounds["q99"])
                    X_clipped = X_raw.clip(lower=q01, upper=q99, axis=1)
                else:
                    X_clipped = X_raw

                # Preprocessing step 2: StandardScaler (fitted on train ONLY)
                X_scaled = self.scaler.transform(X_clipped)

                # Preprocessing step 3: PCA (~95% variance)
                X_pca = self.pca.transform(X_scaled) if self.pca is not None else None

                # Subsystem-specific scoring with matching feature configuration
                for comp in ["engine", "brake", "battery"]:
                    if comp in self.models:
                        try:
                            m = self.models[comp]
                            meta = self.champions_summary.get(comp, {})
                            feat_set = meta.get("Features", "Original_Scaled")
                            uses_pca = (feat_set == "PCA_95_Variance") or meta.get("preprocessing", {}).get("pca", False)

                            if uses_pca:
                                if X_pca is None:
                                    raise RuntimeError(f"Champion for '{comp}' requires PCA, but PCA model is not loaded.")
                                X_input = X_pca
                            else:
                                X_input = X_scaled

                            if hasattr(m, "predict_proba"):
                                prob = float(m.predict_proba(X_input)[0, 1])
                            else:
                                prob = float(m.predict(X_input)[0])

                            subsystem_probabilities[comp] = float(np.clip(prob, 0.001, 0.999))
                            subsystem_status[comp] = "OK"
                        except Exception as e:
                            logger.error(f"Inference error in subsystem '{comp}': {e}", exc_info=True)
                            ml_errors[comp] = str(e)
                            subsystem_status[comp] = "ERROR"
                            subsystem_probabilities[comp] = None
                    else:
                        err_msg = f"Champion model for '{comp}' not loaded"
                        logger.warning(err_msg)
                        ml_errors[comp] = err_msg
                        subsystem_status[comp] = "UNAVAILABLE"
                        subsystem_probabilities[comp] = None

            except Exception as e:
                logger.error(f"Feature transformation failure: {e}", exc_info=True)
                for comp in ["engine", "brake", "battery"]:
                    ml_errors[comp] = f"Preprocessing failed: {str(e)}"
                    subsystem_status[comp] = "ERROR"
                    subsystem_probabilities[comp] = None
        else:
            for comp in ["engine", "brake", "battery"]:
                err_msg = "Artifacts not loaded or incomplete"
                ml_errors[comp] = err_msg
                subsystem_status[comp] = "UNAVAILABLE"
                subsystem_probabilities[comp] = None

        if ml_errors:
            ml_status = "UNAVAILABLE" if len(ml_errors) == 3 else "DEGRADED"

        # 6. Centralized Physical Sensor Safety Checks (Configuration-Aware Safety Net)
        safety_alerts = []
        eth = self.safety_thresholds.get("engine", DEFAULT_SAFETY_THRESHOLDS["engine"])
        bth = self.safety_thresholds.get("brake", DEFAULT_SAFETY_THRESHOLDS["brake"])
        batth = self.safety_thresholds.get("battery", DEFAULT_SAFETY_THRESHOLDS["battery"])

        # Fetch vehicle-specific thresholds and configurations
        vt = getattr(self, "vehicle_thresholds", {}).get(vid, {})
        master_meta = getattr(self, "vehicle_master", {}).get(vid, {})

        profile = row.get("profile") or vt.get("profile") or master_meta.get("profile", "A")
        raw_elec = row.get("electrical_system") or vt.get("electrical_system") or master_meta.get("electrical_system")
        raw_brake = row.get("brake_configuration") or vt.get("brake_configuration") or master_meta.get("brake_configuration")

        config_flags = []
        config_status = "VALID"

        # Check Electrical System Configuration
        is_elec_valid = raw_elec in getattr(self, "supported_electrical_systems", {"12V", "24V"})
        if not is_elec_valid:
            config_status = "UNKNOWN_CONFIGURATION"
            msg = f"Unknown electrical system configuration for {vid} ('{raw_elec}'). 12V/24V thresholds suspended."
            config_flags.append(msg)
            safety_alerts.append(f"CONFIGURATION WARNING: {msg}")

        # Check Brake Configuration
        is_brake_valid = raw_brake in getattr(self, "supported_brake_configurations", {
            "Hydraulic Disc", "Hydraulic Disc/Drum", "Pneumatic/Hydraulic Disc", "Pneumatic S-Cam/Disc"
        })
        if not is_brake_valid:
            config_status = "UNKNOWN_CONFIGURATION"
            msg = f"Unknown brake configuration for {vid} ('{raw_brake}'). Brake wear thresholds suspended."
            config_flags.append(msg)
            safety_alerts.append(f"CONFIGURATION WARNING: {msg}")

        # Engine Safety Checks: High engine temperature triggers alerts
        eng_warn = vt.get("engine_warning_temp_c", eth.get("overheat_warning_temp_c", 105.0))
        eng_crit = vt.get("engine_critical_temp_c", eth.get("overheat_critical_temp_c", 112.0))
        if eng_temp >= eng_crit:
            override = 0.94
            subsystem_probabilities["engine"] = max(subsystem_probabilities.get("engine") or 0.0, override)
            safety_alerts.append(f"Engine Overheating ({eng_temp:.1f} deg C)")
        elif eng_temp >= eng_warn or cool_temp >= eth.get("coolant_overheat_temp_c", 108.0):
            override = 0.48
            subsystem_probabilities["engine"] = max(subsystem_probabilities.get("engine") or 0.0, override)
            safety_alerts.append(f"Engine Temperature Warning ({eng_temp:.1f} deg C)")

        if oil_psi <= eth.get("low_oil_critical_psi", 15.0):
            override = 0.96
            subsystem_probabilities["engine"] = max(subsystem_probabilities.get("engine") or 0.0, override)
            safety_alerts.append(f"Low Oil Pressure Critical ({oil_psi:.1f} PSI)")
        elif oil_psi <= eth.get("low_oil_warning_psi", 22.0):
            override = 0.50
            subsystem_probabilities["engine"] = max(subsystem_probabilities.get("engine") or 0.0, override)
            safety_alerts.append(f"Low Oil Pressure Warning ({oil_psi:.1f} PSI)")

        if eng_vib >= eth.get("excessive_vibration_mms", 3.2):
            override = 0.45
            subsystem_probabilities["engine"] = max(subsystem_probabilities.get("engine") or 0.0, override)
            safety_alerts.append(f"Excessive Engine Vibration ({eng_vib:.2f} mm/s)")

        # Brake Safety Checks: Low brake thickness indicates wear (applied ONLY when configuration is supported)
        brake_warn = None
        brake_crit = None
        if is_brake_valid:
            brake_warn = vt.get("brake_warning_thickness_mm", bth.get("pad_wear_warning_mm", 3.2))
            brake_crit = vt.get("brake_critical_thickness_mm", bth.get("pad_wear_critical_mm", 2.2))
            if brake_pad <= brake_crit:
                override = 0.98
                subsystem_probabilities["brake"] = max(subsystem_probabilities.get("brake") or 0.0, override)
                safety_alerts.append(f"Worn Brake Pads Critical ({brake_pad:.2f} mm)")
            elif brake_pad <= brake_warn:
                override = 0.48
                subsystem_probabilities["brake"] = max(subsystem_probabilities.get("brake") or 0.0, override)
                safety_alerts.append(f"Worn Brake Pads Warning ({brake_pad:.2f} mm)")

        if brake_temp >= bth.get("thermal_spike_c", 260.0):
            override = 0.52
            subsystem_probabilities["brake"] = max(subsystem_probabilities.get("brake") or 0.0, override)
            safety_alerts.append(f"Brake Thermal Warning ({brake_temp:.1f} deg C)")
        if abs_fault == bth.get("abs_fault_indicator", 1):
            override = 0.55
            subsystem_probabilities["brake"] = max(subsystem_probabilities.get("brake") or 0.0, override)
            safety_alerts.append("ABS System Fault Indicator Active")

        # Battery Safety Checks: Low battery voltage triggers alerts (applied ONLY when 12V or 24V configuration supports them)
        batt_warn = None
        batt_crit = None
        alt_warn = None
        alt_crit = None
        if is_elec_valid:
            if raw_elec == "24V":
                batt_warn = vt.get("battery_warning_voltage_v", 23.6)
                batt_crit = vt.get("battery_critical_voltage_v", 22.0)
                alt_warn = 25.6
                alt_crit = 24.4
            else:  # "12V"
                batt_warn = vt.get("battery_warning_voltage_v", batth.get("voltage_warning_v", 11.8))
                batt_crit = vt.get("battery_critical_voltage_v", batth.get("voltage_critical_v", 11.0))
                alt_warn = batth.get("alternator_undervoltage_v", 12.8)
                alt_crit = 12.2

            if batt_v <= batt_crit:
                override = 0.98
                subsystem_probabilities["battery"] = max(subsystem_probabilities.get("battery") or 0.0, override)
                safety_alerts.append(f"Critically Low Battery Voltage ({batt_v:.2f}V)")
            elif batt_v <= batt_warn:
                override = 0.50
                subsystem_probabilities["battery"] = max(subsystem_probabilities.get("battery") or 0.0, override)
                safety_alerts.append(f"Battery Voltage Warning ({batt_v:.2f}V)")

            if alt_v <= alt_crit:
                override = 0.88
                subsystem_probabilities["battery"] = max(subsystem_probabilities.get("battery") or 0.0, override)
                safety_alerts.append(f"Alternator Charging Malfunction ({alt_v:.2f}V)")
            elif alt_v <= alt_warn:
                override = 0.48
                subsystem_probabilities["battery"] = max(subsystem_probabilities.get("battery") or 0.0, override)
                safety_alerts.append(f"Alternator Undervoltage Warning ({alt_v:.2f}V)")

        if batt_soh <= batth.get("soh_critical_pct", 40.0):
            override = 0.92
            subsystem_probabilities["battery"] = max(subsystem_probabilities.get("battery") or 0.0, override)
            safety_alerts.append(f"Battery SOH Severely Degraded ({batt_soh:.1f}%)")
        elif batt_soh <= batth.get("soh_warning_pct", 50.0):
            override = 0.46
            subsystem_probabilities["battery"] = max(subsystem_probabilities.get("battery") or 0.0, override)
            safety_alerts.append(f"Battery SOH Warning ({batt_soh:.1f}%)")

        # 7. Synthesize Overall Risk Score & Calibrated Joint Failure Probability
        valid_probs = [v for v in subsystem_probabilities.values() if v is not None]
        has_physical_alert = any(not a.startswith("CONFIGURATION WARNING") for a in safety_alerts)

        if not valid_probs and not has_physical_alert:
            # Model inference failed completely and no physical sensor breaches: do NOT falsely return Healthy or Critical!
            overall_risk = None
            joint_prob = None
            status = "UNAVAILABLE"
            pred_type = "UNKNOWN"
            safety_alerts.append(f"Telemetry Model Inference Unavailable ({'; '.join(ml_errors.values())})")
        else:
            # overall_risk_score: peak severity among subsystems
            overall_risk = max(valid_probs) if valid_probs else 0.75
            
            # joint_failure_probability: statistical combination assuming conditional independence
            if valid_probs:
                p_e = subsystem_probabilities.get("engine") or 0.0
                p_b = subsystem_probabilities.get("brake") or 0.0
                p_bat = subsystem_probabilities.get("battery") or 0.0
                joint_prob = 1.0 - ((1.0 - p_e) * (1.0 - p_b) * (1.0 - p_bat))
                joint_prob = float(np.clip(joint_prob, 0.001, 0.999))
            else:
                joint_prob = overall_risk

            if overall_risk >= 0.60:
                status = "CRITICAL"
            elif overall_risk >= 0.30:
                status = "WARNING"
            else:
                if ml_status in ["UNAVAILABLE", "DEGRADED"]:
                    status = "WARNING"
                    safety_alerts.append(f"Degraded Subsystem Models ({'; '.join(f'{k}: {v}' for k, v in ml_errors.items())})")
                else:
                    status = "HEALTHY"

            # 8. Determine Specific Predicted Failure Type (Configuration-Aware)
            if overall_risk < 0.30 and not ml_errors:
                pred_type = "Healthy"
            else:
                p_eng = subsystem_probabilities.get("engine") or 0.0
                p_brk = subsystem_probabilities.get("brake") or 0.0
                p_bat = subsystem_probabilities.get("battery") or 0.0

                if p_eng >= max(p_brk, p_bat):
                    pred_type = "Engine_Overheat" if (eng_temp >= 102.0 or cool_temp >= 104.0) else "Low_Oil_Pressure"
                elif p_brk >= p_bat:
                    if brake_temp >= 240.0:
                        pred_type = "Brake_Overheat"
                    elif is_brake_valid and brake_warn is not None and brake_pad <= brake_warn:
                        pred_type = "Brake_Pad_Worn"
                    else:
                        pred_type = "ABS_Malfunction"
                else:
                    if is_elec_valid and alt_warn is not None and alt_v <= alt_warn:
                        pred_type = "Alternator_Failure"
                    else:
                        pred_type = "Battery_Degraded"

        return {
            "overall_risk_score": round(overall_risk, 4) if overall_risk is not None else None,
            "failure_probability": round(overall_risk, 4) if overall_risk is not None else None, # Backwards-compatible alias for peak subsystem risk
            "estimated_joint_failure_probability": round(joint_prob, 4) if joint_prob is not None else None,
            "joint_failure_probability": round(joint_prob, 4) if joint_prob is not None else None, # Backwards-compatible alias
            "joint_probability_status": "UNCALIBRATED_HEURISTIC_ESTIMATE",
            "status": status,
            "predicted_failure_type": pred_type,
            "ml_status": ml_status,
            "ml_errors": ml_errors,
            "prediction_valid": (ml_status == "AVAILABLE"),
            "subsystem_probabilities": {
                k: round(v, 4) if v is not None else None
                for k, v in subsystem_probabilities.items()
            },
            "subsystem_status": subsystem_status,
            "safety_alerts": safety_alerts,
            "profile": profile,
            "electrical_system": raw_elec,
            "brake_configuration": raw_brake,
            "configuration_status": config_status,
            "configuration_flags": config_flags,
            "data_quality_report": dq_report,
            "data_quality_status": dq_report.get("status", "HIGH"),
            "imputed_sensors": dq_report.get("imputed_sensors", []),
            "imputed_count": dq_report.get("imputed_count", 0),
            "sensor_errors": dq_report.get("sensor_errors", []),
            "data_quality_warning": dq_report.get("warning_msg", "")
        }


# Global singleton helper
_predictor_instance = None

def get_predictor() -> UnifiedVehiclePredictor:
    global _predictor_instance
    if _predictor_instance is None:
        _predictor_instance = UnifiedVehiclePredictor()
    return _predictor_instance

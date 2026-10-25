"""
AI-Powered Connected Vehicle Analytics Platform (BDA)
Causal Vehicle-Wise Data Cleaner & Data Quality Auditor

Key Capabilities:
1. Physical Limits vs Genuine Failure Signals:
   - Differentiates physical sensor anomalies (e.g. speed < 0 or speed > 220 km/h, battery > 30V)
     from genuine component failure signals (e.g. engine_temp = 118 deg C is overheating,
     brake_pad = 1.5mm is critical wear). Genuine failure signals are strictly preserved.
2. Causal, Backward-Looking Imputation:
   - For each vehicle, fills missing or invalid sensor values using the rolling median of its
     previous valid readings (default window = 3).
   - Never uses future readings for current pings (strictly zero lookahead).
   - If no vehicle history exists (e.g. cold start / first ping), falls back to training partition
     medians and flags data quality as DEGRADED.
3. Stale Sensor Enforcement:
   - In streaming inference, enforces a maximum age (default 600s) on previous valid readings
     to prevent stale history from being treated as current.
4. Target Leakage Prevention:
   - Data quality metrics (imputed counts, quality flags) are tracked exclusively as operational
     metadata and are NEVER passed as feature inputs into ML models.
"""

import datetime
from collections import defaultdict, deque
from typing import Dict, List, Tuple, Any, Optional
import numpy as np
import pandas as pd

# Physical operational boundaries for automotive sensors
# Values outside these bounds represent electrical/hardware transmission glitches
SENSOR_PHYSICAL_BOUNDS: Dict[str, Tuple[float, float]] = {
    # Identity & Powertrain
    "odometer_km": (0.0, 3000000.0),
    "engine_hours": (0.0, 60000.0),
    "vehicle_speed_kph": (0.0, 220.0),
    "engine_temp_c": (-40.0, 180.0),         # 105-130C is overheat failure; >180C is sensor glitch
    "engine_rpm": (0.0, 9000.0),
    "oil_pressure_psi": (0.0, 120.0),        # 0-20 PSI is low oil failure; negative or >120 is glitch
    "coolant_temp_c": (-40.0, 160.0),
    "fuel_level_percent": (0.0, 100.0),
    "fuel_consumption_lph": (0.0, 100.0),
    "engine_load_percent": (0.0, 100.0),
    "throttle_position_percent": (0.0, 100.0),
    "air_flow_rate_gps": (0.0, 250.0),
    "exhaust_gas_temp_c": (0.0, 1200.0),
    "engine_vibration": (0.0, 15.0),
    # Braking Subsystem
    "brake_pad_wear_mm": (0.0, 20.0),        # 1.0-3.0mm is severe wear failure; negative or >20 is glitch
    "brake_fluid_level": (0.0, 100.0),
    "brake_temp_c": (-40.0, 600.0),
    "abs_fault_indicator": (0, 1),
    "brake_pedal_position": (0.0, 100.0),
    "wheel_speed_fl_kph": (0.0, 220.0),
    "wheel_speed_fr_kph": (0.0, 220.0),
    "wheel_speed_rl_kph": (0.0, 220.0),
    "wheel_speed_rr_kph": (0.0, 220.0),
    # Electrical Subsystem
    "battery_voltage_v": (5.0, 36.0),        # Supports 12V and 24V commercial vehicle electrical systems
    "battery_current_a": (-250.0, 250.0),
    "battery_temp_c": (-40.0, 110.0),
    "alternator_output_v": (0.0, 36.0),
    "battery_charge_percent": (0.0, 100.0),
    "battery_health_percent": (0.0, 100.0),
    # Environment & Location
    "ambient_temp_c": (-50.0, 65.0),
    "gps_latitude": (-90.0, 90.0),
    "gps_longitude": (-180.0, 180.0),
}

SUPPORTED_ELECTRICAL_SYSTEMS = {"12V", "24V"}
SUPPORTED_BRAKE_CONFIGURATIONS = {
    "Hydraulic Disc",
    "Hydraulic Disc/Drum",
    "Pneumatic/Hydraulic Disc",
    "Pneumatic S-Cam/Disc"
}

CORE_MANDATORY_FIELDS = [
    "engine_temp_c",
    "oil_pressure_psi",
    "brake_pad_wear_mm",
    "battery_voltage_v"
]

ALL_TELEMETRY_SENSORS = list(SENSOR_PHYSICAL_BOUNDS.keys())


def detect_and_mask_sensor_errors(row: Dict[str, Any]) -> Tuple[Dict[str, Any], List[str]]:
    """
    Checks each sensor against physical bounds.
    If outside physical limits, flags the sensor error and masks it to None for causal imputation.
    Genuine failure signals within physical limits are kept intact.
    Audits electrical system and brake configuration; flags unknown configurations.
    """
    sanitized = dict(row)
    sensor_errors = []
    for sensor, (low, high) in SENSOR_PHYSICAL_BOUNDS.items():
        val = sanitized.get(sensor)
        if val is not None and not (isinstance(val, float) and np.isnan(val)):
            try:
                num = float(val)
                if num < low or num > high:
                    sensor_errors.append(f"{sensor}={num} (limit [{low}, {high}])")
                    sanitized[sensor] = None
            except (ValueError, TypeError):
                sensor_errors.append(f"{sensor}={val} (non-numeric)")
                sanitized[sensor] = None

    # Audit electrical system configuration
    elec = sanitized.get("electrical_system")
    if elec is not None and str(elec).strip() and elec not in SUPPORTED_ELECTRICAL_SYSTEMS:
        sensor_errors.append(f"electrical_system='{elec}' (unsupported/unknown)")

    # Audit brake configuration
    brake = sanitized.get("brake_configuration")
    if brake is not None and str(brake).strip() and brake not in SUPPORTED_BRAKE_CONFIGURATIONS:
        sensor_errors.append(f"brake_configuration='{brake}' (unsupported/unknown)")

    return sanitized, sensor_errors


def clean_batch_causal(
    df: pd.DataFrame,
    sensor_cols: Optional[List[str]] = None,
    window: int = 3,
    train_medians: Optional[Dict[str, float]] = None
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Applies vehicle-wise causal missing-value imputation and outlier cleaning to a DataFrame.
    
    Process:
    1. Sorts by vehicle_id and timestamp.
    2. Identifies missing and out-of-bounds sensor values.
    3. For each vehicle, imputes missing values using rolling median of past k valid readings.
    4. If no prior valid history exists (e.g. row 0 of vehicle), uses train_medians fallback.
    5. Preserves genuine failure signals and original labels unchanged.
    """
    if sensor_cols is None:
        sensor_cols = [c for c in ALL_TELEMETRY_SENSORS if c in df.columns]

    df_clean = df.copy()
    if "timestamp" in df_clean.columns:
        df_clean["timestamp"] = pd.to_datetime(df_clean["timestamp"])
    
    # Sort strictly by vehicle_id and timestamp
    sort_cols = [c for c in ["vehicle_id", "timestamp"] if c in df_clean.columns]
    if sort_cols:
        df_clean = df_clean.sort_values(by=sort_cols).reset_index(drop=True)

    # Detect & mask physical boundary violations
    sensor_error_counts: Dict[str, int] = defaultdict(int)
    for col in sensor_cols:
        if col in SENSOR_PHYSICAL_BOUNDS:
            low, high = SENSOR_PHYSICAL_BOUNDS[col]
            s = df_clean[col]
            outliers = (s < low) | (s > high)
            n_out = int(outliers.sum())
            if n_out > 0:
                sensor_error_counts[col] += n_out
                df_clean.loc[outliers, col] = np.nan

    # Compute baseline fallback medians if not provided
    if train_medians is None:
        train_medians = df_clean[sensor_cols].median().to_dict()

    total_imputed_history = 0
    total_imputed_fallback = 0

    imputed_mask = pd.DataFrame(False, index=df_clean.index, columns=sensor_cols)

    # Perform causal filling per vehicle
    if "vehicle_id" in df_clean.columns:
        groups = df_clean.groupby("vehicle_id").groups
    else:
        groups = {"single_fleet": df_clean.index}

    for vid, group_indices in groups.items():
        for col in sensor_cols:
            series = df_clean.loc[group_indices, col]
            if series.isnull().any():
                vals = np.array(series, dtype=float, copy=True)
                history = []
                for i in range(len(vals)):
                    v = vals[i]
                    if np.isnan(v):
                        imputed_mask.loc[group_indices[i], col] = True
                        if len(history) > 0:
                            vals[i] = float(np.median(history[-window:]))
                            total_imputed_history += 1
                        else:
                            fallback_val = train_medians.get(col, 0.0)
                            vals[i] = float(fallback_val) if not np.isnan(fallback_val) else 0.0
                            total_imputed_fallback += 1
                        history.append(vals[i])
                    else:
                        history.append(v)
                df_clean.loc[group_indices, col] = vals

    imputed_count_per_row = imputed_mask.sum(axis=1)
    
    # Classify data quality status per record (for monitoring/reporting ONLY, NOT for ML features)
    quality_status = pd.Series("HIGH", index=df_clean.index)
    quality_status[imputed_count_per_row >= 1] = "DEGRADED"
    quality_status[imputed_count_per_row >= 4] = "POOR"

    # Audit configuration columns if present
    unknown_configs = []
    if "electrical_system" in df_clean.columns:
        invalid_elec = df_clean[~df_clean["electrical_system"].isna() & ~df_clean["electrical_system"].isin(SUPPORTED_ELECTRICAL_SYSTEMS)]
        if not invalid_elec.empty:
            unknown_configs.append(f"Found {len(invalid_elec)} rows with unsupported electrical_system")
    if "brake_configuration" in df_clean.columns:
        invalid_brake = df_clean[~df_clean["brake_configuration"].isna() & ~df_clean["brake_configuration"].isin(SUPPORTED_BRAKE_CONFIGURATIONS)]
        if not invalid_brake.empty:
            unknown_configs.append(f"Found {len(invalid_brake)} rows with unsupported brake_configuration")

    stats = {
        "total_records": len(df_clean),
        "total_imputed_history": total_imputed_history,
        "total_imputed_fallback": total_imputed_fallback,
        "sensor_error_counts": dict(sensor_error_counts),
        "unknown_configuration_warnings": unknown_configs,
        "high_quality_count": int((quality_status == "HIGH").sum()),
        "degraded_quality_count": int((quality_status == "DEGRADED").sum()),
        "poor_quality_count": int((quality_status == "POOR").sum()),
        "train_medians": train_medians
    }

    return df_clean, stats


class LiveCausalVehicleBuffer:
    """
    Maintains causal in-memory rolling history per vehicle for streaming inference.
    Enforces maximum age for historical sensor readings and causal rolling median imputation.
    """
    def __init__(self, window: int = 3, max_age_seconds: int = 600, train_medians: Optional[Dict[str, float]] = None):
        self.window = window
        self.max_age_seconds = max_age_seconds
        self.train_medians = train_medians or {}
        # vid -> list of {sensor: val, "timestamp": datetime}
        self.vehicle_history: Dict[str, List[Dict[str, Any]]] = defaultdict(list)

    def set_train_medians(self, medians: Dict[str, float]):
        self.train_medians = medians

    def clear(self, vehicle_id: Optional[str] = None):
        if vehicle_id:
            if vehicle_id in self.vehicle_history:
                self.vehicle_history[vehicle_id].clear()
        else:
            self.vehicle_history.clear()

    def process_telemetry(self, raw_record: Dict[str, Any]) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """
        Validates, cleans, and causally imputes a single incoming streaming ping.
        
        Returns:
            cleaned_record: complete sensor dictionary ready for ML feature generation.
            data_quality_report:
                - status: 'HIGH', 'DEGRADED', 'POOR', 'STALE', or 'UNAVAILABLE'
                - imputed_sensors: list of sensor names that were missing and imputed
                - sensor_errors: list of sensors that violated physical boundaries
                - is_stale: whether previous history exceeded maximum age
                - warning_msg: operator-friendly message
        """
        vid = str(raw_record.get("vehicle_id", "default_vh"))
        ts_str = raw_record.get("timestamp")
        
        # Parse timestamp
        current_dt = None
        if ts_str:
            try:
                current_dt = pd.to_datetime(ts_str, utc=True)
            except Exception:
                current_dt = datetime.datetime.now(datetime.timezone.utc)
        else:
            current_dt = datetime.datetime.now(datetime.timezone.utc)

        # 1. Mask physical sensor errors
        sanitized, sensor_errors = detect_and_mask_sensor_errors(raw_record)

        cleaned = dict(sanitized)
        imputed_sensors = []
        is_stale = False
        unresolved_mandatory = []

        history = self.vehicle_history[vid]
        
        # Check staleness of previous history
        if history:
            last_ts = history[-1].get("_internal_dt")
            if last_ts and current_dt:
                delta_sec = abs((current_dt - last_ts).total_seconds())
                if delta_sec > self.max_age_seconds:
                    is_stale = True

        # 2. Causal Imputation per sensor
        for sensor in ALL_TELEMETRY_SENSORS:
            val = cleaned.get(sensor)
            is_missing = val is None or (isinstance(val, float) and np.isnan(val))

            if is_missing:
                # Look back at vehicle's own history
                valid_past_vals = [
                    h[sensor] for h in history
                    if sensor in h and h[sensor] is not None and not (isinstance(h[sensor], float) and np.isnan(h[sensor]))
                ]

                if valid_past_vals and not is_stale:
                    # Rolling median of up to last k valid readings
                    recent_k = valid_past_vals[-self.window:]
                    cleaned[sensor] = float(np.median(recent_k))
                    imputed_sensors.append(sensor)
                elif sensor not in CORE_MANDATORY_FIELDS and sensor in self.train_medians:
                    # Fallback to train baseline median ONLY for non-mandatory secondary sensors
                    fallback = self.train_medians[sensor]
                    cleaned[sensor] = float(fallback)
                    imputed_sensors.append(sensor)
                else:
                    if sensor in CORE_MANDATORY_FIELDS:
                        unresolved_mandatory.append(sensor)

        # 3. Store valid readings into vehicle history
        hist_entry = dict(cleaned)
        hist_entry["_internal_dt"] = current_dt
        self.vehicle_history[vid].append(hist_entry)
        if len(self.vehicle_history[vid]) > 10:
            self.vehicle_history[vid].pop(0)

        # 4. Formulate Data Quality Status
        if unresolved_mandatory:
            status = "UNAVAILABLE"
            warning_msg = f"Critical sensors missing without history: {', '.join(unresolved_mandatory)}"
        elif is_stale:
            status = "STALE"
            warning_msg = f"Sensor history is stale (> {self.max_age_seconds}s). Imputations uncalibrated."
        elif len(imputed_sensors) >= 4 or len(sensor_errors) >= 2:
            status = "POOR"
            warning_msg = f"High telemetry noise: {len(imputed_sensors)} sensors imputed, {len(sensor_errors)} boundary errors."
        elif len(imputed_sensors) > 0 or len(sensor_errors) > 0:
            status = "DEGRADED"
            warning_msg = f"Sensors imputed from history: {', '.join(imputed_sensors[:3])}"
        else:
            status = "HIGH"
            warning_msg = "All sensors valid within nominal boundaries."

        # Check configuration validity and flag unknown configurations
        elec = cleaned.get("electrical_system")
        brake = cleaned.get("brake_configuration")
        config_errors = []
        if elec is not None and str(elec).strip() and elec not in SUPPORTED_ELECTRICAL_SYSTEMS:
            config_errors.append(f"Unknown electrical system configuration: '{elec}'")
        if brake is not None and str(brake).strip() and brake not in SUPPORTED_BRAKE_CONFIGURATIONS:
            config_errors.append(f"Unknown brake configuration: '{brake}'")

        configuration_status = "UNKNOWN_CONFIGURATION" if config_errors else "VALID"

        data_quality_report = {
            "status": status,
            "data_quality_status": status,
            "configuration_status": configuration_status,
            "configuration_errors": config_errors,
            "imputed_sensors": imputed_sensors,
            "imputed_count": len(imputed_sensors),
            "sensor_errors": sensor_errors,
            "is_stale": is_stale,
            "warning_msg": warning_msg,
            "has_warnings": (status != "HIGH" or configuration_status != "VALID")
        }

        return cleaned, data_quality_report

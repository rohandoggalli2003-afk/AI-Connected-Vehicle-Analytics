"""
AI-Powered Connected Vehicle Analytics Platform (BDA)
Telematics Synthetic Data Generator

Generates:
1. Static Vehicle Master Table (vehicle_id, brand, vehicle_year, model, fuel_type)
2. Streaming Telematics Records (34 columns) matching exact specification
3. Ground-truth Labels for ML training (engine_failure_imminent, brake_issue_imminent, battery_issue_imminent, failure_type)
"""

import os
import json
import random
import datetime
import numpy as np
import pandas as pd
from typing import List, Dict, Tuple

# Set reproducible seed
np.random.seed(42)
random.seed(42)

# Vehicle Master configuration
BRANDS_MODELS = {
    "Toyota": [("Camry", "Gasoline"), ("RAV4", "Hybrid"), ("Tundra", "Gasoline")],
    "Ford": [("F-150", "Gasoline"), ("Explorer", "Gasoline"), ("Mach-E", "Electric")],
    "Volvo": [("VNL 760", "Diesel"), ("XC90", "Hybrid"), ("FH16", "Diesel")],
    "Tesla": [("Model 3", "Electric"), ("Model Y", "Electric"), ("Semi", "Electric")],
    "Freightliner": [("Cascadia", "Diesel"), ("M2 106", "Diesel")],
    "BMW": [("X5", "Hybrid"), ("330i", "Gasoline"), ("i4", "Electric")]
}

# GPS Fleet Centers (realistic fleet operational zones)
FLEET_HUBS = [
    {"name": "San Francisco Bay", "lat": 37.7749, "lon": -122.4194},
    {"name": "Los Angeles Metro", "lat": 34.0522, "lon": -118.2437},
    {"name": "Chicago Hub", "lat": 41.8781, "lon": -87.6298},
    {"name": "Dallas Logistics", "lat": 32.7767, "lon": -96.7970},
    {"name": "Atlanta Distribution", "lat": 33.7490, "lon": -84.3880}
]

def generate_vehicle_master(num_vehicles: int = 50) -> pd.DataFrame:
    """Generate static Vehicle Master Table."""
    vehicles = []
    brands = list(BRANDS_MODELS.keys())
    
    for i in range(1, num_vehicles + 1):
        vehicle_id = f"VH-{1000 + i}"
        brand = random.choice(brands)
        model_info = random.choice(BRANDS_MODELS[brand])
        model, fuel_type = model_info
        year = random.randint(2018, 2024)
        fleet_hub = random.choice(FLEET_HUBS)
        
        vehicles.append({
            "vehicle_id": vehicle_id,
            "brand": brand,
            "vehicle_year": year,
            "model": model,
            "fuel_type": fuel_type,
            "base_hub": fleet_hub["name"],
            "base_lat": fleet_hub["lat"],
            "base_lon": fleet_hub["lon"]
        })
    
    df = pd.DataFrame(vehicles)
    return df
SUPPORTED_ELECTRICAL_SYSTEMS = {"12V", "24V"}
SUPPORTED_BRAKE_CONFIGURATIONS = {
    "Hydraulic Disc",
    "Hydraulic Disc/Drum",
    "Pneumatic/Hydraulic Disc",
    "Pneumatic S-Cam/Disc"
}

class VehicleState:
    """Maintains continuous physical state for a vehicle to create realistic time-series."""
    def __init__(self, master_row: dict):
        self.vehicle_id = master_row["vehicle_id"]
        self.brand = master_row["brand"]
        self.fuel_type = master_row["fuel_type"]
        self.year = master_row["vehicle_year"]
        self.profile = master_row.get("profile", "A")
        
        # Verify electrical system and brake configuration; flag unknown configurations
        raw_elec = master_row.get("electrical_system")
        raw_brake = master_row.get("brake_configuration")
        
        self.is_unknown_electrical = raw_elec not in SUPPORTED_ELECTRICAL_SYSTEMS
        self.is_unknown_brake = raw_brake not in SUPPORTED_BRAKE_CONFIGURATIONS
        
        self.electrical_system = raw_elec if not self.is_unknown_electrical else f"UNKNOWN ({raw_elec})"
        self.brake_configuration = raw_brake if not self.is_unknown_brake else f"UNKNOWN ({raw_brake})"
        
        # Base physical state
        age_years = max(1, 2025 - int(self.year))
        self.odometer = float(random.randint(15000 * age_years, 35000 * age_years))
        self.engine_hours = float(self.odometer / random.uniform(35.0, 50.0))
        self.lat = float(master_row["base_lat"]) + random.uniform(-0.15, 0.15)
        self.lon = float(master_row["base_lon"]) + random.uniform(-0.15, 0.15)
        
        # Wear components configured per profile
        if self.profile == "D":
            self.brake_pad_wear_mm = float(random.uniform(14.0, 19.5))
        elif self.profile == "B":
            self.brake_pad_wear_mm = float(random.uniform(8.5, 12.5))
        else:
            self.brake_pad_wear_mm = float(random.uniform(7.5, 11.5))

        self.brake_fluid_level = float(random.uniform(85.0, 98.0))
        self.battery_health_percent = float(random.uniform(82.0, 99.0))
        self.fuel_level_percent = float(random.uniform(40.0, 95.0))
        
        # Dynamic telemetry state
        self.is_driving = random.choice([True, True, False])
        self.speed = random.uniform(40.0, 105.0) if self.is_driving else 0.0
        self.ambient_temp = random.uniform(15.0, 32.0)
        
        # Injected failure mode (5% engine, 5% brake, 5% battery, 85% healthy)
        failure_roll = random.random()
        if failure_roll < 0.05:
            self.failure_mode = "Engine_Overheat"
        elif failure_roll < 0.09:
            self.failure_mode = "Low_Oil_Pressure"
        elif failure_roll < 0.14:
            self.failure_mode = "Brake_Pad_Worn"
        elif failure_roll < 0.18:
            self.failure_mode = "ABS_Malfunction"
        elif failure_roll < 0.22:
            self.failure_mode = "Battery_Degraded"
        elif failure_roll < 0.26:
            self.failure_mode = "Alternator_Failure"
        else:
            self.failure_mode = "None"
            
        # Degradation progress (0.0 to 1.0)
        self.degradation = 0.1 if self.failure_mode != "None" else 0.0

    def step(self, delta_seconds: int = 10) -> Tuple[dict, dict]:
        """Step forward vehicle physics and return (streaming_34_cols, label_dict)."""
        dt_hours = delta_seconds / 3600.0
        
        # Advance degradation if failing
        if self.failure_mode != "None":
            self.degradation = min(1.0, self.degradation + random.uniform(0.001, 0.008))
            
        # Driving state transition
        if random.random() < 0.05:
            self.is_driving = not self.is_driving
            
        if self.is_driving:
            target_speed = random.uniform(30.0, 110.0)
            self.speed = np.clip(self.speed + random.uniform(-10.0, 10.0), 0.0, 140.0)
            throttle = np.clip((self.speed / 130.0) * 80.0 + random.uniform(-10.0, 15.0), 5.0, 95.0)
            rpm = np.clip(850.0 + (self.speed * 28.0) + (throttle * 8.0) + random.uniform(-100, 100), 750.0, 5800.0)
            engine_load = np.clip((throttle * 0.8) + (self.speed * 0.2) + random.uniform(-5.0, 5.0), 10.0, 98.0)
            brake_pedal = 0.0
            if random.random() < 0.15: # Braking event
                brake_pedal = random.uniform(15.0, 75.0)
                self.speed = max(0.0, self.speed - random.uniform(5.0, 20.0))
                throttle = 0.0
                rpm = max(750.0, rpm - 600.0)
        else:
            self.speed = 0.0
            throttle = 0.0
            rpm = random.uniform(700.0, 850.0)
            engine_load = random.uniform(8.0, 15.0)
            brake_pedal = random.choice([0.0, 20.0])

        # Distance & engine hours
        dist_km = (self.speed * dt_hours)
        self.odometer += dist_km
        self.engine_hours += dt_hours
        
        # GPS movement
        if self.speed > 0:
            self.lat += random.uniform(-0.0005, 0.0005) * (self.speed / 50.0)
            self.lon += random.uniform(-0.0005, 0.0005) * (self.speed / 50.0)
            
        # Fuel consumption
        fuel_consumption_lph = (engine_load / 100.0) * 12.0 + (rpm / 4000.0) * 8.0 + random.uniform(-0.5, 0.5)
        fuel_consumption_lph = max(0.8, fuel_consumption_lph)
        self.fuel_level_percent = max(0.0, self.fuel_level_percent - (fuel_consumption_lph * dt_hours * 1.5))
        if self.fuel_level_percent < 5.0:
            self.fuel_level_percent = 95.0 # refueled
            
        # Air flow rate (grams per sec) & exhaust gas temp
        air_flow_rate_gps = (rpm / 1000.0) * 12.0 + (engine_load / 100.0) * 25.0 + random.uniform(-1.0, 1.0)
        exhaust_gas_temp_c = 320.0 + (engine_load * 3.5) + (rpm / 10.0) + random.uniform(-15.0, 15.0)
        
        # Engine Temperature & Oil Pressure with realistic operational variance
        # Driving uphill, towing load, or hot ambient temps elevate normal temps
        operational_thermal_boost = (engine_load * 0.15) + max(0.0, (self.ambient_temp - 25.0) * 0.3)
        base_engine_temp = 88.0 + operational_thermal_boost + np.random.normal(0, 2.5)
        coolant_temp_c = base_engine_temp - np.random.uniform(2.0, 6.0)
        
        # Oil pressure naturally varies with RPM and oil viscosity/temp
        oil_pressure_psi = 46.0 + (rpm / 160.0) - (base_engine_temp * 0.08) + np.random.normal(0, 3.0)
        engine_vibration = 0.85 + (rpm / 2400.0) * 0.5 + (engine_load / 100.0) * 0.4 + np.random.normal(0, 0.2)
        
        # Inject Engine Failures with realistic progressive degradation & intermittent symptoms
        if self.failure_mode == "Engine_Overheat" and self.degradation > 0.25:
            # Sticking thermostat / cooling loss creates fluctuating severe heat
            thermal_surge = (22.0 * self.degradation) + np.random.normal(0, 4.0)
            base_engine_temp += thermal_surge
            coolant_temp_c += thermal_surge * 0.95
            engine_vibration += 1.5 * self.degradation
        elif self.failure_mode == "Low_Oil_Pressure" and self.degradation > 0.25:
            oil_pressure_loss = (26.0 * self.degradation) + np.random.normal(0, 3.5)
            oil_pressure_psi = max(4.0, oil_pressure_psi - oil_pressure_loss)
            engine_vibration += 2.5 * self.degradation

        # Brakes: friction dynamics
        if brake_pedal > 0:
            self.brake_pad_wear_mm = max(0.8, self.brake_pad_wear_mm - (0.00008 * (self.speed + 15)))
            brake_temp_c = 75.0 + (brake_pedal * 2.4) + (self.speed * 1.2) + np.random.normal(0, 15.0)
        else:
            brake_temp_c = max(35.0, 70.0 - (dt_hours * 220.0) + np.random.normal(0, 5.0))
            
        abs_fault_indicator = 0
        slip = np.random.normal(0, 1.2)
        wheel_fl = max(0.0, self.speed + slip)
        wheel_fr = max(0.0, self.speed - slip)
        wheel_rl = max(0.0, self.speed + slip * 0.7)
        wheel_rr = max(0.0, self.speed - slip * 0.7)
        
        # Inject Brake Failures configured per profile
        if self.failure_mode == "Brake_Pad_Worn" and self.degradation > 0.25:
            if self.profile == "D":
                self.brake_pad_wear_mm = max(1.8, 7.5 - (5.0 * self.degradation) + np.random.normal(0, 0.3))
            elif self.profile == "B":
                self.brake_pad_wear_mm = max(1.1, 4.6 - (3.0 * self.degradation) + np.random.normal(0, 0.25))
            elif self.profile == "C":
                self.brake_pad_wear_mm = max(0.9, 3.8 - (2.5 * self.degradation) + np.random.normal(0, 0.2))
            else:
                self.brake_pad_wear_mm = max(1.1, 3.5 - (2.0 * self.degradation) + np.random.normal(0, 0.25))
            brake_temp_c += (65.0 * self.degradation) + np.random.normal(0, 12.0)
        elif self.failure_mode == "ABS_Malfunction" and self.degradation > 0.35:
            # Intermittent tone ring / sensor fault
            if random.random() < (self.degradation * 0.9):
                abs_fault_indicator = 1
                wheel_fr = max(0.0, wheel_fr * (0.60 + np.random.uniform(-0.1, 0.1)))

        # Battery & Electrical System (12V vs 24V architecture)
        is_24v = (self.electrical_system == "24V")
        is_12v = (self.electrical_system == "12V")
        if is_24v:
            alternator_output_v = 28.2 + np.random.normal(0, 0.4)
            battery_voltage_v = 27.4 + np.random.normal(0, 0.35)
        elif is_12v:
            alternator_output_v = 14.1 + np.random.normal(0, 0.35)
            battery_voltage_v = 13.7 + np.random.normal(0, 0.3)
        else:
            # Unknown electrical configuration: cannot assume 12V or 24V
            alternator_output_v = 0.0
            battery_voltage_v = 0.0

        battery_current_a = np.random.normal(12.0, 6.0) if self.speed > 0 else np.random.normal(-6.0, 4.0)
        battery_temp_c = self.ambient_temp + 8.0 + (battery_current_a * 0.15) + np.random.normal(0, 3.0)
        battery_charge_percent = np.clip(85.0 + np.random.normal(0, 6.0), 10.0, 100.0)
        
        # Inject Battery Failures (respecting electrical architecture)
        if self.failure_mode == "Battery_Degraded" and self.degradation > 0.25:
            self.battery_health_percent = max(30.0, 68.0 - (35.0 * self.degradation) + np.random.normal(0, 3.0))
            if is_24v:
                battery_voltage_v = max(20.2, 24.8 - (3.8 * self.degradation) + np.random.normal(0, 0.35))
            elif is_12v:
                battery_voltage_v = max(10.2, 12.2 - (1.9 * self.degradation) + np.random.normal(0, 0.3))
            battery_temp_c += 18.0 * self.degradation
        elif self.failure_mode == "Alternator_Failure" and self.degradation > 0.30:
            if is_24v:
                alternator_output_v = max(21.0, 25.0 - (3.2 * self.degradation) + np.random.normal(0, 0.35))
                battery_voltage_v = max(20.5, 24.0 - (2.8 * self.degradation) + np.random.normal(0, 0.3))
            elif is_12v:
                alternator_output_v = max(11.0, 12.5 - (1.6 * self.degradation) + np.random.normal(0, 0.3))
                battery_voltage_v = max(10.4, 11.9 - (1.4 * self.degradation) + np.random.normal(0, 0.25))
            battery_current_a = -30.0 * self.degradation + np.random.normal(0, 4.0)

        # Labels: Imminent failure defined by active severe degradation (>0.48 threshold with noise)
        is_engine_fail = int(self.failure_mode in ["Engine_Overheat", "Low_Oil_Pressure"] and (self.degradation + np.random.normal(0, 0.04)) > 0.48)
        is_brake_fail = int(self.failure_mode in ["Brake_Pad_Worn", "ABS_Malfunction"] and (self.degradation + np.random.normal(0, 0.04)) > 0.48)
        is_battery_fail = int(self.failure_mode in ["Battery_Degraded", "Alternator_Failure"] and (self.degradation + np.random.normal(0, 0.04)) > 0.48)
        failure_type_label = self.failure_mode if (is_engine_fail or is_brake_fail or is_battery_fail) else "None"

        # Exactly 34 streaming columns
        stream_record = {
            # Identity (2)
            "vehicle_id": self.vehicle_id,
            "timestamp": "",  # filled by caller with ISO timestamp
            # Engine (14)
            "odometer_km": round(self.odometer, 2),
            "engine_hours": round(self.engine_hours, 2),
            "vehicle_speed_kph": round(float(self.speed), 2),
            "engine_temp_c": round(float(base_engine_temp), 2),
            "engine_rpm": round(float(rpm), 1),
            "oil_pressure_psi": round(float(oil_pressure_psi), 2),
            "coolant_temp_c": round(float(coolant_temp_c), 2),
            "fuel_level_percent": round(float(self.fuel_level_percent), 2),
            "fuel_consumption_lph": round(float(fuel_consumption_lph), 2),
            "engine_load_percent": round(float(engine_load), 2),
            "throttle_position_percent": round(float(throttle), 2),
            "air_flow_rate_gps": round(float(air_flow_rate_gps), 2),
            "exhaust_gas_temp_c": round(float(exhaust_gas_temp_c), 2),
            "engine_vibration": round(float(engine_vibration), 3),
            # Brakes (9)
            "brake_pad_wear_mm": round(float(self.brake_pad_wear_mm), 2),
            "brake_fluid_level": round(float(self.brake_fluid_level), 2),
            "brake_temp_c": round(float(brake_temp_c), 2),
            "abs_fault_indicator": int(abs_fault_indicator),
            "brake_pedal_position": round(float(brake_pedal), 2),
            "wheel_speed_fl_kph": round(float(wheel_fl), 2),
            "wheel_speed_fr_kph": round(float(wheel_fr), 2),
            "wheel_speed_rl_kph": round(float(wheel_rl), 2),
            "wheel_speed_rr_kph": round(float(wheel_rr), 2),
            # Battery (6)
            "battery_voltage_v": round(float(battery_voltage_v), 2),
            "battery_current_a": round(float(battery_current_a), 2),
            "battery_temp_c": round(float(battery_temp_c), 2),
            "alternator_output_v": round(float(alternator_output_v), 2),
            "battery_charge_percent": round(float(battery_charge_percent), 2),
            "battery_health_percent": round(float(self.battery_health_percent), 2),
            # Environment and location (3)
            "ambient_temp_c": round(float(self.ambient_temp), 2),
            "gps_latitude": round(float(self.lat), 6),
            "gps_longitude": round(float(self.lon), 6)
        }
        
        labels = {
            "engine_failure_imminent": is_engine_fail,
            "brake_issue_imminent": is_brake_fail,
            "battery_issue_imminent": is_battery_fail,
            "failure_type": failure_type_label
        }
        
        return stream_record, labels

def inject_telematics_noise(record: dict, missing_prob: float = 0.05, noise_prob: float = 0.005) -> dict:
    """
    Optionally injects missing values (None) or physical boundary anomalies for testing
    data quality validation and causal imputation pipelines.
    """
    corrupted = dict(record)
    sensor_cols = [
        "odometer_km", "engine_hours", "vehicle_speed_kph", "engine_temp_c",
        "engine_rpm", "oil_pressure_psi", "coolant_temp_c", "fuel_level_percent",
        "fuel_consumption_lph", "engine_load_percent", "throttle_position_percent",
        "air_flow_rate_gps", "exhaust_gas_temp_c", "engine_vibration",
        "brake_pad_wear_mm", "brake_fluid_level", "brake_temp_c",
        "brake_pedal_position", "wheel_speed_fl_kph", "wheel_speed_fr_kph",
        "wheel_speed_rl_kph", "wheel_speed_rr_kph", "battery_voltage_v",
        "battery_current_a", "battery_temp_c", "alternator_output_v",
        "battery_charge_percent", "battery_health_percent", "ambient_temp_c",
        "gps_latitude", "gps_longitude"
    ]
    for col in sensor_cols:
        if col in corrupted:
            r = random.random()
            if r < missing_prob:
                corrupted[col] = None
            elif r < (missing_prob + noise_prob):
                # Out-of-bounds electrical glitch
                corrupted[col] = round(float(corrupted[col]) * 3.5 + 50.0, 2)
    return corrupted


def generate_telematics_datasets(
    output_dir: str = "data",
    num_vehicles: int = 50,
    historical_records_per_vehicle: int = 400,
    num_streaming_batches: int = 5,
    records_per_batch: int = 50
):
    """
    Generates all required data artifacts:
    1. vehicle_master.csv
    2. historical_labeled_telematics.csv & .parquet (for Phase B ML training)
    3. landing/batch_*.json files (for Phase A Auto Loader / streaming)
    """
    os.makedirs(output_dir, exist_ok=True)
    landing_dir = os.path.join(output_dir, "landing")
    os.makedirs(landing_dir, exist_ok=True)
    
    master_path = os.path.join(output_dir, "vehicle_master.csv")
    if os.path.exists(master_path):
        print(f"[*] Reusing existing Vehicle Master table: {master_path}")
        df_master = pd.read_csv(master_path)
    else:
        print(f"[*] Generating Vehicle Master table for {num_vehicles} vehicles...")
        df_master = generate_vehicle_master(num_vehicles)
        df_master.to_csv(master_path, index=False)
        print(f"    Saved: {master_path} ({len(df_master)} rows)")
    
    # Initialize vehicle state objects
    master_dict = df_master.to_dict(orient="records")
    vehicle_states = [VehicleState(row) for row in master_dict]
    
    # 1. Generate Historical Labeled Batch Dataset (for ML Phase B)
    print(f"[*] Generating historical labeled dataset ({num_vehicles * historical_records_per_vehicle} records)...")
    history_records = []
    
    start_time = datetime.datetime.now(datetime.timezone.utc) - datetime.timedelta(days=7)
    current_time = start_time
    time_step = datetime.timedelta(minutes=25)
    
    for step_idx in range(historical_records_per_vehicle):
        timestamp_str = current_time.strftime("%Y-%m-%d %H:%M:%S")
        for v in vehicle_states:
            record, labels = v.step(delta_seconds=1500)
            record["timestamp"] = timestamp_str
            # Combine telemetry with labels for ML training
            combined = {**record, **labels}
            history_records.append(combined)
        current_time += time_step
        
    df_history = pd.DataFrame(history_records)
    
    # Save CSV and Parquet with PermissionError resilience
    history_csv = os.path.join(output_dir, "historical_labeled_telematics.csv")
    history_parquet = os.path.join(output_dir, "historical_labeled_telematics.parquet")
    try:
        df_history.to_csv(history_csv, index=False)
    except PermissionError:
        print(f"[!] Warning: {history_csv} is locked by another program. Saving to historical_labeled_telematics_new.csv")
        df_history.to_csv(os.path.join(output_dir, "historical_labeled_telematics_new.csv"), index=False)
    try:
        df_history.to_parquet(history_parquet, index=False)
    except PermissionError:
        print(f"[!] Warning: {history_parquet} is locked. Saving to historical_labeled_telematics_new.parquet")
        df_history.to_parquet(os.path.join(output_dir, "historical_labeled_telematics_new.parquet"), index=False)
    print(f"    Saved historical dataset: {history_csv} & {history_parquet}")
    print(f"    Total rows: {len(df_history)}, Columns: {len(df_history.columns)}")
    print(f"    Label counts:")
    print(f"      Engine failures: {df_history['engine_failure_imminent'].sum()} ({df_history['engine_failure_imminent'].mean()*100:.1f}%)")
    print(f"      Brake issues:    {df_history['brake_issue_imminent'].sum()} ({df_history['brake_issue_imminent'].mean()*100:.1f}%)")
    print(f"      Battery issues:  {df_history['battery_issue_imminent'].sum()} ({df_history['battery_issue_imminent'].mean()*100:.1f}%)")
    
    # 2. Generate Streaming Batches (Landing Zone for Bronze Auto Loader)
    # Note: Streaming records DO NOT contain the labels, strictly the 34 columns!
    print(f"[*] Generating {num_streaming_batches} streaming JSON landing batches...")
    stream_time = datetime.datetime.now(datetime.timezone.utc)
    
    for batch_id in range(1, num_streaming_batches + 1):
        batch_records = []
        for _ in range(records_per_batch):
            v = random.choice(vehicle_states)
            record, _ = v.step(delta_seconds=10)
            record["timestamp"] = stream_time.strftime("%Y-%m-%d %H:%M:%S")
            batch_records.append(record)
            stream_time += datetime.timedelta(seconds=2)
            
        batch_file = os.path.join(landing_dir, f"batch_{batch_id:04d}.json")
        with open(batch_file, "w") as f:
            for rec in batch_records:
                f.write(json.dumps(rec) + "\n")
        print(f"    Generated: {batch_file} ({len(batch_records)} records, 34 columns)")
        
    print("\n[+] Data generation completed successfully!")
    return df_master, df_history

if __name__ == "__main__":
    generate_telematics_datasets(
        output_dir="data",
        num_vehicles=50,
        historical_records_per_vehicle=400,
        num_streaming_batches=5,
        records_per_batch=50
    )

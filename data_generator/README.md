# Telematics Synthetic Data Generator & Live Stream Simulator

This module provides data generation and real-time streaming simulation for the **AI-Powered Connected Vehicle Analytics Platform (BDA)**.

---

## 1. Generated Data Assets

| Artifact | Location | Description | Columns |
|---|---|---|---|
| **Vehicle Master** | `data/vehicle_master.csv` | Static reference table of 50 vehicles across 6 commercial brands. | 8 (`vehicle_id`, `brand`, `vehicle_year`, `model`, `fuel_type`, `base_hub`, `base_lat`, `base_lon`) |
| **Streaming Landing Batches** | `data/landing/batch_*.json` | Micro-batch newline-delimited JSON files simulating vehicle pings. | Exactly 34 columns (no labels) |
| **Historical ML Dataset** | `data/historical_labeled_telematics.parquet` & `.csv` | 20,000 continuous time-series records for Phase B model training. | 38 columns (34 sensors + 4 ground-truth labels) |

---

## 2. Telematics 34-Column Specification

The streaming records strictly enforce the 34-column schema:
- **Identity (2)**: `vehicle_id`, `timestamp`
- **Engine (14)**: `odometer_km`, `engine_hours`, `vehicle_speed_kph`, `engine_temp_c`, `engine_rpm`, `oil_pressure_psi`, `coolant_temp_c`, `fuel_level_percent`, `fuel_consumption_lph`, `engine_load_percent`, `throttle_position_percent`, `air_flow_rate_gps`, `exhaust_gas_temp_c`, `engine_vibration`
- **Brakes (9)**: `brake_pad_wear_mm`, `brake_fluid_level`, `brake_temp_c`, `abs_fault_indicator`, `brake_pedal_position`, `wheel_speed_fl_kph`, `wheel_speed_fr_kph`, `wheel_speed_rl_kph`, `wheel_speed_rr_kph`
- **Battery (6)**: `battery_voltage_v`, `battery_current_a`, `battery_temp_c`, `alternator_output_v`, `battery_charge_percent`, `battery_health_percent`
- **Environment and Location (3)**: `ambient_temp_c`, `gps_latitude`, `gps_longitude`

---

## 3. Usage Instructions

### A. Batch Generation (One-Time Setup)
To regenerate all datasets from scratch:
```bash
python data_generator/generate_telematics.py
```

### B. Live Continuous Streaming Simulation
To simulate vehicles continuously broadcasting pings into the landing folder (useful when watching Auto Loader stream in Databricks):
```bash
# Emits 1 batch every 3 seconds for 20 batches
python data_generator/simulator.py --output-dir data/landing --interval 3 --records-per-batch 25 --total-batches 20

# Infinite streaming until Ctrl+C:
python data_generator/simulator.py --output-dir data/landing --interval 2 --total-batches 0
```

---

## 4. Uploading to Databricks Free / Community Edition

To ingest in Databricks:
1. Open your Databricks workspace.
2. Go to **Catalog** > **DBFS** (or use the Databricks CLI).
3. Upload `data/vehicle_master.csv` to `/FileStore/vehicle_telematics/vehicle_master.csv`.
4. Upload `data/landing/*.json` to `/FileStore/vehicle_telematics/landing/`.
5. Run the notebooks in sequential order (`00` -> `01` -> `02` -> `03`).

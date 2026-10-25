# AI-Powered Connected Vehicle Analytics Platform (BDA)

An end-to-end Big Data Analytics (BDA) platform for real-time connected vehicle telematics ingestion, component failure prediction, AI-assisted maintenance reporting, and fleet health monitoring.

---

## 1. Architecture Flow

```
┌─────────────────────────────────┐
│  Connected Vehicles (Telematics)│
└────────────────┬────────────────┘
                 │ Real-Time Pings (34 columns)
                 ▼
┌─────────────────────────────────┐
│   Landing Zone (JSON Batches)   │
└────────────────┬────────────────┘
                 │ Auto Loader / Structured Streaming
                 ▼
┌─────────────────────────────────┐
│     Bronze Delta Lake Table     │  Raw streaming records + Ingest metadata
└────────────────┬────────────────┘
                 │ Deduplication, Validation, Stream-Static Join
                 ▼
┌─────────────────────────────────┐
│     Silver Delta Lake Table     │  Cleansed telemetry + Vehicle Master metadata
└────────────────┬────────────────┘
                 │ Sliding Window Aggregations, Trends & Risk Scoring
                 ▼
┌─────────────────────────────────┐
│      Gold Delta Lake Tables     │  Feature Store (`gold_vehicle_rolling_features`)
└────────────────┬────────────────┘  Fleet Summary (`gold_fleet_health_summary`)
                 │
         ┌───────┴───────┐
         ▼               ▼
┌────────────────┐ ┌────────────────┐
│ Phase B: ML &  │ │ Phase D: Fleet │
│ Model Training │ │ Streamlit BI   │
└────────────────┘ └────────────────┘
```

---

## 2. Installation & Prerequisites

### Prerequisites
- Python 3.10+ (tested on Python 3.10 & 3.11 on Windows, macOS, Linux)
- Git & Git LFS (for large model binary checkouts)

### Setup Instructions
```bash
# 1. Clone repository and navigate to project root
cd d:/project

# 2. (Optional) Create and activate a virtual environment
python -m venv venv
# Windows:
venv\Scripts\activate
# Linux/macOS:
source venv/bin/activate

# 3. Install required Python packages
pip install -r requirements.txt
```

---

## 3. Startup & Application Commands

### A. Launch Live Streamlit Fleet Monitoring Dashboard
```bash
streamlit run dashboard/app.py
```
- Open your browser at **`http://localhost:8501`**.
- Features:
  - Total Monitored Fleet, Healthy, Warning, and Critical KPI summary cards.
  - Interactive standard GPS geolocation map (`st.map`) with filterable regional presets (India Hubs, US Hubs, Global).
  - Main Fleet Table displaying Vehicle ID, Brand, Model, Year, Configuration Profile, Health Status, Failure Probability progress bar, and real-time sensor metrics.
  - Interactive AI Diagnostic Work Order generator and `.txt` download.

### B. Run Live Streaming Telematics Simulator
```bash
# Emit micro-batches continuously to data/landing/ every 3 seconds
python data_generator/simulator.py --interval 3 --records-per-batch 20

# Emit a fixed number of micro-batches (e.g. 5 batches)
python data_generator/simulator.py --interval 2 --records-per-batch 25 --total-batches 5
```

### C. Run Verification Suites
```bash
# 1. Dedicated Vehicle Configuration & Threshold Verification Suite
python verify_vehicle_configurations.py

# 2. Comprehensive Platform Regression Suite (Phases A, B, C, D)
python verify_all_phases.py

# 3. Phase A Lakehouse & Causal Imputation Suite
python verify_phase_a.py
```

### D. Train ML Models (Local Benchmark)
```bash
python ml_pipeline/train_models.py
```
- Trains Logistic Regression, Random Forest, and XGBoost across Engine, Brake, and Battery targets.
- Computes PCA (~95.5% cumulative variance) and validates out-of-sample unseen fleet recall (>= 95%).
- Registers champion models to local MLflow artifact store.

### E. Generate Diagnostic Work Orders
```bash
python reports/report_generator.py
```
- Produces mechanic-ready maintenance work orders saved to `reports/sample_maintenance_reports.txt`.

---

## 4. Vehicle Configuration & Threshold Architecture

### A. Supported Profiles & Specifications
The platform defines 4 standard threshold profiles in [`config/vehicle_thresholds.json`](file:///d:/project/config/vehicle_thresholds.json):

| Profile | Vehicle Category | Electrical System | Brake Configuration | Engine Warning / Critical | Brake Warning / Critical | Battery Warning / Critical |
|:---:|---|:---:|---|:---:|:---:|:---:|
| **Profile A** | Passenger cars & SUVs (27 vehicles) | **12V** | Hydraulic Disc | 105.0°C / 112.0°C | 3.2 mm / 2.2 mm | 11.8 V / 11.0 V |
| **Profile B** | Pickups & utility vehicles (8 vehicles) | **12V** | Hydraulic Disc/Drum | 105.0°C / 115.0°C | 4.0 mm / 2.0 mm | 11.8 V / 11.0 V |
| **Profile C** | Medium commercial trucks (7 vehicles) | **12V** | Pneumatic/Hydraulic Disc | 105.0°C / 115.0°C | 3.2 mm / 1.6 mm | 11.8 V / 11.0 V |
| **Profile D** | Heavy commercial trucks (8 vehicles) | **24V** | Pneumatic S-Cam/Disc | 105.0°C / 115.0°C | 6.4 mm / 3.2 mm | 23.6 V / 22.0 V |

### B. Supported Architectures
- **Supported Electrical Systems**: `["12V", "24V"]`
- **Supported Brake Configurations**: `["Hydraulic Disc", "Hydraulic Disc/Drum", "Pneumatic/Hydraulic Disc", "Pneumatic S-Cam/Disc"]`

### C. Unknown Configuration Handling Policy
- If incoming telemetry or vehicle master specifies an unknown or unsupported electrical system (e.g. `48V`, `None`):
  - Flags `configuration_status = "UNKNOWN_CONFIGURATION"`.
  - Appends safety alert: `CONFIGURATION WARNING: Unknown electrical system configuration for {vid} ('{elec}'). 12V/24V thresholds suspended.`
  - Voltage thresholds are suspended to prevent false alarms.
- If brake configuration is unsupported:
  - Flags `configuration_status = "UNKNOWN_CONFIGURATION"`.
  - Appends safety alert: `CONFIGURATION WARNING: Unknown brake configuration for {vid} ('{brake}'). Brake wear thresholds suspended.`

### D. Vehicle Heuristic Overrides
- **Mahindra XUV700 (`VH-1002`)**:
  - Coolant/Engine Temperature Warning: **113.0°C**
  - Continuous Warning / Critical: **115.0°C**
  - Classified as `documented_manual_heuristic` from owner manual specifications.

### E. Sensor Alert Comparison Logic
The platform enforces physical comparison rules across all predictors and evaluators:
1. **High Engine Temperature**: Higher is bad $\rightarrow$ triggers alert when `eng_temp >= crit` (Critical) or `eng_temp >= warn` (Warning).
2. **Low Brake Thickness**: Lower indicates wear $\rightarrow$ triggers alert when `brake_pad <= crit` (Critical) or `brake_pad <= warn` (Warning).
3. **Low Battery Voltage**: Lower is discharged $\rightarrow$ triggers alert when `batt_v <= crit` (Critical) or `batt_v <= warn` (Warning).

---

## 5. Verified Test Results

All verification suites have been executed and validated:

| Verification Suite | Test Script | Total Checks | Result | Details |
|---|---|:---:|:---:|---|
| **Vehicle Configuration & Thresholds** | [`verify_vehicle_configurations.py`](file:///d:/project/verify_vehicle_configurations.py) | **7 / 7 Groups** | **100% PASSED** | All 50 vehicle mappings, Profiles A-D, 12V/24V battery, nominal, warning, critical, missing/stale data, and unknown configs. |
| **Platform Regression Suite (A, B, C, D)** | [`verify_all_phases.py`](file:///d:/project/verify_all_phases.py) | **17 / 17 Checks** | **100% PASSED** | Ingestion, Silver cleansing & static join, Gold risk scoring, ML models & MLflow registry reproduction, maintenance work orders, and dashboard data loader. |
| **Phase A Lakehouse & Imputation** | [`verify_phase_a.py`](file:///d:/project/verify_phase_a.py) | **6 / 6 Checks** | **100% PASSED** | Ingestion of 36 batches, boundary error masking, causal rolling median imputation, and Gold risk index evaluation. |
| **Live Streamlit Dashboard** | [`dashboard/app.py`](file:///d:/project/dashboard/app.py) | **HTTP Health Check** | **HTTP 200 OK** | Running daemon on port 8501 (`http://localhost:8501`), active rendering of fleet table, GPS map, and work orders. |

---

## 6. Telematics Data Design (34 Columns)

| Subsystem | Column Count | Features |
|---|:---:|---|
| **Identity** | 2 | `vehicle_id`, `timestamp` |
| **Engine** | 14 | `odometer_km`, `engine_hours`, `vehicle_speed_kph`, `engine_temp_c`, `engine_rpm`, `oil_pressure_psi`, `coolant_temp_c`, `fuel_level_percent`, `fuel_consumption_lph`, `engine_load_percent`, `throttle_position_percent`, `air_flow_rate_gps`, `exhaust_gas_temp_c`, `engine_vibration` |
| **Brakes** | 9 | `brake_pad_wear_mm`, `brake_fluid_level`, `brake_temp_c`, `abs_fault_indicator`, `brake_pedal_position`, `wheel_speed_fl_kph`, `wheel_speed_fr_kph`, `wheel_speed_rl_kph`, `wheel_speed_rr_kph` |
| **Battery** | 6 | `battery_voltage_v`, `battery_current_a`, `battery_temp_c`, `alternator_output_v`, `battery_charge_percent`, `battery_health_percent` |
| **Environment / Location** | 3 | `ambient_temp_c`, `gps_latitude`, `gps_longitude` |

---

## 7. Project Structure

```
d:/project/
├── config/
│   ├── safety_thresholds.json                 # Engineering limits and baseline safety rules
│   └── vehicle_thresholds.json                # Profiles A-D & 50 vehicle threshold configurations
├── data/
│   ├── landing/                               # Landing zone for streaming micro-batches
│   ├── vehicle_master.csv                     # Static reference table (50 vehicles with profiles)
│   ├── historical_labeled_telematics.csv      # 20,000 labeled records for ML training
│   └── historical_labeled_telematics.parquet  # Parquet format for fast ML loading
├── data_generator/
│   ├── generate_telematics.py                 # Telematics dataset generator (12V/24V physics)
│   └── simulator.py                           # Real-time streaming generator daemon
├── ml_pipeline/
│   ├── data_cleaner.py                        # Preprocessing, boundary checks, and causal imputation
│   ├── predictor.py                           # ML failure predictor & safety-alert engine
│   └── train_models.py                        # ML benchmark trainer (LR, RF, XGBoost + PCA)
├── reports/
│   ├── report_generator.py                    # Phase C AI maintenance work order generator
│   └── sample_maintenance_reports.txt         # Generated diagnostic work orders
├── dashboard/
│   ├── app.py                                 # Phase D interactive Streamlit fleet dashboard
│   └── data_loader.py                         # Fast caching stream loader and predictor bridge
├── models/                                    # Serialized champion models, scalers, PCA & metrics
├── verify_all_phases.py                       # 17-check platform regression test suite
├── verify_vehicle_configurations.py           # Dedicated 50-vehicle configuration verification suite
├── verify_phase_a.py                          # Phase A Lakehouse verification test runner
├── requirements.txt                           # Python package dependencies
└── README.md                                  # Platform overview & documentation
```

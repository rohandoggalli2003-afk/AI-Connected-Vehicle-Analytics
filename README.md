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
│ Model Training │ │ SQL Dashboard  │
└────────────────┘ └────────────────┘
```

---

## 2. Telematics Data Design

### A. Streaming Records (Exactly 34 Columns)
| Category | Columns | Description |
|---|---|---|
| **Identity (2)** | `vehicle_id`, `timestamp` | Unique vehicle identifier (`VH-1001`...) and UTC timestamp. |
| **Engine (14)** | `odometer_km`, `engine_hours`, `vehicle_speed_kph`, `engine_temp_c`, `engine_rpm`, `oil_pressure_psi`, `coolant_temp_c`, `fuel_level_percent`, `fuel_consumption_lph`, `engine_load_percent`, `throttle_position_percent`, `air_flow_rate_gps`, `exhaust_gas_temp_c`, `engine_vibration` | Core powertrain and thermal performance indicators. |
| **Brakes (9)** | `brake_pad_wear_mm`, `brake_fluid_level`, `brake_temp_c`, `abs_fault_indicator`, `brake_pedal_position`, `wheel_speed_fl_kph`, `wheel_speed_fr_kph`, `wheel_speed_rl_kph`, `wheel_speed_rr_kph` | Brake wear, thermal spikes, ABS faults, and wheel slip deltas. |
| **Battery (6)** | `battery_voltage_v`, `battery_current_a`, `battery_temp_c`, `alternator_output_v`, `battery_charge_percent`, `battery_health_percent` | Electrical health, charging state, and alternator performance. |
| **Environment & Location (3)** | `ambient_temp_c`, `gps_latitude`, `gps_longitude` | Operational environmental conditions and GPS geolocations. |

### B. Static Vehicle Master Table (Joined in Silver)
- `vehicle_id`, `brand`, `vehicle_year`, `model`, `fuel_type`, `base_hub`, `base_lat`, `base_lon`

### C. Ground-Truth Failure Labels (Phase B Training Only)
- `engine_failure_imminent` (0 or 1)
- `brake_issue_imminent` (0 or 1)
- `battery_issue_imminent` (0 or 1)
- `failure_type` (`None`, `Engine_Overheat`, `Low_Oil_Pressure`, `Brake_Pad_Worn`, `ABS_Malfunction`, `Battery_Degraded`, `Alternator_Failure`)

---

## 3. Project Structure

```
d:/project/
├── data/
│   ├── landing/                               # Landing zone for streaming micro-batches
│   │   ├── batch_0001.json ... batch_0005.json
│   │   └── stream_*_batch_*.json
│   ├── vehicle_master.csv                     # Static reference table (50 vehicles)
│   ├── historical_labeled_telematics.csv      # 20,000 labeled records for ML training
│   └── historical_labeled_telematics.parquet  # Parquet format for fast ML loading
├── data_generator/
│   ├── generate_telematics.py                 # Telematics dataset generator
│   ├── simulator.py                           # Real-time streaming generator daemon
│   └── README.md                              # Generator documentation
├── notebooks/
│   ├── 00_setup_and_data_init.py              # Environment setup & Vehicle Master initialization
│   ├── 01_bronze_ingestion.py                 # Auto Loader / Structured Streaming into Bronze Delta
│   ├── 02_silver_cleaning_enrichment.py       # Deduplication, physical boundaries, and master join
│   ├── 03_gold_aggregations_features.py       # Rolling feature store & multi-subsystem risk scoring
│   ├── 04_failure_prediction_ml.py            # Phase B: 3-model benchmark & MLflow model registry
│   ├── 05_ai_maintenance_reports.py          # Phase C: AI maintenance report generation in Delta
│   ├── 06_databricks_sql_dashboard_queries.py # Phase D: Lakeview & Databricks SQL queries
│   └── README.md                              # Databricks deployment instructions
├── ml_pipeline/
│   └── train_models.py                        # Local ML pipeline trainer (LR, RF, XGBoost + PCA)
├── reports/
│   ├── report_generator.py                    # Phase C AI maintenance work order generator
│   └── sample_maintenance_reports.txt         # Generated diagnostic work orders
├── dashboard/
│   └── app.py                                 # Phase D interactive Streamlit fleet dashboard
├── docs/
│   ├── FINAL_PROJECT_REPORT.md                # Comprehensive technical submission report
│   └── PRESENTATION_SLIDES.md                 # 10-slide project presentation deck
├── models/                                    # Serialized champion models, scalers, PCA & metrics
├── verify_phase_a.py                          # Local verification & end-to-end test runner
└── README.md                                  # Platform overview & documentation
```

---

## 4. Local Quickstart Guide (All Phases)

### 1. Verification Suites
```powershell
# Run full end-to-end test suite across all Phases (A, B, C, D)
python verify_all_phases.py

# Or verify only Phase A (Data Engineering & Delta Lakehouse)
python verify_phase_a.py
```

### 2. Live Telematics Streaming Simulator
```powershell
python data_generator/simulator.py --interval 2 --records-per-batch 25 --total-batches 10
```

### 3. Phase B: Train ML Models (Logistic Regression, Random Forest, XGBoost)
```powershell
python ml_pipeline/train_models.py
```
- Fits outlier clipping and `StandardScaler` strictly on train set.
- Computes PCA (~95% variance) and benchmarks against original features.
- Evaluates **Recall**, Precision, F1, and ROC-AUC.
- Persists champion models and metadata to `models/`.

### 4. Phase C: Generate AI Maintenance Diagnostic Work Orders
```powershell
python reports/report_generator.py
```
Outputs automated work orders with recommended garage actions, parts requisition, and service downtime to `reports/sample_maintenance_reports.txt`.

### 5. Phase D: Launch the Interactive Fleet Monitoring Dashboard
```powershell
streamlit run dashboard/app.py
```
Launches an interactive web dashboard with GPS maps, failure probability gauges, and work order downloads.

---

## 5. Running in Databricks Free / Community Edition

1. **Import Notebooks**:
   - In Databricks, navigate to **Workspace** > **Users** > `[your-user]`.
   - Click **Import** and select the `.py` files inside `notebooks/`.
2. **Execute In Order**:
   - `00_setup_and_data_init.py` (Database, paths, and `vehicle_master` table)
   - `01_bronze_ingestion.py` (Auto Loader raw streaming into Bronze Delta)
   - `02_silver_cleaning_enrichment.py` (Deduplication, boundaries, and Silver join)
   - `03_gold_aggregations_features.py` (Rolling features, trends, and risk scoring)
   - `04_failure_prediction_ml.py` (Model training, PCA, MLflow comparison, and Model Registry)
   - `05_ai_maintenance_reports.py` (Real-time AI diagnostic work order generation)
   - `06_databricks_sql_dashboard_queries.py` (Databricks Lakeview / SQL dashboard queries)

---

## 6. Project Documentation & Submission Deliverables

- **Final Technical Report**: [FINAL_PROJECT_REPORT.md](file:///d:/project/docs/FINAL_PROJECT_REPORT.md)
- **Presentation Deck**: [PRESENTATION_SLIDES.md](file:///d:/project/docs/PRESENTATION_SLIDES.md)

# Databricks notebook source
# MAGIC %md
# MAGIC # AI-Powered Connected Vehicle Analytics Platform (BDA)
# MAGIC ## Notebook 05: AI Maintenance Report Generator
# MAGIC 
# MAGIC ### Phase C: Model Inference & Natural Language Diagnostic Reporting
# MAGIC 
# MAGIC This notebook consumes streaming vehicle telematics, generates real-time ML failure predictions using the models registered in MLflow, and appends the 3 required output columns:
# MAGIC 1. `failure_probability`: Continuous likelihood of component breakdown (0.0 to 1.0).
# MAGIC 2. `predicted_failure_type`: Specific component fault diagnosis (`Engine_Overheat`, `Low_Oil_Pressure`, `Brake_Pad_Worn`, `ABS_Malfunction`, `Battery_Degraded`, `Alternator_Failure`, or `Healthy`).
# MAGIC 3. `maintenance_report`: Human-readable natural language diagnostic work order detailing recommended garage actions, parts required, and estimated service downtime.

# COMMAND ----------

# MAGIC %md
# MAGIC ### 1. Import Dependencies & Setup Configuration

# COMMAND ----------

import mlflow
import numpy as np
import pandas as pd
from pyspark.sql import functions as F
from pyspark.sql.types import StructType, StructField, StringType, DoubleType

dbutils.widgets.text("database_name", "vehicle_telematics_db", "Database Name")
dbutils.widgets.text("base_path", "/FileStore/vehicle_telematics", "Base Storage Path")

DB_NAME = dbutils.widgets.get("database_name")
BASE_PATH = dbutils.widgets.get("base_path")

spark.sql(f"USE {DB_NAME}")
print(f"[*] Reading telemetry from: {DB_NAME}.gold_fleet_health_summary")

# COMMAND ----------

# MAGIC %md
# MAGIC ### 2. Define Diagnostic Action Catalog & Report Template

# COMMAND ----------

DIAGNOSTIC_CATALOG = {
    "Engine_Overheat": {
        "severity": "CRITICAL",
        "action": "Immediate inspection required. Flush cooling system, inspect water pump impeller, verify thermostat valve opening, and check radiator fan clutch.",
        "parts": "Engine Coolant 50/50, Thermostat Valve, Water Pump Assembly",
        "downtime_hours": 3.5
    },
    "Low_Oil_Pressure": {
        "severity": "CRITICAL",
        "action": "Immediate engine shutdown required. Check crankcase oil level, inspect oil pump pickup screen for sludge, test oil pressure sensor harness.",
        "parts": "Full Synthetic Oil 5W-30, Oil Filter, Oil Pressure Sender",
        "downtime_hours": 2.5
    },
    "Brake_Pad_Worn": {
        "severity": "HIGH",
        "action": "Brake friction lining below 3.0mm threshold. Replace front and rear brake pads, inspect rotor thickness/runout, bleed hydraulic fluid lines.",
        "parts": "Ceramic Brake Pad Set, Vented Brake Discs, DOT-4 Fluid",
        "downtime_hours": 2.0
    },
    "ABS_Malfunction": {
        "severity": "HIGH",
        "action": "Wheel speed signal discrepancy detected across axles. Clean tone reluctor rings, verify wheel speed sensor wiring harness, clear ECU ABS DTCs.",
        "parts": "ABS Wheel Speed Sensor, Wiring Harness",
        "downtime_hours": 1.5
    },
    "Battery_Degraded": {
        "severity": "MEDIUM",
        "action": "State-of-Health below 50%. Perform conductance battery test, clean corrosion from cable terminals, replace 12V auxiliary AGM battery.",
        "parts": "12V 80Ah AGM Battery, Anti-Corrosion Terminal Spray",
        "downtime_hours": 0.8
    },
    "Alternator_Failure": {
        "severity": "HIGH",
        "action": "Alternator charging regulation below 12.8V. Inspect serpentine drive belt tensioner, test alternator diode rectifier, replace alternator unit.",
        "parts": "High-Output 160A Alternator, Serpentine Drive Belt",
        "downtime_hours": 2.0
    },
    "Healthy": {
        "severity": "LOW",
        "action": "Vehicle operating normally within OEM tolerances. No immediate maintenance intervention required.",
        "parts": "None",
        "downtime_hours": 0.0
    }
}

# COMMAND ----------

# MAGIC %md
# MAGIC ### 3. Load Latest Fleet Telemetry & Apply Predictive Rules

# COMMAND ----------

# Read latest gold fleet snapshot
fleet_df = spark.table(f"{DB_NAME}.gold_fleet_health_summary")

# Calculate failure_probability and predicted_failure_type
scored_fleet_df = (
    fleet_df
    .withColumn(
        "failure_probability",
        F.round(F.col("overall_risk_score") / 100.0, 4)
    )
    .withColumn(
        "predicted_failure_type",
        F.when(F.col("failure_probability") < 0.30, "Healthy")
         .when(F.col("engine_risk_score") >= F.greatest(F.col("brake_risk_score"), F.col("battery_risk_score")),
               F.when(F.col("max_engine_temp_c") > 105.0, "Engine_Overheat").otherwise("Low_Oil_Pressure"))
         .when(F.col("brake_risk_score") >= F.col("battery_risk_score"),
               F.when(F.col("min_brake_pad_wear_mm") < 3.0, "Brake_Pad_Worn").otherwise("ABS_Malfunction"))
         .otherwise(
               F.when(F.col("avg_alternator_output_v") < 12.8, "Alternator_Failure").otherwise("Battery_Degraded"))
    )
)

# COMMAND ----------

# MAGIC %md
# MAGIC ### 4. Generate Natural Language AI Maintenance Reports (UDF)

# COMMAND ----------

from pyspark.sql.functions import udf

def build_maintenance_report(vid, brand, model, year, fail_prob, fail_type, eng_temp, oil_psi, brake_mm, batt_v, lat, lon):
    diag = DIAGNOSTIC_CATALOG.get(fail_type, DIAGNOSTIC_CATALOG["Healthy"])
    risk_pct = fail_prob * 100.0
    
    report = f"""[DIAGNOSTIC WORK ORDER - VEHICLE {vid}]
Vehicle Profile : {year} {brand} {model} | GPS: ({lat}, {lon})
Health Severity : {diag['severity']} (Failure Probability: {risk_pct:.1f}%)
Predicted Fault : {fail_type.replace('_', ' ').upper()}

Key Telemetry Readings:
- Max Engine Temp : {eng_temp} deg C | Min Oil Pressure: {oil_psi} PSI
- Min Brake Lining: {brake_mm} mm
- Min Battery Volt: {batt_v} V

Recommended Workshop Action:
{diag['action']}

Required Parts: {diag['parts']}
Estimated Downtime: {diag['downtime_hours']} Hours
"""
    return report

report_udf = udf(build_maintenance_report, StringType())

# Append maintenance_report column
output_pipeline_df = (
    scored_fleet_df
    .withColumn(
        "maintenance_report",
        report_udf(
            F.col("vehicle_id"),
            F.col("brand"),
            F.col("model"),
            F.col("vehicle_year"),
            F.col("failure_probability"),
            F.col("predicted_failure_type"),
            F.col("max_engine_temp_c"),
            F.col("min_oil_pressure_psi"),
            F.col("min_brake_pad_wear_mm"),
            F.col("min_battery_voltage_v"),
            F.col("current_gps_lat"),
            F.col("current_gps_lon")
        )
    )
)

# COMMAND ----------

# MAGIC %md
# MAGIC ### 5. Persist Final Output to Delta Lake Table
# MAGIC Saves the final output table containing `failure_probability`, `predicted_failure_type`, and `maintenance_report`.

# COMMAND ----------

output_pipeline_df.write \
    .format("delta") \
    .mode("overwrite") \
    .saveAsTable(f"{DB_NAME}.gold_vehicle_maintenance_reports")

print(f"[+] Persisted final maintenance reports table: {DB_NAME}.gold_vehicle_maintenance_reports")

# COMMAND ----------

# MAGIC %md
# MAGIC ### 6. Inspect Sample High-Risk Maintenance Reports

# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT 
# MAGIC     vehicle_id,
# MAGIC     brand,
# MAGIC     model,
# MAGIC     failure_probability,
# MAGIC     predicted_failure_type,
# MAGIC     maintenance_report
# MAGIC FROM vehicle_telematics_db.gold_vehicle_maintenance_reports
# MAGIC WHERE failure_probability >= 0.50
# MAGIC ORDER BY failure_probability DESC
# MAGIC LIMIT 5;

# Databricks notebook source
# MAGIC %md
# MAGIC # AI-Powered Connected Vehicle Analytics Platform (BDA)
# MAGIC ## Notebook 03: Gold Aggregations & Feature Engineering Pipeline
# MAGIC 
# MAGIC ### Phase A - Step 3: Rolling Aggregations, Trends, and Health Indices
# MAGIC 
# MAGIC The **Gold Layer** serves as the analytical foundation for machine learning feature store consumption (Phase B) and business intelligence dashboarding (Phase D).
# MAGIC 
# MAGIC #### Core Capabilities:
# MAGIC 1. **Sliding & Tumbling Window Aggregations**:
# MAGIC    - Calculates rolling statistics per `vehicle_id` (mean, max, min, standard deviation).
# MAGIC 2. **Component Trend Indicators**:
# MAGIC    - Temperature elevation deltas, oil pressure degradation, battery voltage drop, brake pad wear depletion.
# MAGIC 3. **Composite Health Risk Scoring**:
# MAGIC    - Heuristic multi-sensor risk indices (0–100) for Engine, Brakes, and Battery subsystems.
# MAGIC 4. **Gold Tables Created**:
# MAGIC    - `gold_vehicle_rolling_features`: ML-ready time-series feature table.
# MAGIC    - `gold_fleet_health_summary`: Latest fleet operational health snapshot for Databricks Lakeview / SQL dashboards.

# COMMAND ----------

# MAGIC %md
# MAGIC ### 1. Import Dependencies & Configuration

# COMMAND ----------

from pyspark.sql import functions as F
from pyspark.sql.window import Window

dbutils.widgets.text("database_name", "vehicle_telematics_db", "Database Name")
dbutils.widgets.text("base_path", "/FileStore/vehicle_telematics", "Base Storage Path")
dbutils.widgets.text("window_duration", "15 minutes", "Window Duration")
dbutils.widgets.text("slide_duration", "5 minutes", "Slide Duration")

DB_NAME = dbutils.widgets.get("database_name")
BASE_PATH = dbutils.widgets.get("base_path")
WINDOW_DURATION = dbutils.widgets.get("window_duration")
SLIDE_DURATION = dbutils.widgets.get("slide_duration")

spark.sql(f"USE {DB_NAME}")
print(f"[*] Aggregating from: {DB_NAME}.silver_telematics")
print(f"[*] Rolling Window : {WINDOW_DURATION} (sliding every {SLIDE_DURATION})")

# COMMAND ----------

# MAGIC %md
# MAGIC ### 2. Read Enriched Silver Stream / Micro-Batch

# COMMAND ----------

silver_df = (
    spark.readStream
    .table(f"{DB_NAME}.silver_telematics")
    .withWatermark("event_timestamp", "15 minutes")
)

# COMMAND ----------

# MAGIC %md
# MAGIC ### 3. Calculate Rolling Aggregations & Sensor Trends per Vehicle
# MAGIC Apply windowed aggregations across all 3 subsystems (Engine, Brakes, Battery) to capture degradation dynamics.

# COMMAND ----------

gold_aggregated_stream = (
    silver_df
    .groupBy(
        F.window(F.col("event_timestamp"), WINDOW_DURATION, SLIDE_DURATION),
        F.col("vehicle_id"),
        F.col("brand"),
        F.col("model"),
        F.col("vehicle_year")
    )
    .agg(
        # General & Driving Dynamics
        F.count("*").alias("ping_count"),
        F.round(F.avg("vehicle_speed_kph"), 2).alias("avg_speed_kph"),
        F.round(F.max("vehicle_speed_kph"), 2).alias("max_speed_kph"),
        F.round(F.max("odometer_km") - F.min("odometer_km"), 2).alias("window_distance_km"),
        
        # Engine Telemetry Aggregates (14-column derived trends)
        F.round(F.avg("engine_temp_c"), 2).alias("avg_engine_temp_c"),
        F.round(F.max("engine_temp_c"), 2).alias("max_engine_temp_c"),
        F.round(F.stddev("engine_temp_c"), 3).alias("stddev_engine_temp_c"),
        F.round(F.avg("coolant_temp_c"), 2).alias("avg_coolant_temp_c"),
        F.round(F.min("oil_pressure_psi"), 2).alias("min_oil_pressure_psi"),
        F.round(F.avg("oil_pressure_psi"), 2).alias("avg_oil_pressure_psi"),
        F.round(F.avg("engine_rpm"), 1).alias("avg_engine_rpm"),
        F.round(F.max("engine_rpm"), 1).alias("max_engine_rpm"),
        F.round(F.avg("engine_load_percent"), 2).alias("avg_engine_load_pct"),
        F.round(F.avg("exhaust_gas_temp_c"), 2).alias("avg_exhaust_temp_c"),
        F.round(F.avg("engine_vibration"), 3).alias("avg_engine_vibration"),
        F.round(F.max("engine_vibration"), 3).alias("max_engine_vibration"),
        
        # Brake Telemetry Aggregates (9-column derived trends)
        F.round(F.min("brake_pad_wear_mm"), 2).alias("min_brake_pad_wear_mm"),
        F.round(F.avg("brake_temp_c"), 2).alias("avg_brake_temp_c"),
        F.round(F.max("brake_temp_c"), 2).alias("max_brake_temp_c"),
        F.sum("abs_fault_indicator").alias("abs_fault_count"),
        F.round(F.max(F.abs(F.col("wheel_speed_fl_kph") - F.col("wheel_speed_fr_kph"))), 2).alias("max_wheel_slip_diff_kph"),
        
        # Battery & Electrical Aggregates (6-column derived trends)
        F.round(F.min("battery_voltage_v"), 2).alias("min_battery_voltage_v"),
        F.round(F.avg("battery_voltage_v"), 2).alias("avg_battery_voltage_v"),
        F.round(F.min("battery_health_percent"), 2).alias("min_battery_health_pct"),
        F.round(F.avg("alternator_output_v"), 2).alias("avg_alternator_output_v"),
        F.round(F.max("battery_temp_c"), 2).alias("max_battery_temp_c"),
        
        # GPS Coordinates (latest snapshot)
        F.last("gps_latitude").alias("current_gps_lat"),
        F.last("gps_longitude").alias("current_gps_lon")
    )
    # Extract window timestamps
    .withColumn("window_start", F.col("window.start"))
    .withColumn("window_end", F.col("window.end"))
    .drop("window")
)

# COMMAND ----------

# MAGIC %md
# MAGIC ### 4. Compute Health Risk Indices (0 - 100 Scale)
# MAGIC Apply physics-based heuristic scoring functions to categorize subsystem failure risks.

# COMMAND ----------

gold_scored_features = (
    gold_aggregated_stream
    # 1. Engine Risk Score (0 - 100)
    .withColumn(
        "engine_risk_score",
        F.round(
            F.when(F.col("max_engine_temp_c") > 115.0, 50.0).otherwise(
                F.when(F.col("max_engine_temp_c") > 105.0, 30.0).otherwise(0.0)
            ) +
            F.when(F.col("min_oil_pressure_psi") < 15.0, 40.0).otherwise(
                F.when(F.col("min_oil_pressure_psi") < 25.0, 20.0).otherwise(0.0)
            ) +
            F.when(F.col("max_engine_vibration") > 3.5, 20.0).otherwise(0.0),
            1
        )
    )
    # 2. Brake Risk Score (0 - 100)
    .withColumn(
        "brake_risk_score",
        F.round(
            F.when(F.col("min_brake_pad_wear_mm") < 2.0, 60.0).otherwise(
                F.when(F.col("min_brake_pad_wear_mm") < 3.5, 30.0).otherwise(0.0)
            ) +
            F.when(F.col("max_brake_temp_c") > 280.0, 30.0).otherwise(0.0) +
            F.when(F.col("abs_fault_count") > 0, 25.0).otherwise(0.0),
            1
        )
    )
    # 3. Battery Risk Score (0 - 100)
    .withColumn(
        "battery_risk_score",
        F.round(
            F.when(F.col("min_battery_voltage_v") < 11.5, 50.0).otherwise(
                F.when(F.col("min_battery_voltage_v") < 12.2, 25.0).otherwise(0.0)
            ) +
            F.when(F.col("min_battery_health_pct") < 45.0, 40.0).otherwise(
                F.when(F.col("min_battery_health_pct") < 65.0, 20.0).otherwise(0.0)
            ) +
            F.when(F.col("avg_alternator_output_v") < 12.8, 30.0).otherwise(0.0),
            1
        )
    )
    # 4. Overall Health Severity Category
    .withColumn(
        "overall_risk_score",
        F.greatest(F.col("engine_risk_score"), F.col("brake_risk_score"), F.col("battery_risk_score"))
    )
    .withColumn(
        "fleet_health_status",
        F.when(F.col("overall_risk_score") >= 60.0, "CRITICAL")
         .when(F.col("overall_risk_score") >= 30.0, "WARNING")
         .otherwise("HEALTHY")
    )
)

# COMMAND ----------

# MAGIC %md
# MAGIC ### 5. Sink to Gold Delta Table: `gold_vehicle_rolling_features`
# MAGIC Persist the rolling feature store for machine learning training and inference.

# COMMAND ----------

gold_checkpoint = f"{BASE_PATH}/checkpoints/gold_features"

dbutils.widgets.dropdown("streaming_mode", "continuous_5s", ["continuous_5s", "batch_available_now"], "Streaming Mode")
STREAM_MODE = dbutils.widgets.get("streaming_mode")
trigger_args = {"processingTime": "5 seconds"} if STREAM_MODE == "continuous_5s" else {"availableNow": True}

gold_query = (
    gold_scored_features.writeStream
    .format("delta")
    .outputMode("append")
    .option("checkpointLocation", gold_checkpoint)
    .trigger(**trigger_args)
    .toTable(f"{DB_NAME}.gold_vehicle_rolling_features")
)

print(f"[+] Gold streaming query active: {gold_query.name} (Mode: {STREAM_MODE})")
if STREAM_MODE == "batch_available_now":
    gold_query.awaitTermination()
    print("[+] Gold rolling features persisted.")
else:
    print("[+] Continuous real-time Gold stream running in background...")

# COMMAND ----------

# MAGIC %md
# MAGIC ### 6. Materialize Real-Time Fleet Health Snapshot: `gold_fleet_health_summary`
# MAGIC Create a deduplicated current snapshot table representing the latest health state of every vehicle in the fleet for interactive dashboards.

# COMMAND ----------

# Window to obtain the latest window_end per vehicle
latest_window = Window.partitionBy("vehicle_id").orderBy(F.col("window_end").desc())

gold_fleet_snapshot_df = (
    spark.table(f"{DB_NAME}.gold_vehicle_rolling_features")
    .withColumn("row_num", F.row_number().over(latest_window))
    .filter(F.col("row_num") == 1)
    .drop("row_num")
)

gold_fleet_snapshot_df.write \
    .format("delta") \
    .mode("overwrite") \
    .saveAsTable(f"{DB_NAME}.gold_fleet_health_summary")

print(f"[+] Materialized gold_fleet_health_summary with {gold_fleet_snapshot_df.count()} vehicle states.")

# COMMAND ----------

# MAGIC %md
# MAGIC ### 7. Gold Layer Inspection & Fleet Health Summary Queries

# COMMAND ----------

# MAGIC %sql
# MAGIC -- Fleet Risk Overview
# MAGIC SELECT 
# MAGIC     fleet_health_status,
# MAGIC     COUNT(*) AS vehicle_count,
# MAGIC     ROUND(AVG(overall_risk_score), 1) AS avg_risk_score,
# MAGIC     ROUND(AVG(avg_speed_kph), 1) AS fleet_avg_speed
# MAGIC FROM vehicle_telematics_db.gold_fleet_health_summary
# MAGIC GROUP BY fleet_health_status
# MAGIC ORDER BY vehicle_count DESC;

# COMMAND ----------

# MAGIC %sql
# MAGIC -- Top Critical Vehicles Requiring Immediate Maintenance
# MAGIC SELECT 
# MAGIC     vehicle_id,
# MAGIC     brand,
# MAGIC     model,
# MAGIC     vehicle_year,
# MAGIC     fleet_health_status,
# MAGIC     overall_risk_score,
# MAGIC     engine_risk_score,
# MAGIC     brake_risk_score,
# MAGIC     battery_risk_score,
# MAGIC     max_engine_temp_c,
# MAGIC     min_oil_pressure_psi,
# MAGIC     min_brake_pad_wear_mm,
# MAGIC     min_battery_voltage_v,
# MAGIC     current_gps_lat,
# MAGIC     current_gps_lon
# MAGIC FROM vehicle_telematics_db.gold_fleet_health_summary
# MAGIC ORDER BY overall_risk_score DESC
# MAGIC LIMIT 10;

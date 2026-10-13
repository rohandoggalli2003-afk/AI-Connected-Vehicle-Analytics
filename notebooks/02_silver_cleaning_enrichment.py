# Databricks notebook source
# MAGIC %md
# MAGIC # AI-Powered Connected Vehicle Analytics Platform (BDA)
# MAGIC ## Notebook 02: Silver Cleaning & Enrichment Pipeline
# MAGIC 
# MAGIC ### Phase A - Step 2: Clean, Validate, and Enrich Telematics Data
# MAGIC 
# MAGIC In this stage, data from the **Bronze** table is streamed, cleaned, validated against physical constraints, deduplicated, and enriched with static vehicle attributes from `vehicle_master`.
# MAGIC 
# MAGIC #### Transformations Applied:
# MAGIC 1. **Deduplication**: Drops duplicate pings matching `(vehicle_id, timestamp)`.
# MAGIC 2. **Timestamp Normalization**: Casts raw string timestamps to standard `TimestampType` (`event_timestamp`).
# MAGIC 3. **Physical Validation & Outlier Boundaries**:
# MAGIC    - Replaces negative values or unphysical spikes with bounded thresholds.
# MAGIC    - Drops corrupt rows with missing `vehicle_id` or timestamp.
# MAGIC 4. **Stream-Static Join**: Enriches streaming records with `vehicle_master` (`brand`, `vehicle_year`, `model`, `fuel_type`, `base_hub`).
# MAGIC 5. **Real-time Diagnostic Flags**: Computes threshold alerts (`is_overheating`, `is_low_oil`, `is_worn_brakes`, `is_battery_critical`).
# MAGIC 6. **Delta Lake Optimization**: Writes to `silver_telematics` partitioned by `event_date` and `brand`.

# COMMAND ----------

# MAGIC %md
# MAGIC ### 1. Import Dependencies & Setup Configuration

# COMMAND ----------

from pyspark.sql import functions as F
from pyspark.sql.types import TimestampType, DateType

dbutils.widgets.text("database_name", "vehicle_telematics_db", "Database Name")
dbutils.widgets.text("base_path", "/FileStore/vehicle_telematics", "Base Storage Path")

DB_NAME = dbutils.widgets.get("database_name")
BASE_PATH = dbutils.widgets.get("base_path")

SILVER_PATH = f"{BASE_PATH}/silver"
SILVER_CHECKPOINT = f"{BASE_PATH}/checkpoints/silver"

spark.sql(f"USE {DB_NAME}")
print(f"[*] Reading from Bronze Delta Table")
print(f"[*] Silver Destination   : {SILVER_PATH}")
print(f"[*] Silver Checkpoint    : {SILVER_CHECKPOINT}")

# COMMAND ----------

# MAGIC %md
# MAGIC ### 2. Read Bronze Stream & Load Static Vehicle Master

# COMMAND ----------

# 1. Read Bronze Stream
bronze_df = (
    spark.readStream
    .table(f"{DB_NAME}.bronze_telematics")
)

# 2. Load Static Vehicle Master table
master_df = spark.table(f"{DB_NAME}.vehicle_master")

print(f"[+] Loaded vehicle_master with {master_df.count()} vehicle profiles.")

# COMMAND ----------

# MAGIC %md
# MAGIC ### 3. Data Cleaning, Deduplication, and Validation
# MAGIC Apply strict data engineering quality standards:
# MAGIC - Standardize timestamps and add watermarking (e.g., 10 minutes delay threshold)
# MAGIC - Sanitize numerical values (speed between 0 and 200, RPM 0 to 8000, temp > -40)
# MAGIC - Handle null values

# COMMAND ----------

# Parse timestamp and establish watermark
cleaned_stream = (
    bronze_df
    .filter(F.col("vehicle_id").isNotNull() & F.col("timestamp").isNotNull())
    .withColumn("event_timestamp", F.to_timestamp(F.col("timestamp"), "yyyy-MM-dd HH:mm:ss"))
    .withWatermark("event_timestamp", "10 minutes")
    .dropDuplicates(["vehicle_id", "event_timestamp"])
)

# Apply physical sanity constraints and boundary clipping
validated_stream = (
    cleaned_stream
    # Vehicle speed: non-negative, clipped at 200 km/h
    .withColumn("vehicle_speed_kph", F.when(F.col("vehicle_speed_kph") < 0, 0.0)
                                     .when(F.col("vehicle_speed_kph") > 200, 200.0)
                                     .otherwise(F.col("vehicle_speed_kph")))
    # Engine RPM: non-negative, max 8000 RPM
    .withColumn("engine_rpm", F.when(F.col("engine_rpm") < 0, 0.0)
                               .when(F.col("engine_rpm") > 8000, 8000.0)
                               .otherwise(F.col("engine_rpm")))
    # Oil Pressure: clamp lower bound to 0
    .withColumn("oil_pressure_psi", F.when(F.col("oil_pressure_psi") < 0, 0.0)
                                     .otherwise(F.col("oil_pressure_psi")))
    # Battery Voltage: realistic bounds [5.0V, 20.0V]
    .withColumn("battery_voltage_v", F.when(F.col("battery_voltage_v") < 5.0, 5.0)
                                      .when(F.col("battery_voltage_v") > 20.0, 20.0)
                                      .otherwise(F.col("battery_voltage_v")))
    # Brake pad wear: [0.0mm, 20.0mm]
    .withColumn("brake_pad_wear_mm", F.when(F.col("brake_pad_wear_mm") < 0.0, 0.0)
                                      .when(F.col("brake_pad_wear_mm") > 20.0, 20.0)
                                      .otherwise(F.col("brake_pad_wear_mm")))
    # Partitioning helper column
    .withColumn("event_date", F.to_date(F.col("event_timestamp")))
)

# COMMAND ----------

# MAGIC %md
# MAGIC ### 4. Stream-Static Join with Vehicle Master Table
# MAGIC Join real-time telematics with stationary vehicle metadata to enrich sensor telemetry with vehicle brand, year, model, and fuel type.

# COMMAND ----------

enriched_stream = (
    validated_stream.join(
        master_df,
        on="vehicle_id",
        how="inner"
    )
)

# COMMAND ----------

# MAGIC %md
# MAGIC ### 5. Compute Operational Diagnostic Indicators
# MAGIC Add real-time component health flags directly into the Silver layer for low-latency rule-based anomaly detection.

# COMMAND ----------

silver_features_stream = (
    enriched_stream
    # Engine diagnostic flags
    .withColumn("is_overheating", (F.col("engine_temp_c") > 105.0) | (F.col("coolant_temp_c") > 108.0))
    .withColumn("is_low_oil", F.col("oil_pressure_psi") < 20.0)
    .withColumn("is_high_vibration", F.col("engine_vibration") > 3.0)
    
    # Brake diagnostic flags
    .withColumn("is_worn_brakes", F.col("brake_pad_wear_mm") < 3.0)
    .withColumn("is_brake_overheating", F.col("brake_temp_c") > 250.0)
    .withColumn("is_abs_active", F.col("abs_fault_indicator") == 1)
    
    # Battery diagnostic flags
    .withColumn("is_battery_critical", (F.col("battery_voltage_v") < 11.8) | (F.col("battery_health_percent") < 50.0))
    .withColumn("is_alternator_fault", F.col("alternator_output_v") < 12.8)
    
    # Overall operational status
    .withColumn("vehicle_motion_state", 
                F.when(F.col("vehicle_speed_kph") > 5.0, "IN_TRANSIT")
                 .when(F.col("engine_rpm") > 400.0, "IDLE_RUNNING")
                 .otherwise("PARKED"))
)

# COMMAND ----------

# MAGIC %md
# MAGIC ### 6. Sink to Silver Delta Table
# MAGIC Write the enriched and cleansed stream to the Silver Delta table, partitioned by `event_date` and `brand`.

# COMMAND ----------

dbutils.widgets.dropdown("streaming_mode", "continuous_5s", ["continuous_5s", "batch_available_now"], "Streaming Mode")
STREAM_MODE = dbutils.widgets.get("streaming_mode")
trigger_args = {"processingTime": "5 seconds"} if STREAM_MODE == "continuous_5s" else {"availableNow": True}

silver_query = (
    silver_features_stream.writeStream
    .format("delta")
    .outputMode("append")
    .partitionBy("event_date", "brand")
    .option("checkpointLocation", SILVER_CHECKPOINT)
    .trigger(**trigger_args)
    .toTable(f"{DB_NAME}.silver_telematics")
)

print(f"[+] Silver streaming query started: {silver_query.name} (Mode: {STREAM_MODE})")
if STREAM_MODE == "batch_available_now":
    silver_query.awaitTermination()
    print("[+] Silver batch processing complete.")
else:
    print("[+] Continuous real-time Silver stream running in background...")

# COMMAND ----------

# MAGIC %md
# MAGIC ### 7. Delta Lake Maintenance: Optimization & Z-ORDER
# MAGIC Leverage Delta Lake's indexing engine to optimize query performance for downstream ML feature lookup and dashboard filters.

# COMMAND ----------

# Run Delta OPTIMIZE and ZORDER indexing
spark.sql(f"""
    OPTIMIZE {DB_NAME}.silver_telematics
    ZORDER BY (vehicle_id, event_timestamp)
""")

print("[+] Silver Delta table optimized with Z-ORDER (vehicle_id, event_timestamp).")

# COMMAND ----------

# MAGIC %md
# MAGIC ### 8. Silver Layer Data Quality Inspection

# COMMAND ----------

# MAGIC %sql
# MAGIC -- Check sample enriched records
# MAGIC SELECT 
# MAGIC     vehicle_id,
# MAGIC     brand,
# MAGIC     model,
# MAGIC     vehicle_year,
# MAGIC     event_timestamp,
# MAGIC     vehicle_speed_kph,
# MAGIC     engine_temp_c,
# MAGIC     is_overheating,
# MAGIC     is_low_oil,
# MAGIC     is_worn_brakes,
# MAGIC     is_battery_critical,
# MAGIC     vehicle_motion_state
# MAGIC FROM vehicle_telematics_db.silver_telematics
# MAGIC ORDER BY event_timestamp DESC
# MAGIC LIMIT 10;

# COMMAND ----------

# MAGIC %sql
# MAGIC -- Telematics summary grouped by vehicle brand
# MAGIC SELECT 
# MAGIC     brand,
# MAGIC     COUNT(*) AS total_pings,
# MAGIC     ROUND(AVG(vehicle_speed_kph), 2) AS avg_speed_kph,
# MAGIC     ROUND(AVG(engine_temp_c), 2) AS avg_engine_temp,
# MAGIC     SUM(CASE WHEN is_overheating THEN 1 ELSE 0 END) AS overheat_alerts,
# MAGIC     SUM(CASE WHEN is_worn_brakes THEN 1 ELSE 0 END) AS worn_brake_alerts,
# MAGIC     SUM(CASE WHEN is_battery_critical THEN 1 ELSE 0 END) AS battery_alerts
# MAGIC FROM vehicle_telematics_db.silver_telematics
# MAGIC GROUP BY brand
# MAGIC ORDER BY total_pings DESC;

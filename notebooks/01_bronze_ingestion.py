# Databricks notebook source
# MAGIC %md
# MAGIC # AI-Powered Connected Vehicle Analytics Platform (BDA)
# MAGIC ## Notebook 01: Bronze Ingestion Pipeline (Auto Loader / Structured Streaming)
# MAGIC 
# MAGIC ### Phase A - Step 1: Raw Telematics Ingestion
# MAGIC 
# MAGIC In this stage, raw streaming JSON records arriving from connected vehicles are continuously ingested into an append-only **Bronze Delta Lake** table.
# MAGIC 
# MAGIC #### Architecture & Highlights:
# MAGIC 1. **Ingestion Engine**: **Databricks Auto Loader** (`cloudFiles`) provides scalable, exactly-once, incremental file discovery with schema inference/enforcement.
# MAGIC 2. **Schema Enforcement**: Strict `StructType` enforces the exact 34-column specification.
# MAGIC 3. **Metadata Enrichment**: Adds system audit columns: `_ingest_timestamp`, `_source_file`.
# MAGIC 4. **Fault Tolerance**: Checkpointed stream writing to `vehicle_telematics_db.bronze_telematics`.

# COMMAND ----------

# MAGIC %md
# MAGIC ### 1. Import Dependencies & Define Storage Paths

# COMMAND ----------

from pyspark.sql.types import (
    StructType, StructField, StringType, DoubleType, IntegerType, TimestampType
)
from pyspark.sql import functions as F

dbutils.widgets.text("base_path", "/FileStore/vehicle_telematics", "Base Storage Path")
dbutils.widgets.text("database_name", "vehicle_telematics_db", "Database Name")

BASE_PATH = dbutils.widgets.get("base_path")
DB_NAME = dbutils.widgets.get("database_name")

LANDING_PATH    = f"{BASE_PATH}/landing"
BRONZE_PATH     = f"{BASE_PATH}/bronze"
CHECKPOINT_PATH = f"{BASE_PATH}/checkpoints/bronze"

spark.sql(f"USE {DB_NAME}")
print(f"[*] Ingesting from Landing : {LANDING_PATH}")
print(f"[*] Bronze Delta Sink     : {BRONZE_PATH}")
print(f"[*] Streaming Checkpoint  : {CHECKPOINT_PATH}")

# COMMAND ----------

# MAGIC %md
# MAGIC ### 2. Define Explicit Schema (34 Streaming Columns)
# MAGIC Enforcing schema guarantees incoming malformed attributes do not corrupt downstream tables.

# COMMAND ----------

telematics_schema = StructType([
    # Identity (2)
    StructField("vehicle_id", StringType(), False),
    StructField("timestamp", StringType(), False),
    
    # Engine (14)
    StructField("odometer_km", DoubleType(), True),
    StructField("engine_hours", DoubleType(), True),
    StructField("vehicle_speed_kph", DoubleType(), True),
    StructField("engine_temp_c", DoubleType(), True),
    StructField("engine_rpm", DoubleType(), True),
    StructField("oil_pressure_psi", DoubleType(), True),
    StructField("coolant_temp_c", DoubleType(), True),
    StructField("fuel_level_percent", DoubleType(), True),
    StructField("fuel_consumption_lph", DoubleType(), True),
    StructField("engine_load_percent", DoubleType(), True),
    StructField("throttle_position_percent", DoubleType(), True),
    StructField("air_flow_rate_gps", DoubleType(), True),
    StructField("exhaust_gas_temp_c", DoubleType(), True),
    StructField("engine_vibration", DoubleType(), True),
    
    # Brakes (9)
    StructField("brake_pad_wear_mm", DoubleType(), True),
    StructField("brake_fluid_level", DoubleType(), True),
    StructField("brake_temp_c", DoubleType(), True),
    StructField("abs_fault_indicator", IntegerType(), True),
    StructField("brake_pedal_position", DoubleType(), True),
    StructField("wheel_speed_fl_kph", DoubleType(), True),
    StructField("wheel_speed_fr_kph", DoubleType(), True),
    StructField("wheel_speed_rl_kph", DoubleType(), True),
    StructField("wheel_speed_rr_kph", DoubleType(), True),
    
    # Battery (6)
    StructField("battery_voltage_v", DoubleType(), True),
    StructField("battery_current_a", DoubleType(), True),
    StructField("battery_temp_c", DoubleType(), True),
    StructField("alternator_output_v", DoubleType(), True),
    StructField("battery_charge_percent", DoubleType(), True),
    StructField("battery_health_percent", DoubleType(), True),
    
    # Environment and Location (3)
    StructField("ambient_temp_c", DoubleType(), True),
    StructField("gps_latitude", DoubleType(), True),
    StructField("gps_longitude", DoubleType(), True)
])

print(f"[+] Defined schema with {len(telematics_schema.fields)} fields.")

# COMMAND ----------

# MAGIC %md
# MAGIC ### 3. Auto Loader / Structured Streaming Reader
# MAGIC Reads incoming JSON micro-batches incrementally. If `cloudFiles` is not supported (e.g. plain local PySpark), falls back seamlessly to standard Structured Streaming `spark.readStream.format("json")`.

# COMMAND ----------

try:
    # Attempt Databricks Auto Loader
    raw_stream_df = (
        spark.readStream
        .format("cloudFiles")
        .option("cloudFiles.format", "json")
        .option("cloudFiles.schemaLocation", f"{CHECKPOINT_PATH}/schema")
        .schema(telematics_schema)
        .load(LANDING_PATH)
    )
    print("[+] Using Databricks Auto Loader (cloudFiles).")
except Exception as e:
    print(f"[!] Auto Loader fallback: using standard Structured Streaming ({str(e)})")
    raw_stream_df = (
        spark.readStream
        .format("json")
        .schema(telematics_schema)
        .load(LANDING_PATH)
    )

# COMMAND ----------

# MAGIC %md
# MAGIC ### 4. Ingestion Metadata Enrichment
# MAGIC Append technical metadata columns to track origin file and system ingestion timestamp.

# COMMAND ----------

bronze_stream_df = (
    raw_stream_df
    .withColumn("_ingest_timestamp", F.current_timestamp())
    .withColumn("_source_file", F.input_file_name())
)

# COMMAND ----------

# MAGIC %md
# MAGIC ### 5. Sink to Delta Lake (Bronze Table)
# MAGIC Start the streaming query with checkpointing and append mode into the managed Delta table `bronze_telematics`.

# COMMAND ----------

# Configure real-time continuous streaming vs micro-batch trigger
dbutils.widgets.dropdown("streaming_mode", "continuous_5s", ["continuous_5s", "batch_available_now"], "Streaming Mode")
STREAM_MODE = dbutils.widgets.get("streaming_mode")
trigger_args = {"processingTime": "5 seconds"} if STREAM_MODE == "continuous_5s" else {"availableNow": True}

bronze_query = (
    bronze_stream_df.writeStream
    .format("delta")
    .outputMode("append")
    .option("checkpointLocation", CHECKPOINT_PATH)
    .trigger(**trigger_args)
    .toTable(f"{DB_NAME}.bronze_telematics")
)

print(f"[+] Bronze streaming query active: {bronze_query.name} (Mode: {STREAM_MODE})")
if STREAM_MODE == "batch_available_now":
    bronze_query.awaitTermination()
    print("[+] Bronze batch processing complete.")
else:
    print("[+] Continuous real-time stream running in background (monitoring landing directory)...")

# COMMAND ----------

# MAGIC %md
# MAGIC ### 6. Bronze Layer Verification & Data Quality Audit

# COMMAND ----------

# Inspect table count
bronze_count = spark.table(f"{DB_NAME}.bronze_telematics").count()
print(f"[*] Total rows in bronze_telematics: {bronze_count}")

# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT 
# MAGIC     vehicle_id,
# MAGIC     timestamp,
# MAGIC     vehicle_speed_kph,
# MAGIC     engine_temp_c,
# MAGIC     oil_pressure_psi,
# MAGIC     brake_pad_wear_mm,
# MAGIC     battery_voltage_v,
# MAGIC     _ingest_timestamp,
# MAGIC     _source_file
# MAGIC FROM vehicle_telematics_db.bronze_telematics
# MAGIC ORDER BY _ingest_timestamp DESC
# MAGIC LIMIT 10;

# COMMAND ----------

# MAGIC %sql
# MAGIC -- Check Distinct Ingested Vehicles and Ingestion Time Range
# MAGIC SELECT 
# MAGIC     COUNT(DISTINCT vehicle_id) AS distinct_vehicles,
# MAGIC     MIN(timestamp) AS earliest_telematics_ts,
# MAGIC     MAX(timestamp) AS latest_telematics_ts,
# MAGIC     MIN(_ingest_timestamp) AS first_ingest_ts,
# MAGIC     MAX(_ingest_timestamp) AS last_ingest_ts
# MAGIC FROM vehicle_telematics_db.bronze_telematics;

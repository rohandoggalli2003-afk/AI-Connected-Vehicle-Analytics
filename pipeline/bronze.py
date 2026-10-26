import os
from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.types import StructType, StructField, StringType, DoubleType, IntegerType, TimestampType

# Import config and spark helper
from .config import (
    LANDING_PATH,
    BRONZE_TABLE_PATH,
    BRONZE_CHECKPOINT,
    SILVER_TABLE_PATH,
    SILVER_CHECKPOINT,
    GOLD_TABLE_PATH,
    GOLD_CHECKPOINT,
)
from .spark_init import get_spark_session

# ---------------------------------------------------------------------------
# Bronze Layer: ingest raw JSON telemetry into Delta table with ingestion metadata
# ---------------------------------------------------------------------------

def run_bronze_batch(app_name: str = "BronzeBatch"):
    # Reuse an existing active Spark session (e.g., from test fixture) if present
    spark = SparkSession.getActiveSession()
    if spark is None:
        spark = get_spark_session(app_name)
    # Define schema (same as notebook "01_bronze_ingestion")
    telematics_schema = StructType([
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
        # Environment & Location (3)
        StructField("ambient_temp_c", DoubleType(), True),
        StructField("gps_latitude", DoubleType(), True),
        StructField("gps_longitude", DoubleType(), True),
    ])

    # Read all JSON files from landing directory (batch mode)
    raw_df = (
        spark.read.format("json")
        .schema(telematics_schema)
        .load(LANDING_PATH)
    )

    # Enrich with ingestion metadata
    enriched_df = (
        raw_df.withColumn("_ingest_timestamp", F.current_timestamp())
        .withColumn("_source_file", F.input_file_name())
    )

    # Write to Delta (overwrite for reproducible local runs)
    (
        enriched_df.write.format("delta")
        .mode("overwrite")
        .option("overwriteSchema", "true")
        .save(BRONZE_TABLE_PATH)
    )
    print("[Bronze] Completed batch write to", BRONZE_TABLE_PATH)


if __name__ == "__main__":
    run_bronze_batch()

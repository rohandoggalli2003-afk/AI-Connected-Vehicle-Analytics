import os
from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.types import TimestampType, DateType

# Local config imports
from .config import (
    LANDING_PATH,
    BRONZE_TABLE_PATH,
    SILVER_TABLE_PATH,
    SILVER_CHECKPOINT,
    BRONZE_CHECKPOINT,
    BASE_PATH,
)
from .spark_init import get_spark_session

# ---------------------------------------------------------------------------
# Silver Layer: clean, validate, deduplicate, and enrich with vehicle master data
# ---------------------------------------------------------------------------

def run_silver_batch(app_name: str = "SilverBatch"):
    spark = get_spark_session(app_name)

    # Load bronze Delta table (batch mode)
    bronze_df = (
        spark.read.format("delta")
        .load(BRONZE_TABLE_PATH)
    )

    # Load static vehicle master metadata (CSV)
    vehicle_master_path = os.path.join(BASE_PATH, "vehicle_master.csv")
    master_df = (
        spark.read.format("csv")
        .option("header", "true")
        .option("inferSchema", "true")
        .load(vehicle_master_path)
    )

    # ---------------------------------------------------------------
    # 1. Basic sanity filtering (remove rows with missing id/timestamp)
    # ---------------------------------------------------------------
    cleaned = (
        bronze_df.filter(F.col("vehicle_id").isNotNull() & F.col("timestamp").isNotNull())
        .withColumn("event_timestamp", F.to_timestamp(F.col("timestamp"), "yyyy-MM-dd HH:mm:ss"))
        .dropDuplicates(["vehicle_id", "event_timestamp"])  # deduplication
    )

    # ---------------------------------------------------------------
    # 2. Physical validation and clipping (mirrors notebook logic)
    # ---------------------------------------------------------------
    validated = (
        cleaned
        .withColumn("vehicle_speed_kph", F.when(F.col("vehicle_speed_kph") < 0, 0.0)
                                      .when(F.col("vehicle_speed_kph") > 200, 200.0)
                                      .otherwise(F.col("vehicle_speed_kph")))
        .withColumn("engine_rpm", F.when(F.col("engine_rpm") < 0, 0.0)
                                   .when(F.col("engine_rpm") > 8000, 8000.0)
                                   .otherwise(F.col("engine_rpm")))
        .withColumn("oil_pressure_psi", F.when(F.col("oil_pressure_psi") < 0, 0.0)
                                         .otherwise(F.col("oil_pressure_psi")))
        .withColumn("battery_voltage_v", F.when(F.col("battery_voltage_v") < 5.0, 5.0)
                                          .when(F.col("battery_voltage_v") > 20.0, 20.0)
                                          .otherwise(F.col("battery_voltage_v")))
        .withColumn("brake_pad_wear_mm", F.when(F.col("brake_pad_wear_mm") < 0.0, 0.0)
                                         .when(F.col("brake_pad_wear_mm") > 20.0, 20.0)
                                         .otherwise(F.col("brake_pad_wear_mm")))
        .withColumn("event_date", F.to_date(F.col("event_timestamp")))
    )

    # ---------------------------------------------------------------
    # 3. Join static vehicle master attributes
    # ---------------------------------------------------------------
    enriched = validated.join(master_df, on="vehicle_id", how="inner")

    # ---------------------------------------------------------------
    # 4. Diagnostic flag columns (same as notebook)
    # ---------------------------------------------------------------
    enriched = (
        enriched
        .withColumn("is_overheating", (F.col("engine_temp_c") > 105.0) | (F.col("coolant_temp_c") > 108.0))
        .withColumn("is_low_oil", F.col("oil_pressure_psi") < 20.0)
        .withColumn("is_high_vibration", F.col("engine_vibration") > 3.0)
        .withColumn("is_worn_brakes", F.col("brake_pad_wear_mm") < 3.0)
        .withColumn("is_brake_overheating", F.col("brake_temp_c") > 250.0)
        .withColumn("is_abs_active", F.col("abs_fault_indicator") == 1)
        .withColumn("is_battery_critical", (F.col("battery_voltage_v") < 11.8) | (F.col("battery_health_percent") < 50.0))
        .withColumn("is_alternator_fault", F.col("alternator_output_v") < 12.8)
        .withColumn(
            "vehicle_motion_state",
            F.when(F.col("vehicle_speed_kph") > 5.0, "IN_TRANSIT")
             .when(F.col("engine_rpm") > 400.0, "IDLE_RUNNING")
             .otherwise("PARKED"),
        )
    )

    # ---------------------------------------------------------------
    # 5. Write to Silver Delta (overwrite for safe local reruns)
    # ---------------------------------------------------------------
    (
        enriched.write.format("delta")
        .mode("overwrite")
        .partitionBy("event_date", "brand")
        .option("overwriteSchema", "true")
        .save(SILVER_TABLE_PATH)
    )
    print("[Silver] Batch write completed to", SILVER_TABLE_PATH)


if __name__ == "__main__":
    run_silver_batch()

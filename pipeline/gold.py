# pipeline/gold.py
"""Gold layer batch processing.
Creates rolling feature aggregates and fleet health summary tables from the Silver Delta lake.
This mirrors the logic in the Databricks notebook but runs locally in batch mode.
"""

import os
from pyspark.sql import functions as F
from pyspark.sql.window import Window

from .config import (
    SILVER_TABLE_PATH,
    GOLD_TABLE_PATH,
    BASE_PATH,
)
from .spark_init import get_spark_session

def run_gold_batch(app_name: str = "GoldBatch"):
    """Execute the Gold batch pipeline.

    Steps:
    1. Load Silver Delta table.
    2. Compute windowed aggregations (using a fixed 15‑minute window over the entire dataset).
    3. Calculate risk scores (engine, brake, battery) and overall health status.
    4. Write two Delta tables:
       - `gold_vehicle_rolling_features` (feature store for ML)
       - `gold_fleet_health_summary` (latest snapshot per vehicle)
    """
    spark = get_spark_session(app_name)

    # Load silver data
    silver_df = spark.read.format("delta").load(SILVER_TABLE_PATH)

    # Ensure an event_timestamp column exists (converted in Silver)
    if "event_timestamp" not in silver_df.columns:
        silver_df = silver_df.withColumn(
            "event_timestamp",
            F.to_timestamp(F.col("timestamp"), "yyyy-MM-dd HH:mm:ss"),
        )

    # Define the rolling window – 15 minutes sliding every 5 minutes (as in notebook)
    window_duration = "15 minutes"
    slide_duration = "5 minutes"

    gold_aggregated = (
        silver_df.groupBy(
            F.window(F.col("event_timestamp"), window_duration, slide_duration),
            "vehicle_id",
            "brand",
            "model",
            "vehicle_year",
        )
        .agg(
            # General & Driving Dynamics
            F.count("*").alias("ping_count"),
            F.round(F.avg("vehicle_speed_kph"), 2).alias("avg_speed_kph"),
            F.round(F.max("vehicle_speed_kph"), 2).alias("max_speed_kph"),
            F.round(F.max("odometer_km") - F.min("odometer_km"), 2).alias("window_distance_km"),
            # Engine Telemetry
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
            # Brake Telemetry
            F.round(F.min("brake_pad_wear_mm"), 2).alias("min_brake_pad_wear_mm"),
            F.round(F.avg("brake_temp_c"), 2).alias("avg_brake_temp_c"),
            F.round(F.max("brake_temp_c"), 2).alias("max_brake_temp_c"),
            F.sum("abs_fault_indicator").alias("abs_fault_count"),
            F.round(F.max(F.abs(F.col("wheel_speed_fl_kph") - F.col("wheel_speed_fr_kph"))), 2).alias("max_wheel_slip_diff_kph"),
            # Battery & Electrical
            F.round(F.min("battery_voltage_v"), 2).alias("min_battery_voltage_v"),
            F.round(F.avg("battery_voltage_v"), 2).alias("avg_battery_voltage_v"),
            F.round(F.min("battery_health_percent"), 2).alias("min_battery_health_pct"),
            F.round(F.avg("alternator_output_v"), 2).alias("avg_alternator_output_v"),
            F.round(F.max("battery_temp_c"), 2).alias("max_battery_temp_c"),
            # GPS – latest value in the window
            F.last("gps_latitude").alias("current_gps_lat"),
            F.last("gps_longitude").alias("current_gps_lon"),
        )
        .withColumn("window_start", F.col("window.start"))
        .withColumn("window_end", F.col("window.end"))
        .drop("window")
    )

    # Risk scoring – mirrors notebook logic
    gold_scored = (
        gold_aggregated
        .withColumn(
            "engine_risk_score",
            F.round(
                F.when(F.col("max_engine_temp_c") > 115.0, 50.0).otherwise(
                    F.when(F.col("max_engine_temp_c") > 105.0, 30.0).otherwise(0.0)
                )
                + F.when(F.col("min_oil_pressure_psi") < 15.0, 40.0).otherwise(
                    F.when(F.col("min_oil_pressure_psi") < 25.0, 20.0).otherwise(0.0)
                )
                + F.when(F.col("max_engine_vibration") > 3.5, 20.0).otherwise(0.0),
                1,
            ),
        )
        .withColumn(
            "brake_risk_score",
            F.round(
                F.when(F.col("min_brake_pad_wear_mm") < 2.0, 60.0).otherwise(
                    F.when(F.col("min_brake_pad_wear_mm") < 3.5, 30.0).otherwise(0.0)
                )
                + F.when(F.col("max_brake_temp_c") > 280.0, 30.0).otherwise(0.0)
                + F.when(F.col("abs_fault_count") > 0, 25.0).otherwise(0.0),
                1,
            ),
        )
        .withColumn(
            "battery_risk_score",
            F.round(
                F.when(F.col("min_battery_voltage_v") < 11.5, 50.0).otherwise(
                    F.when(F.col("min_battery_voltage_v") < 12.2, 25.0).otherwise(0.0)
                )
                + F.when(F.col("min_battery_health_pct") < 45.0, 40.0).otherwise(
                    F.when(F.col("min_battery_health_pct") < 65.0, 20.0).otherwise(0.0)
                )
                + F.when(F.col("avg_alternator_output_v") < 12.8, 30.0).otherwise(0.0),
                1,
            ),
        )
        .withColumn(
            "overall_risk_score",
            F.greatest(
                F.col("engine_risk_score"),
                F.col("brake_risk_score"),
                F.col("battery_risk_score"),
            ),
        )
        .withColumn(
            "fleet_health_status",
            F.when(F.col("overall_risk_score") >= 60.0, "CRITICAL")
            .when(F.col("overall_risk_score") >= 30.0, "WARNING")
            .otherwise("HEALTHY"),
        )
    )

    # Write rolling feature store
    gold_features_path = os.path.join(GOLD_TABLE_PATH, "gold_vehicle_rolling_features")
    gold_scored.write.format("delta").mode("overwrite").save(gold_features_path)
    print("[Gold] Rolling features written to", gold_features_path)

    # Materialize latest snapshot per vehicle
    latest_window = Window.partitionBy("vehicle_id").orderBy(F.col("window_end").desc())
    fleet_snapshot = (
        gold_scored.withColumn("row_num", F.row_number().over(latest_window))
        .filter(F.col("row_num") == 1)
        .drop("row_num")
    )
    gold_snapshot_path = os.path.join(GOLD_TABLE_PATH, "gold_fleet_health_summary")
    fleet_snapshot.write.format("delta").mode("overwrite").save(gold_snapshot_path)
    print("[Gold] Fleet health snapshot written to", gold_snapshot_path)

    spark.stop()

if __name__ == "__main__":
    run_gold_batch()

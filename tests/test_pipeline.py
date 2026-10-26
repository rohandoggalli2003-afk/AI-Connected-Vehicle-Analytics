# tests/test_pipeline.py
"""Pipeline integration tests for local Bronze and Silver layers.
These tests verify:
- Raw JSON telemetry files are ingested into a Delta Bronze table with metadata.
- Silver processing cleans, validates, deduplicates and joins vehicle master data.
- The resulting Delta tables can be read back and contain expected columns.
"""

import os
import shutil
import tempfile
import pytest
from pyspark.sql import SparkSession

# Project imports
from pipeline.spark_init import get_spark_session
from pipeline.config import (
    LANDING_PATH,
    BRONZE_TABLE_PATH,
    SILVER_TABLE_PATH,
    BASE_PATH,
)
from pipeline.bronze import run_bronze_batch
from pipeline.silver import run_silver_batch

@pytest.fixture(scope="module")
def spark():
    spark = get_spark_session("PipelineTest")
    yield spark
    spark.stop()

def test_bronze_ingestion(spark):
    # Clean Bronze directory
    if os.path.isdir(BRONZE_TABLE_PATH):
        shutil.rmtree(BRONZE_TABLE_PATH)
    run_bronze_batch()
    df = spark.read.format("delta").load(BRONZE_TABLE_PATH)
    assert df.count() > 0, "Bronze table should contain records"
    for meta_col in ["_ingest_timestamp", "_source_file"]:
        assert meta_col in df.columns, f"Missing {meta_col} in Bronze"
    for col_name in ["vehicle_id", "timestamp", "engine_temp_c", "brake_pad_wear_mm"]:
        assert col_name in df.columns, f"Missing expected column {col_name} in Bronze"

def test_silver_processing(spark):
    # Clean Silver directory
    if os.path.isdir(SILVER_TABLE_PATH):
        shutil.rmtree(SILVER_TABLE_PATH)
    # Ensure Bronze exists
    if not os.path.isdir(BRONZE_TABLE_PATH):
        run_bronze_batch()
    run_silver_batch()
    df = spark.read.format("delta").load(SILVER_TABLE_PATH)
    assert df.count() > 0, "Silver table should contain records"
    # Verify join added master columns
    for master_col in ["brand", "model", "vehicle_year", "fuel_type"]:
        assert master_col in df.columns, f"Missing master column {master_col}"
    # Diagnostic flags
    for flag in ["is_overheating", "is_low_oil", "is_battery_critical"]:
        assert flag in df.columns, f"Missing flag {flag}"
    # No duplicate vehicle_id+event_timestamp pairs
    distinct = df.select("vehicle_id", "event_timestamp").distinct().count()
    total = df.count()
    assert distinct == total, "Duplicates should be removed in Silver layer"

def test_empty_landing_handling(spark):
    # Temporarily move landing data
    backup_path = LANDING_PATH + "_backup"
    if os.path.isdir(LANDING_PATH):
        os.rename(LANDING_PATH, backup_path)
    os.makedirs(LANDING_PATH, exist_ok=True)
    try:
        # Clean Bronze
        if os.path.isdir(BRONZE_TABLE_PATH):
            shutil.rmtree(BRONZE_TABLE_PATH)
        run_bronze_batch()
        df = spark.read.format("delta").load(BRONZE_TABLE_PATH)
        assert df.count() == 0, "Bronze should be empty when landing folder has no files"
    finally:
        # Restore landing data
        shutil.rmtree(LANDING_PATH)
        if os.path.isdir(backup_path):
            os.rename(backup_path, LANDING_PATH)

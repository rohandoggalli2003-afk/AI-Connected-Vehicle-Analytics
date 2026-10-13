# Databricks notebook source
# MAGIC %md
# MAGIC # AI-Powered Connected Vehicle Analytics Platform (BDA)
# MAGIC ## Notebook 00: Environment Setup & Data Initialization
# MAGIC 
# MAGIC ### Overview
# MAGIC This notebook initializes the storage paths, Delta database/schema, and static reference tables required for the connected vehicle telematics pipeline.
# MAGIC 
# MAGIC - **Database**: `vehicle_telematics_db`
# MAGIC - **Architecture**: Medallion Architecture (Bronze -> Silver -> Gold)
# MAGIC - **Static Reference Data**: `vehicle_master` table (vehicle metadata, brands, models, fleet operational hubs)
# MAGIC - **Streaming Ingestion Source**: Auto Loader landing path

# COMMAND ----------

# MAGIC %md
# MAGIC ### 1. Parameter Configuration & Widgets
# MAGIC Configure base storage paths. In Databricks Community / Free Edition, DBFS `/FileStore` or Unity Catalog Volumes are used.

# COMMAND ----------

# Create widgets for configurable execution paths
dbutils.widgets.text("base_path", "/FileStore/vehicle_telematics", "Base Storage Path")
dbutils.widgets.text("database_name", "vehicle_telematics_db", "Database Name")
dbutils.widgets.dropdown("reset_tables", "false", ["false", "true"], "Reset Existing Tables")

BASE_PATH = dbutils.widgets.get("base_path")
DB_NAME = dbutils.widgets.get("database_name")
RESET_TABLES = dbutils.widgets.get("reset_tables").lower() == "true"

LANDING_PATH = f"{BASE_PATH}/landing"
BRONZE_PATH  = f"{BASE_PATH}/bronze"
SILVER_PATH  = f"{BASE_PATH}/silver"
GOLD_PATH    = f"{BASE_PATH}/gold"
CHECKPOINT_PATH = f"{BASE_PATH}/checkpoints"

print(f"[*] Base Path       : {BASE_PATH}")
print(f"[*] Database Name   : {DB_NAME}")
print(f"[*] Landing Path    : {LANDING_PATH}")
print(f"[*] Bronze Delta    : {BRONZE_PATH}")
print(f"[*] Silver Delta    : {SILVER_PATH}")
print(f"[*] Gold Delta      : {GOLD_PATH}")
print(f"[*] Checkpoints     : {CHECKPOINT_PATH}")

# COMMAND ----------

# MAGIC %md
# MAGIC ### 2. Initialize Database & Directory Layout

# COMMAND ----------

# Create Database / Schema
spark.sql(f"CREATE DATABASE IF NOT EXISTS {DB_NAME}")
spark.sql(f"USE {DB_NAME}")

# Create required directories on DBFS
dbutils.fs.mkdirs(LANDING_PATH)
dbutils.fs.mkdirs(BRONZE_PATH)
dbutils.fs.mkdirs(SILVER_PATH)
dbutils.fs.mkdirs(GOLD_PATH)
dbutils.fs.mkdirs(f"{CHECKPOINT_PATH}/bronze")
dbutils.fs.mkdirs(f"{CHECKPOINT_PATH}/silver")
dbutils.fs.mkdirs(f"{CHECKPOINT_PATH}/gold_features")
dbutils.fs.mkdirs(f"{CHECKPOINT_PATH}/gold_fleet")

if RESET_TABLES:
    print("[!] Reset flag active. Dropping existing tables and checkpoints...")
    spark.sql(f"DROP TABLE IF EXISTS {DB_NAME}.bronze_telematics")
    spark.sql(f"DROP TABLE IF EXISTS {DB_NAME}.silver_telematics")
    spark.sql(f"DROP TABLE IF EXISTS {DB_NAME}.vehicle_master")
    spark.sql(f"DROP TABLE IF EXISTS {DB_NAME}.gold_vehicle_rolling_features")
    spark.sql(f"DROP TABLE IF EXISTS {DB_NAME}.gold_fleet_health_summary")
    dbutils.fs.rm(CHECKPOINT_PATH, recurse=True)
    print("[+] Tables and checkpoints cleared.")

# COMMAND ----------

# MAGIC %md
# MAGIC ### 3. Create & Populate Static Vehicle Master Table
# MAGIC The Vehicle Master table contains stationary vehicle attributes (`vehicle_id`, `brand`, `vehicle_year`, `model`, `fuel_type`, `base_hub`) joined in the Silver layer.

# COMMAND ----------

from pyspark.sql.types import StructType, StructField, StringType, IntegerType, DoubleType
from pyspark.sql import functions as F

master_schema = StructType([
    StructField("vehicle_id", StringType(), False),
    StructField("brand", StringType(), False),
    StructField("vehicle_year", IntegerType(), False),
    StructField("model", StringType(), True),
    StructField("fuel_type", StringType(), True),
    StructField("base_hub", StringType(), True),
    StructField("base_lat", DoubleType(), True),
    StructField("base_lon", DoubleType(), True)
])

# Sample vehicle fleet data (50 commercial & passenger fleet vehicles)
fleet_records = [
    ("VH-1001", "Toyota", 2022, "Camry", "Gasoline", "San Francisco Bay", 37.7749, -122.4194),
    ("VH-1002", "Ford", 2021, "F-150", "Gasoline", "Dallas Logistics", 32.7767, -96.7970),
    ("VH-1003", "Volvo", 2020, "VNL 760", "Diesel", "Chicago Hub", 41.8781, -87.6298),
    ("VH-1004", "Tesla", 2023, "Model 3", "Electric", "Los Angeles Metro", 34.0522, -118.2437),
    ("VH-1005", "Freightliner", 2019, "Cascadia", "Diesel", "Atlanta Distribution", 33.7490, -84.3880),
    ("VH-1006", "BMW", 2023, "X5", "Hybrid", "San Francisco Bay", 37.7749, -122.4194),
    ("VH-1007", "Toyota", 2020, "RAV4", "Hybrid", "Los Angeles Metro", 34.0522, -118.2437),
    ("VH-1008", "Ford", 2022, "Explorer", "Gasoline", "Chicago Hub", 41.8781, -87.6298),
    ("VH-1009", "Volvo", 2021, "XC90", "Hybrid", "San Francisco Bay", 37.7749, -122.4194),
    ("VH-1010", "Tesla", 2024, "Model Y", "Electric", "San Francisco Bay", 37.7749, -122.4194),
    ("VH-1011", "Freightliner", 2020, "M2 106", "Diesel", "Dallas Logistics", 32.7767, -96.7970),
    ("VH-1012", "BMW", 2021, "330i", "Gasoline", "Atlanta Distribution", 33.7490, -84.3880),
    ("VH-1013", "Toyota", 2019, "Tundra", "Gasoline", "Dallas Logistics", 32.7767, -96.7970),
    ("VH-1014", "Ford", 2023, "Mach-E", "Electric", "Los Angeles Metro", 34.0522, -118.2437),
    ("VH-1015", "Volvo", 2022, "FH16", "Diesel", "Chicago Hub", 41.8781, -87.6298),
    ("VH-1016", "Tesla", 2022, "Semi", "Electric", "Dallas Logistics", 32.7767, -96.7970),
    ("VH-1017", "Freightliner", 2021, "Cascadia", "Diesel", "Chicago Hub", 41.8781, -87.6298),
    ("VH-1018", "BMW", 2024, "i4", "Electric", "San Francisco Bay", 37.7749, -122.4194),
    ("VH-1019", "Toyota", 2021, "Camry", "Gasoline", "Atlanta Distribution", 33.7490, -84.3880),
    ("VH-1020", "Ford", 2020, "F-150", "Gasoline", "Dallas Logistics", 32.7767, -96.7970),
    ("VH-1021", "Volvo", 2023, "VNL 760", "Diesel", "Chicago Hub", 41.8781, -87.6298),
    ("VH-1022", "Tesla", 2023, "Model 3", "Electric", "Los Angeles Metro", 34.0522, -118.2437),
    ("VH-1023", "Freightliner", 2018, "Cascadia", "Diesel", "Atlanta Distribution", 33.7490, -84.3880),
    ("VH-1024", "BMW", 2022, "X5", "Hybrid", "San Francisco Bay", 37.7749, -122.4194),
    ("VH-1025", "Toyota", 2024, "RAV4", "Hybrid", "Los Angeles Metro", 34.0522, -118.2437),
    ("VH-1026", "Ford", 2021, "Explorer", "Gasoline", "Dallas Logistics", 32.7767, -96.7970),
    ("VH-1027", "Volvo", 2020, "XC90", "Hybrid", "Chicago Hub", 41.8781, -87.6298),
    ("VH-1028", "Tesla", 2024, "Model Y", "Electric", "Atlanta Distribution", 33.7490, -84.3880),
    ("VH-1029", "Freightliner", 2022, "M2 106", "Diesel", "San Francisco Bay", 37.7749, -122.4194),
    ("VH-1030", "BMW", 2020, "330i", "Gasoline", "Dallas Logistics", 32.7767, -96.7970),
    ("VH-1031", "Toyota", 2022, "Tundra", "Gasoline", "Chicago Hub", 41.8781, -87.6298),
    ("VH-1032", "Ford", 2024, "Mach-E", "Electric", "Los Angeles Metro", 34.0522, -118.2437),
    ("VH-1033", "Volvo", 2021, "FH16", "Diesel", "Atlanta Distribution", 33.7490, -84.3880),
    ("VH-1034", "Tesla", 2023, "Semi", "Electric", "Dallas Logistics", 32.7767, -96.7970),
    ("VH-1035", "Freightliner", 2023, "Cascadia", "Diesel", "Chicago Hub", 41.8781, -87.6298),
    ("VH-1036", "BMW", 2023, "i4", "Electric", "San Francisco Bay", 37.7749, -122.4194),
    ("VH-1037", "Toyota", 2023, "Camry", "Gasoline", "Los Angeles Metro", 34.0522, -118.2437),
    ("VH-1038", "Ford", 2019, "F-150", "Gasoline", "Dallas Logistics", 32.7767, -96.7970),
    ("VH-1039", "Volvo", 2022, "VNL 760", "Diesel", "Atlanta Distribution", 33.7490, -84.3880),
    ("VH-1040", "Tesla", 2024, "Model 3", "Electric", "San Francisco Bay", 37.7749, -122.4194),
    ("VH-1041", "Freightliner", 2020, "Cascadia", "Diesel", "Chicago Hub", 41.8781, -87.6298),
    ("VH-1042", "BMW", 2021, "X5", "Hybrid", "Los Angeles Metro", 34.0522, -118.2437),
    ("VH-1043", "Toyota", 2021, "RAV4", "Hybrid", "Dallas Logistics", 32.7767, -96.7970),
    ("VH-1044", "Ford", 2023, "Explorer", "Gasoline", "Atlanta Distribution", 33.7490, -84.3880),
    ("VH-1045", "Volvo", 2024, "XC90", "Hybrid", "San Francisco Bay", 37.7749, -122.4194),
    ("VH-1046", "Tesla", 2022, "Model Y", "Electric", "Chicago Hub", 41.8781, -87.6298),
    ("VH-1047", "Freightliner", 2021, "M2 106", "Diesel", "Dallas Logistics", 32.7767, -96.7970),
    ("VH-1048", "BMW", 2024, "330i", "Gasoline", "Los Angeles Metro", 34.0522, -118.2437),
    ("VH-1049", "Toyota", 2020, "Tundra", "Gasoline", "Atlanta Distribution", 33.7490, -84.3880),
    ("VH-1050", "Ford", 2022, "Mach-E", "Electric", "San Francisco Bay", 37.7749, -122.4194)
]

df_master_spark = spark.createDataFrame(fleet_records, schema=master_schema)

# Write to Delta Lake
df_master_spark.write \
    .format("delta") \
    .mode("overwrite") \
    .saveAsTable(f"{DB_NAME}.vehicle_master")

print(f"[+] Static vehicle_master Delta table created with {df_master_spark.count()} vehicles.")

# COMMAND ----------

# MAGIC %md
# MAGIC ### 4. Verify Static Table

# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT brand, fuel_type, count(*) as vehicle_count, min(vehicle_year) as oldest_year, max(vehicle_year) as newest_year
# MAGIC FROM vehicle_telematics_db.vehicle_master
# MAGIC GROUP BY brand, fuel_type
# MAGIC ORDER BY brand, vehicle_count DESC;

# COMMAND ----------

# MAGIC %md
# MAGIC ### 5. Environment Readiness Check Complete
# MAGIC - Database `vehicle_telematics_db` initialized.
# MAGIC - Storage paths and checkpoints configured.
# MAGIC - Static `vehicle_master` table ready for Silver layer join.
# MAGIC - Proceed to Notebook **01_bronze_ingestion**.

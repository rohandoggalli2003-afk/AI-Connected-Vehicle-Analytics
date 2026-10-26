import os
from pyspark.sql import SparkSession

def get_spark_session(app_name: str = "VehicleMedallionPipeline") -> SparkSession:
    """Create a SparkSession configured for Delta Lake.

    The function sets the necessary extensions and catalog for Delta Lake
    and respects the environment variables ``BASE_PATH`` and ``JAR_PATH``
    if they are provided. It also enables Hive support so that ``spark.sql``
    statements can be used to create databases and tables.
    """
    builder = (
        SparkSession.builder
        .appName(app_name)
        .master("local[*]")
        .config("spark.sql.extensions", "io.delta.sql.DeltaSparkSessionExtension")
        .config(
            "spark.sql.catalog.spark_catalog",
            "org.apache.spark.sql.delta.catalog.DeltaCatalog",
        )
        .config("spark.serializer", "org.apache.spark.serializer.KryoSerializer")
        .config("spark.jars.packages", "io.delta:delta-spark_2.12:3.0.0")
        .enableHiveSupport()
    )
    jar_path = os.getenv("DELTA_JAR_PATH")
    if jar_path:
        builder = builder.config("spark.jars", jar_path)
    spark = builder.getOrCreate()
    return spark

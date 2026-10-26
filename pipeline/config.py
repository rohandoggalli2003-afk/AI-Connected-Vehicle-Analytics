import os

# Base storage path (mirrors the Databricks /FileStore layout for local runs)
# Users can override via the BASE_PATH environment variable.
# Default to the project's data folder so that landing data is found.
BASE_PATH = os.getenv(
    "BASE_PATH",
    os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data")),
)

# Table directories (Delta tables are just folders with _delta_log)
BRONZE_TABLE_PATH = os.path.join(BASE_PATH, "bronze")
SILVER_TABLE_PATH = os.path.join(BASE_PATH, "silver")
GOLD_TABLE_PATH = os.path.join(BASE_PATH, "gold")

# Landing (raw) data location – matches the existing JSON batches
LANDING_PATH = os.path.join(BASE_PATH, "landing")

# Checkpoint directories for each streaming job
BRONZE_CHECKPOINT = os.path.join(BASE_PATH, "checkpoints", "bronze")
SILVER_CHECKPOINT = os.path.join(BASE_PATH, "checkpoints", "silver")
GOLD_CHECKPOINT = os.path.join(BASE_PATH, "checkpoints", "gold")

# Ensure required directories exist (idempotent)
for _p in [BRONZE_TABLE_PATH, SILVER_TABLE_PATH, GOLD_TABLE_PATH,
           BRONZE_CHECKPOINT, SILVER_CHECKPOINT, GOLD_CHECKPOINT]:
    os.makedirs(_p, exist_ok=True)

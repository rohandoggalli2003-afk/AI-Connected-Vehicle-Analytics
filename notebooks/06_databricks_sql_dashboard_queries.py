# Databricks notebook source
# MAGIC %md
# MAGIC # AI-Powered Connected Vehicle Analytics Platform (BDA)
# MAGIC ## Notebook 06: Databricks SQL & Lakeview Fleet Dashboard
# MAGIC 
# MAGIC ### Phase D: Intelligent Fleet Monitoring & BI Dashboard
# MAGIC 
# MAGIC This notebook provides the complete set of analytical queries for building an interactive **Databricks SQL Dashboard** or **Lakeview Dashboard** on top of the Gold Delta Lake tables.
# MAGIC 
# MAGIC #### Visualizations Included:
# MAGIC 1. **Fleet Health Gauge & KPI Counters** (Active fleet, healthy %, critical count)
# MAGIC 2. **GPS Geolocation Map** (Vehicle positions colored by operational status)
# MAGIC 3. **High-Risk Vehicle Priority Queue** (Ranked by `failure_probability`)
# MAGIC 4. **Subsystem Anomaly Spikes by Brand** (Engine, Brake, Battery)
# MAGIC 5. **Predictive AI Diagnostic Work Orders** (Directly inspect generated maintenance reports)

# COMMAND ----------

# MAGIC %sql
# MAGIC USE vehicle_telematics_db;

# COMMAND ----------

# MAGIC %md
# MAGIC ### Widget 1: Fleet Health KPI Summary
# MAGIC Use this query in a Databricks Counter / KPI widget.

# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT 
# MAGIC     COUNT(*) AS total_fleet_size,
# MAGIC     SUM(CASE WHEN fleet_health_status = 'HEALTHY' THEN 1 ELSE 0 END) AS healthy_vehicles,
# MAGIC     SUM(CASE WHEN fleet_health_status = 'WARNING' THEN 1 ELSE 0 END) AS warning_vehicles,
# MAGIC     SUM(CASE WHEN fleet_health_status = 'CRITICAL' THEN 1 ELSE 0 END) AS critical_vehicles,
# MAGIC     ROUND(AVG(avg_speed_kph), 1) AS fleet_avg_speed_kph,
# MAGIC     ROUND(AVG(overall_risk_score), 1) AS avg_fleet_risk_index
# MAGIC FROM gold_fleet_health_summary;

# COMMAND ----------

# MAGIC %md
# MAGIC ### Widget 2: Fleet Risk Distribution by Status
# MAGIC Use this query for a Donut / Pie Chart widget.

# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT 
# MAGIC     fleet_health_status,
# MAGIC     COUNT(*) AS vehicle_count,
# MAGIC     ROUND(COUNT(*) * 100.0 / SUM(COUNT(*)) OVER (), 1) AS percentage
# MAGIC FROM gold_fleet_health_summary
# MAGIC GROUP BY fleet_health_status
# MAGIC ORDER BY vehicle_count DESC;

# COMMAND ----------

# MAGIC %md
# MAGIC ### Widget 3: High-Risk Priority Queue (Failure Probabilities & Diagnosis)
# MAGIC Use this query for a Table or Details View.

# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT 
# MAGIC     m.vehicle_id,
# MAGIC     m.brand,
# MAGIC     m.model,
# MAGIC     m.vehicle_year,
# MAGIC     m.failure_probability,
# MAGIC     m.predicted_failure_type,
# MAGIC     g.max_engine_temp_c,
# MAGIC     g.min_oil_pressure_psi,
# MAGIC     g.min_brake_pad_wear_mm,
# MAGIC     g.min_battery_voltage_v,
# MAGIC     g.current_gps_lat,
# MAGIC     g.current_gps_lon
# MAGIC FROM gold_vehicle_maintenance_reports m
# MAGIC JOIN gold_fleet_health_summary g ON m.vehicle_id = g.vehicle_id
# MAGIC WHERE m.failure_probability >= 0.30
# MAGIC ORDER BY m.failure_probability DESC;

# COMMAND ----------

# MAGIC %md
# MAGIC ### Widget 4: Real-Time Subsystem Health by Brand
# MAGIC Use this query for a Grouped Bar Chart.

# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT 
# MAGIC     brand,
# MAGIC     ROUND(AVG(engine_risk_score), 1) AS avg_engine_risk,
# MAGIC     ROUND(AVG(brake_risk_score), 1) AS avg_brake_risk,
# MAGIC     ROUND(AVG(battery_risk_score), 1) AS avg_battery_risk,
# MAGIC     COUNT(*) AS total_vehicles
# MAGIC FROM gold_fleet_health_summary
# MAGIC GROUP BY brand
# MAGIC ORDER BY brand;

# COMMAND ----------

# MAGIC %md
# MAGIC ### Widget 5: Interactive GPS Fleet Map
# MAGIC Use this query in the Databricks Map Visualization (Point Map).
# MAGIC - Latitude: `current_gps_lat`
# MAGIC - Longitude: `current_gps_lon`
# MAGIC - Color: `fleet_health_status`

# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT 
# MAGIC     vehicle_id,
# MAGIC     brand,
# MAGIC     model,
# MAGIC     fleet_health_status,
# MAGIC     overall_risk_score,
# MAGIC     avg_speed_kph,
# MAGIC     current_gps_lat,
# MAGIC     current_gps_lon
# MAGIC FROM gold_fleet_health_summary
# MAGIC WHERE current_gps_lat IS NOT NULL AND current_gps_lon IS NOT NULL;

# COMMAND ----------

# MAGIC %md
# MAGIC ### Widget 6: AI Diagnostic Work Order Inspector
# MAGIC Inspect the generated natural language maintenance work order for any selected vehicle.

# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT 
# MAGIC     vehicle_id,
# MAGIC     brand,
# MAGIC     model,
# MAGIC     failure_probability,
# MAGIC     predicted_failure_type,
# MAGIC     maintenance_report
# MAGIC FROM gold_vehicle_maintenance_reports
# MAGIC WHERE failure_probability >= 0.50
# MAGIC ORDER BY failure_probability DESC;

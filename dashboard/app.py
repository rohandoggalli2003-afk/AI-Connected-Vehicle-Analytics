"""
AI-Powered Connected Vehicle Analytics Platform (BDA)
Clean & Visible Fleet Monitoring Dashboard (Streamlit)

Focused on core requirements:
1. Fleet Health KPIs (Total, Healthy, Warning, Critical)
2. Live Vehicle Component Failure Predictions (Engine, Brake, Battery)
3. Direct AI Maintenance Diagnostic Work Order Viewer
"""

import os
import sys
import time
import glob
import json
import datetime

# Ensure project root is in sys.path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import pandas as pd
import numpy as np
import streamlit as st

from data_generator.generate_telematics import generate_vehicle_master, VehicleState
from reports.report_generator import MaintenanceReportGenerator

# Page setup
st.set_page_config(
    page_title="Fleet Health & Failure Predictor",
    page_icon="🚛",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom clear CSS styling
st.markdown("""
<style>
    .kpi-box {
        background: #1e293b;
        border-radius: 10px;
        padding: 16px;
        text-align: center;
        border: 1px solid #334155;
    }
    .kpi-title {
        color: #94a3b8;
        font-size: 0.85rem;
        font-weight: 600;
        text-transform: uppercase;
    }
    .kpi-num {
        font-size: 2.2rem;
        font-weight: 700;
        margin-top: 4px;
    }
    .status-badge-critical {
        color: #ef4444;
        font-weight: 700;
    }
    .status-badge-warning {
        color: #f59e0b;
        font-weight: 700;
    }
    .status-badge-healthy {
        color: #10b981;
        font-weight: 700;
    }
    .stream-info {
        background: #0f172a;
        border-left: 4px solid #3b82f6;
        padding: 10px 14px;
        border-radius: 6px;
        margin-bottom: 20px;
        font-size: 0.92rem;
    }
</style>
""", unsafe_allow_html=True)

from dashboard.data_loader import load_fleet_data, emit_realtime_batch

# Load data
df_fleet, total_pings, last_mtime = load_fleet_data()

# Sidebar: Controls & Filters
st.sidebar.header("⚙️ Streaming & Filters")
auto_refresh = st.sidebar.toggle("Enable Live Auto-Refresh", value=False)
refresh_secs = st.sidebar.slider("Refresh Rate (seconds)", min_value=1, max_value=10, value=2)

if st.sidebar.button("⚡ Push Live Batch (20 Pings)"):
    fname, count, ts = emit_realtime_batch(20, inject_faults=True)
    time.sleep(0.5)
    st.rerun()

st.sidebar.markdown("---")

# Filters
if not df_fleet.empty:
    brand_options = ["All Brands"] + sorted(list(df_fleet["brand"].dropna().unique()))
    chosen_brand = st.sidebar.selectbox("Filter Brand", brand_options)
    
    available_statuses = sorted(list(df_fleet["status"].dropna().unique()))
    status_options = ["All Statuses"] + available_statuses
    chosen_status = st.sidebar.selectbox("Filter Status", status_options)
    
    display_df = df_fleet.copy()
    if chosen_brand != "All Brands":
        display_df = display_df[display_df["brand"] == chosen_brand]
    if chosen_status != "All Statuses":
        display_df = display_df[display_df["status"] == chosen_status]

    # Add intuitive visual badges for health status
    display_df["status_display"] = display_df["status"].map({
        "HEALTHY": "🟢 Healthy",
        "WARNING": "🟠 Warning",
        "CRITICAL": "🔴 Critical"
    }).fillna(display_df["status"])
else:
    display_df = pd.DataFrame()

# Main Dashboard View
st.title("🚛 Connected Vehicle Analytics & Failure Prediction")

last_update_str = datetime.datetime.fromtimestamp(last_mtime).strftime("%H:%M:%S UTC") if last_mtime > 0 else "Just Now"

st.markdown(f"""
<div class="stream-info">
    📡 <b>Live Pipeline Status:</b> Ingested <b>{total_pings:,}</b> real-time records. Latest stream batch received at <b>{last_update_str}</b>.
</div>
""", unsafe_allow_html=True)

if df_fleet.empty:
    st.warning("No data found. Please run `python data_generator/generate_telematics.py` first.")
    st.stop()

# 1. Executive Fleet Health KPIs (4 Clear KPI Boxes with Probability Ranges)
col1, col2, col3, col4 = st.columns(4)
total_veh = len(df_fleet)
heal_cnt = (df_fleet["status"] == "HEALTHY").sum()
warn_cnt = (df_fleet["status"] == "WARNING").sum()
crit_cnt = (df_fleet["status"] == "CRITICAL").sum()

with col1:
    st.markdown(f"""
    <div class="kpi-box">
        <div class="kpi-title">Total Monitored Fleet</div>
        <div class="kpi-num" style="color: #38bdf8;">{total_veh}</div>
        <div style="font-size: 0.78rem; color: #94a3b8; font-weight: 500; margin-top: 4px;">Commercial Trucks</div>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown(f"""
    <div class="kpi-box">
        <div class="kpi-title">Healthy (Normal)</div>
        <div class="kpi-num" style="color: #10b981;">{heal_cnt}</div>
        <div style="font-size: 0.78rem; color: #6ee7b7; font-weight: 600; margin-top: 4px;">Risk: 0% – 29%</div>
    </div>
    """, unsafe_allow_html=True)

with col3:
    st.markdown(f"""
    <div class="kpi-box">
        <div class="kpi-title">Warning (Inspection)</div>
        <div class="kpi-num" style="color: #f59e0b;">{warn_cnt}</div>
        <div style="font-size: 0.78rem; color: #fde68a; font-weight: 600; margin-top: 4px;">Risk: 30% – 59%</div>
    </div>
    """, unsafe_allow_html=True)

with col4:
    st.markdown(f"""
    <div class="kpi-box">
        <div class="kpi-title">Critical (Failure Imminent)</div>
        <div class="kpi-num" style="color: #ef4444;">{crit_cnt}</div>
        <div style="font-size: 0.78rem; color: #fca5a5; font-weight: 600; margin-top: 4px;">Risk: 60% – 100%</div>
    </div>
    """, unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# 2. Probability Range & Health Status Pills
st.markdown("""
<div style="display: flex; gap: 14px; margin-bottom: 20px; flex-wrap: wrap;">
    <div style="background: #064e3b; border-left: 4px solid #10b981; padding: 9px 18px; border-radius: 6px; font-weight: 700; color: #6ee7b7; font-size: 0.92rem; display: flex; align-items: center;">
        🟢 HEALTHY (0% – 29%)
    </div>
    <div style="background: #78350f; border-left: 4px solid #f59e0b; padding: 9px 18px; border-radius: 6px; font-weight: 700; color: #fde68a; font-size: 0.92rem; display: flex; align-items: center;">
        🟠 WARNING (30% – 59%)
    </div>
    <div style="background: #7f1d1d; border-left: 4px solid #ef4444; padding: 9px 18px; border-radius: 6px; font-weight: 700; color: #fca5a5; font-size: 0.92rem; display: flex; align-items: center;">
        🔴 CRITICAL (60% – 100%)
    </div>
</div>
""", unsafe_allow_html=True)

# 3. Main Fleet Health & Component Failure Table
st.subheader("📋 Fleet Component Failure Status & Predictions")

table_cols = [
    "vehicle_id", "brand", "model", "vehicle_year",
    "status_display", "failure_probability", "predicted_failure_type",
    "engine_temp_c", "oil_pressure_psi", "brake_pad_wear_mm", "battery_voltage_v"
]

sorted_table = display_df.sort_values(by="failure_probability", ascending=False)[table_cols]

st.dataframe(
    sorted_table,
    column_config={
        "vehicle_id": "Vehicle ID",
        "brand": "Brand",
        "model": "Model",
        "vehicle_year": "Year",
        "status_display": "Health Status",
        "failure_probability": st.column_config.ProgressColumn(
            "Failure Probability",
            format="%.0f%%",
            min_value=0.0,
            max_value=1.0,
        ),
        "predicted_failure_type": "Predicted Failure Type",
        "engine_temp_c": st.column_config.NumberColumn("Engine Temp (°C)", format="%.1f"),
        "oil_pressure_psi": st.column_config.NumberColumn("Oil Pressure (PSI)", format="%.1f"),
        "brake_pad_wear_mm": st.column_config.NumberColumn("Brake Pad (mm)", format="%.2f"),
        "battery_voltage_v": st.column_config.NumberColumn("Battery (V)", format="%.2f")
    },
    width="stretch",
    hide_index=True
)

st.markdown("---")

# 3. AI-Generated Maintenance Report Section
st.subheader("📑 AI Maintenance Report Generator")
st.caption("Select any vehicle from the fleet to view its instant diagnostic work order:")

veh_list = sorted(list(df_fleet["vehicle_id"].unique()))
selected_vid = st.selectbox("Select Vehicle ID", veh_list, index=0)

if selected_vid:
    v_info = df_fleet[df_fleet["vehicle_id"] == selected_vid].iloc[0]
    
    rep_gen = MaintenanceReportGenerator()
    report_text = rep_gen.generate_report(selected_vid, v_info.to_dict(), v_info.to_dict())
    
    col_rep_left, col_rep_right = st.columns([3, 1])
    
    with col_rep_left:
        st.text_area("Generated Diagnostic Work Order", report_text, height=320)
        
    with col_rep_right:
        st.markdown(f"**Vehicle:** `{selected_vid}`")
        st.markdown(f"**Model:** {v_info.get('vehicle_year', '')} {v_info.get('brand', '')} {v_info.get('model', '')}")
        st.markdown(f"**Failure Risk:** `{v_info['failure_probability']*100:.0f}%`")
        st.markdown(f"**Predicted Fault:** `{v_info['predicted_failure_type']}`")
        
        st.download_button(
            label="📥 Download Work Order (.txt)",
            data=report_text,
            file_name=f"Maintenance_Work_Order_{selected_vid}.txt",
            mime="text/plain",
            width="stretch"
        )

# Auto-refresh loop
if auto_refresh:
    time.sleep(refresh_secs)
    st.rerun()

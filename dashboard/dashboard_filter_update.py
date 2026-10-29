"""
AI-Powered Connected Vehicle Analytics Platform (BDA)
Clean & Visible Fleet Monitoring Dashboard (Streamlit)

Focused on core requirements:
1. Fleet Health KPIs (Total, Healthy, Warning, Critical)
2. Live Fleet GPS Geolocation & Interactive Map Tracking
3. Live Vehicle Component Failure Predictions (Engine, Brake, Battery)
4. Direct AI Maintenance Diagnostic Work Order Viewer
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
import pydeck as pdk

try:
    import plotly.graph_objects as go
except ImportError:
    go = None

from data_generator.generate_telematics import generate_vehicle_master, VehicleState
from reports.report_generator import MaintenanceReportGenerator
from dashboard.data_loader_updated import load_fleet_data, emit_realtime_batch, reset_fleet_cache


# Persistent historical analytics storage. This path resolves to <project>/data when
# app.py is located in the dashboard directory, matching the existing project layout.
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
HISTORY_PATH = os.path.join(DATA_DIR, "dashboard_history.csv")
HISTORY_CAPTURE_INTERVAL_SECONDS = 60
HISTORY_SENSOR_LABELS = {
    "engine_temp_c": "Engine Temperature (°C)",
    "battery_voltage_v": "Battery Voltage (V)",
    "brake_pad_wear_mm": "Brake Pad Wear (mm)",
}


def _normalise_history_frame(frame: pd.DataFrame) -> pd.DataFrame:
    """Standardise timestamps, numeric fields, and status labels for historical charts."""
    if frame.empty:
        return frame
    if "captured_at" not in frame.columns:
        if "timestamp" in frame.columns:
            frame["captured_at"] = frame["timestamp"]
        else:
            frame["captured_at"] = pd.NaT
    frame["captured_at"] = pd.to_datetime(frame["captured_at"], utc=True, errors="coerce")
    if "status" not in frame.columns:
        frame["status"] = ""
    frame["status"] = frame["status"].astype(str).str.upper()
    for column in ["failure_probability", *HISTORY_SENSOR_LABELS.keys()]:
        if column not in frame.columns:
            frame[column] = np.nan
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    return frame.dropna(subset=["captured_at"])


def _read_historical_data() -> pd.DataFrame:
    """Load persistent snapshots, and seed sensor history from existing landing telemetry once."""
    os.makedirs(DATA_DIR, exist_ok=True)
    if os.path.exists(HISTORY_PATH):
        try:
            return _normalise_history_frame(pd.read_csv(HISTORY_PATH))
        except (pd.errors.EmptyDataError, OSError, ValueError):
            pass

    # If no snapshot file exists yet, reuse timestamped telemetry already on disk.
    # Raw historical records provide sensor trends; model risk/status become available
    # as the dashboard captures scored fleet snapshots.
    seed_rows = []
    landing_dir = os.path.join(DATA_DIR, "landing")
    for file_path in sorted(glob.glob(os.path.join(landing_dir, "*.json"))):
        try:
            with open(file_path, "r", encoding="utf-8") as source:
                for raw_line in source:
                    raw_line = raw_line.strip()
                    if not raw_line:
                        continue
                    try:
                        record = json.loads(raw_line)
                    except json.JSONDecodeError:
                        continue
                    if not record.get("vehicle_id"):
                        continue
                    stamp = record.get("timestamp") or record.get("captured_at")
                    if not stamp:
                        continue
                    row = {
                        "vehicle_id": record.get("vehicle_id"),
                        "captured_at": stamp,
                        "status": "",
                        "failure_probability": np.nan,
                    }
                    row.update({sensor: record.get(sensor, np.nan) for sensor in HISTORY_SENSOR_LABELS})
                    seed_rows.append(row)
        except OSError:
            continue

    if seed_rows:
        seeded = _normalise_history_frame(pd.DataFrame(seed_rows))
        # Save a seed marker through the history CSV so later runs do not rescan
        # and duplicate every landing record.
        seeded.to_csv(HISTORY_PATH, index=False)
        return seeded
    return pd.DataFrame(columns=[
        "vehicle_id", "captured_at", "status", "failure_probability",
        *HISTORY_SENSOR_LABELS.keys()
    ])


def _capture_fleet_snapshot(fleet: pd.DataFrame) -> None:
    """Persist one scored fleet snapshot at most once per minute per Streamlit session."""
    if fleet.empty or "vehicle_id" not in fleet.columns:
        return
    # Seed sensor history from existing timestamped landing files before the first
    # scored snapshot creates the persistent history file.
    if not os.path.exists(HISTORY_PATH):
        _read_historical_data()

    now = datetime.datetime.now(datetime.timezone.utc)
    last_capture = st.session_state.get("historical_last_capture_epoch", 0.0)
    if time.time() - float(last_capture) < HISTORY_CAPTURE_INTERVAL_SECONDS:
        return

    columns = ["vehicle_id", "status", "failure_probability", *HISTORY_SENSOR_LABELS.keys()]
    snapshot = fleet[[column for column in columns if column in fleet.columns]].copy()
    snapshot["captured_at"] = now.isoformat()
    for column in ["failure_probability", *HISTORY_SENSOR_LABELS.keys()]:
        if column in snapshot.columns:
            snapshot[column] = pd.to_numeric(snapshot[column], errors="coerce")
    # Keep only the fields needed by the historical analytics feature.
    for column in ["status", "failure_probability", *HISTORY_SENSOR_LABELS.keys()]:
        if column not in snapshot.columns:
            snapshot[column] = np.nan if column != "status" else ""
    snapshot = snapshot[["vehicle_id", "captured_at", "status", "failure_probability", *HISTORY_SENSOR_LABELS.keys()]]

    os.makedirs(DATA_DIR, exist_ok=True)
    snapshot.to_csv(HISTORY_PATH, mode="a", header=not os.path.exists(HISTORY_PATH), index=False)
    st.session_state["historical_last_capture_epoch"] = time.time()

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
        background: #111827;
        color: #f9fafb;
        border: 1px solid #374151;
        border-left: 5px solid #60a5fa;
        padding: 13px 16px;
        border-radius: 8px;
        margin-bottom: 20px;
        font-size: 0.96rem;
        line-height: 1.6;
        font-weight: 500;
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.16);
    }
    .stream-info b {
        color: #ffffff;
        font-weight: 750;
    }
    .stream-info i {
        color: #d1d5db;
        font-style: normal;
    }
</style>
""", unsafe_allow_html=True)

# Initial data load for reference and filter options
df_fleet_init, total_pings_init, last_mtime_init = load_fleet_data()

# Sidebar: Controls & Filters
st.sidebar.header("⚙️ Streaming & Filters")
auto_refresh = st.sidebar.toggle("Enable Live Auto-Refresh", value=False, key="auto_refresh_toggle")
refresh_secs = st.sidebar.slider("Refresh Rate (seconds)", min_value=1, max_value=10, value=2, key="refresh_rate_slider")
auto_simulate = st.sidebar.toggle(
    "Auto-Simulate Live Telematics Stream",
    value=True,
    help="Continuously emits live vehicle telematics pings on each refresh cycle so you see real-time data motion",
    key="auto_sim_toggle"
)

if st.sidebar.button("⚡ Push Live Batch (20 Pings)", key="push_batch_btn"):
    fname, count, ts = emit_realtime_batch(20, inject_faults=True)
    st.sidebar.success(f"Emitted {count} pings!")
    st.rerun()

if st.sidebar.button("🔄 Reload Fleet Cache", key="reload_cache_btn"):
    reset_fleet_cache()
    df_fleet_init, total_pings_init, last_mtime_init = load_fleet_data(force_reload=True)
    st.sidebar.success("Cache refreshed from vehicle_master!")
    st.rerun()

# Real-time status badge in sidebar
if auto_refresh:
    sim_text = " + Auto-Simulating" if auto_simulate else ""
    st.sidebar.success(f"🟢 Streaming Active ({refresh_secs}s interval{sim_text})")
else:
    st.sidebar.info("⏸️ Auto-Refresh Paused")

st.sidebar.markdown("---")

# Filters in Sidebar
if not df_fleet_init.empty:
    brand_options = ["All Brands"] + sorted(list(df_fleet_init["brand"].dropna().unique()))
    chosen_brand = st.sidebar.selectbox("Filter Brand", brand_options, key="filter_brand_select")
    
    available_statuses = sorted(list(df_fleet_init["status"].dropna().unique()))
    status_options = ["All Statuses"] + available_statuses
    chosen_status = st.sidebar.selectbox("Filter Status", status_options, key="filter_status_select")

    hub_options = ["All Hubs"] + sorted(list(df_fleet_init["base_hub"].dropna().unique()))
    chosen_hub = st.sidebar.selectbox("Filter Base Hub", hub_options, key="filter_hub_select")
else:
    chosen_brand = "All Brands"
    chosen_status = "All Statuses"
    chosen_hub = "All Hubs"

# Main Dashboard View Header
st.title("🚛 Connected Vehicle Analytics & Failure Prediction")

# Dynamic auto-refresh fragment: updates on client timer without blocking the Python thread
run_interval = f"{refresh_secs}s" if auto_refresh else None

@st.fragment(run_every=run_interval)
def render_live_telematics_dashboard(filter_brand: str, filter_status: str, filter_hub: str, is_auto: bool, is_sim: bool):
    # If auto-refresh and auto-simulate are enabled, emit continuous live telemetry
    if is_auto and is_sim:
        emit_realtime_batch(10, inject_faults=True)

    # Ingest / load latest data (incremental cache executes in < 0.2s)
    df_fleet, total_pings, last_mtime = load_fleet_data()

    if df_fleet.empty:
        st.warning("No data found. Please run `python data_generator/generate_telematics.py` first.")
        return

    # Keep a durable, time-spaced history of scored fleet states for trend analytics.
    _capture_fleet_snapshot(df_fleet)

    # Filter according to sidebar selections
    display_df = df_fleet.copy()
    if filter_brand != "All Brands":
        display_df = display_df[display_df["brand"] == filter_brand]
    if filter_status != "All Statuses":
        display_df = display_df[display_df["status"] == filter_status]
    if filter_hub != "All Hubs":
        display_df = display_df[display_df["base_hub"] == filter_hub]

    # Visual badges for health status
    display_df["status_display"] = display_df["status"].map({
        "HEALTHY": "🟢 Healthy",
        "WARNING": "🟠 Warning",
        "CRITICAL": "🔴 Critical"
    }).fillna(display_df["status"])

    if "data_quality_status" in display_df.columns:
        display_df["data_quality_display"] = display_df["data_quality_status"].map({
            "HIGH": "🟢 High",
            "DEGRADED": "🟡 Degraded (Imputed)",
            "POOR": "🟠 Noisy",
            "STALE": "⚪ Stale",
            "UNAVAILABLE": "🔴 Unavailable"
        }).fillna(display_df["data_quality_status"])
    else:
        display_df["data_quality_display"] = "🟢 High"

    last_update_str = datetime.datetime.fromtimestamp(last_mtime).strftime("%H:%M:%S UTC") if last_mtime > 0 else "Just Now"
    now_str = datetime.datetime.now(datetime.timezone.utc).strftime("%H:%M:%S UTC")

    pulse_indicator = f"🟢 <b>Live Stream:</b> Refreshing every <b>{refresh_secs}s</b>" if is_auto else "⚪ <b>Stream:</b> Paused (toggle in sidebar)"

    st.markdown(f"""
    <div class="stream-info">
        📡 <b>Live Pipeline Status:</b> Ingested <b>{total_pings:,}</b> real-time records. Latest batch at <b>{last_update_str}</b>. {pulse_indicator} &nbsp;·&nbsp; <i>Synced at {now_str}</i>
    </div>
    """, unsafe_allow_html=True)

    # 1. Executive Fleet Health KPIs (4 Clear KPI Boxes)
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
            <div style="font-size: 0.78rem; color: #94a3b8; font-weight: 500; margin-top: 4px;">Commercial & Passenger</div>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown(f"""
        <div class="kpi-box">
            <div class="kpi-title">Healthy (Normal)</div>
            <div class="kpi-num" style="color: #10b981;">{heal_cnt}</div>
        </div>
        """, unsafe_allow_html=True)

    with col3:
        st.markdown(f"""
        <div class="kpi-box">
            <div class="kpi-title">Warning (Inspection)</div>
            <div class="kpi-num" style="color: #f59e0b;">{warn_cnt}</div>
        </div>
        """, unsafe_allow_html=True)

    with col4:
        st.markdown(f"""
        <div class="kpi-box">
            <div class="kpi-title">Critical (Failure Imminent)</div>
            <div class="kpi-num" style="color: #ef4444;">{crit_cnt}</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # 3. Interactive Fleet GPS Geolocation Map (Streamlit Standard Map)
    st.subheader("🗺️ Live Fleet GPS Geolocation Tracking")

    # Region focus selector
    region_choice = st.radio(
        "Select Map Focus Region:",
        ["🌐 Global Fleet", "🇮🇳 India Hubs (Bengaluru, Mumbai, Delhi, Chennai, Hyderabad, Pune, Manipal)", "🇺🇸 US Hubs (Chicago, SF, LA, Atlanta, Dallas)"],
        horizontal=True,
        key="map_region_focus"
    )

    # Filter map records based on chosen region
    map_df = display_df.copy()
    if "India" in region_choice:
        map_df = map_df[map_df["longitude"] > 0]
        default_zoom = 4
    elif "US" in region_choice:
        map_df = map_df[map_df["longitude"] < 0]
        default_zoom = 3
    else:
        default_zoom = 2

    # Stats banner right above map
    mapped_count = len(map_df)
    m_crit = (map_df["status"] == "CRITICAL").sum()
    m_warn = (map_df["status"] == "WARNING").sum()
    m_heal = (map_df["status"] == "HEALTHY").sum()

    st.markdown(f"""
    <div style="background: #1e293b; border: 1px solid #334155; padding: 8px 14px; border-radius: 8px; margin-bottom: 12px; font-size: 0.88rem; display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap;">
        <div>📍 Active Plotted Markers: <b>{mapped_count}</b> vehicles</div>
        <div>
            <span style="color: #ef4444; font-weight: 700; margin-right: 14px;">🔴 Critical: {m_crit}</span>
            <span style="color: #f59e0b; font-weight: 700; margin-right: 14px;">🟠 Warning: {m_warn}</span>
            <span style="color: #10b981; font-weight: 700;">🟢 Healthy: {m_heal}</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

    if not map_df.empty and "latitude" in map_df.columns and "longitude" in map_df.columns:
        # Standard Streamlit Map with status color and size mapping
        st.map(
            map_df,
            latitude="latitude",
            longitude="longitude",
            color="map_color",
            size="map_size",
            zoom=default_zoom
        )

        with st.expander("📍 View Hub Distribution & Status Breakdown", expanded=False):
            hub_summary = map_df.groupby("base_hub").agg(
                Total_Vehicles=("vehicle_id", "count"),
                Critical=("status", lambda s: (s == "CRITICAL").sum()),
                Warning=("status", lambda s: (s == "WARNING").sum()),
                Healthy=("status", lambda s: (s == "HEALTHY").sum()),
                Avg_Risk=("failure_probability", lambda p: f"{p.mean()*100:.1f}%")
            ).reset_index()
            st.dataframe(hub_summary, hide_index=True, width="stretch")
    else:
        st.info("No vehicles match the selected filters for map plotting.")

    st.markdown("<br>", unsafe_allow_html=True)

    # 4. Main Fleet Health & Component Failure Table
    st.subheader("📋 Fleet Component Failure Status & Predictions")

    # Format configuration display
    if "configuration_display" not in display_df.columns:
        display_df["configuration_display"] = display_df.apply(
            lambda r: f"⚠️ UNKNOWN ({r.get('electrical_system', '')})" if r.get("configuration_status") == "UNKNOWN_CONFIGURATION"
            else f"Profile {r.get('profile', 'A')} ({r.get('electrical_system', '12V')})",
            axis=1
        )

    table_cols = [
        "vehicle_id", "brand", "model", "vehicle_year", "configuration_display", "base_hub",
        "status_display", "failure_probability", "predicted_failure_type",
        "data_quality_display",
        "vehicle_speed_kph", "engine_temp_c", "oil_pressure_psi", "brake_pad_wear_mm", "battery_voltage_v"
    ]

    # Ensure all table columns exist
    for col in table_cols:
        if col not in display_df.columns:
            display_df[col] = 0.0

    sorted_table = display_df.sort_values(by="failure_probability", ascending=False)[table_cols]

    st.dataframe(
        sorted_table,
        column_config={
            "vehicle_id": "Vehicle ID",
            "brand": "Brand",
            "model": "Model",
            "vehicle_year": "Year",
            "configuration_display": "Configuration",
            "base_hub": "Operational Hub",
            "status_display": "Health Status",
            "failure_probability": st.column_config.ProgressColumn(
                "Failure Probability",
                format="%.0f%%",
                min_value=0.0,
                max_value=1.0,
            ),
            "predicted_failure_type": "Predicted Failure Type",
            "data_quality_display": "Data Quality",
            "vehicle_speed_kph": st.column_config.NumberColumn("Speed (km/h)", format="%.1f"),
            "engine_temp_c": st.column_config.NumberColumn("Engine Temp (°C)", format="%.1f"),
            "oil_pressure_psi": st.column_config.NumberColumn("Oil Pressure (PSI)", format="%.1f"),
            "brake_pad_wear_mm": st.column_config.NumberColumn("Brake Pad (mm)", format="%.2f"),
            "battery_voltage_v": st.column_config.NumberColumn("Battery (V)", format="%.2f")
        },
        width="stretch",
        hide_index=True
    )

# Render live dashboard fragment
render_live_telematics_dashboard(chosen_brand, chosen_status, chosen_hub, auto_refresh, auto_simulate)

st.markdown("---")

# 5. Live Sensor Threshold Monitor
st.subheader("📈 Live Telemetry Threshold Monitor")
st.caption("Plots incoming live telemetry only. The plot changes colour when a reading enters warning or critical territory.")

LIVE_SENSOR_CONFIG = {
    "engine_temp_c": {"label": "Engine Temperature (°C)", "warning": 100.0, "critical": 110.0, "direction": "high"},
    "brake_pad_wear_mm": {"label": "Brake Pad Wear (mm)", "warning": 5.0, "critical": 3.0, "direction": "low"},
}

def _live_threshold_state(sensor, value, electrical_system="12V"):
    if pd.isna(value):
        return "NO DATA"
    if sensor == "battery_voltage_v":
        is_24v = "24" in str(electrical_system)
        low_critical, low_warning = (23.6, 24.4) if is_24v else (11.8, 12.2)
        high_warning, high_critical = (28.8, 29.6) if is_24v else (14.7, 15.0)
        if value <= low_critical or value >= high_critical:
            return "CRITICAL"
        if value <= low_warning or value >= high_warning:
            return "WARNING"
        return "NORMAL"
    cfg = LIVE_SENSOR_CONFIG[sensor]
    if cfg["direction"] == "high":
        if value >= cfg["critical"]: return "CRITICAL"
        if value >= cfg["warning"]: return "WARNING"
    else:
        if value <= cfg["critical"]: return "CRITICAL"
        if value <= cfg["warning"]: return "WARNING"
    return "NORMAL"

@st.fragment(run_every=run_interval)
def render_live_threshold_monitor(filter_brand: str, filter_status: str, filter_hub: str):
    live_df, _, _ = load_fleet_data()
    if live_df.empty:
        st.info("Live sensor plots will appear when telemetry is available. Enable Live Auto-Refresh to watch readings update automatically.")
        return

    # Keep this monitor synchronized with the dashboard sidebar filters.
    filtered_df = live_df.copy()
    if filter_brand != "All Brands" and "brand" in filtered_df.columns:
        filtered_df = filtered_df[filtered_df["brand"] == filter_brand]
    if filter_status != "All Statuses" and "status" in filtered_df.columns:
        filtered_df = filtered_df[filtered_df["status"] == filter_status]
    if filter_hub != "All Hubs" and "base_hub" in filtered_df.columns:
        filtered_df = filtered_df[filtered_df["base_hub"] == filter_hub]

    if filtered_df.empty:
        st.info("No vehicles match the selected brand, status, and hub filters. Update the sidebar filters to view sensor graphs.")
        return

    sensor_configs = {
        "engine_temp_c": {"label": "Engine Temperature (°C)", "unit": "°C"},
        "battery_voltage_v": {"label": "Battery Voltage (V)", "unit": "V"},
        "brake_pad_wear_mm": {"label": "Brake Pad Wear (mm)", "unit": "mm"},
    }
    sensors = [
        sensor for sensor in sensor_configs
        if sensor in filtered_df.columns
        and pd.to_numeric(filtered_df[sensor], errors="coerce").notna().any()
    ]
    if not sensors:
        st.warning("No numeric engine temperature, battery voltage, or brake pad wear readings were found in the latest telemetry.")
        return

    vehicles = sorted(filtered_df["vehicle_id"].dropna().astype(str).unique()) if "vehicle_id" in filtered_df else []
    # If a sidebar filter removes the previously selected vehicle, choose the first
    # available vehicle so the selector and all three charts stay in sync.
    current_selection = st.session_state.get("live_threshold_vehicle")
    if vehicles and current_selection not in vehicles:
        st.session_state["live_threshold_vehicle"] = vehicles[0]
    selected_vehicle = st.selectbox(
        "Vehicle to monitor",
        vehicles,
        key="live_threshold_vehicle",
    ) if vehicles else None

    if selected_vehicle is None:
        st.info("No vehicle IDs are available in the current telemetry.")
        return

    source = filtered_df[filtered_df["vehicle_id"].astype(str) == selected_vehicle]
    if source.empty:
        st.info("No current reading is available for this vehicle.")
        return

    vehicle_row = source.iloc[0]
    electrical = str(vehicle_row.get("electrical_system", "12V"))
    colors = {"NORMAL": "#16a34a", "WARNING": "#f59e0b", "CRITICAL": "#dc2626", "NO DATA": "#64748b"}

    # Display all three sensor charts for the selected vehicle at the same time.
    chart_columns = st.columns(3)
    for index, sensor in enumerate(sensor_configs):
        cfg_label = sensor_configs[sensor]["label"]
        unit = sensor_configs[sensor]["unit"]
        with chart_columns[index]:
            st.markdown(f"**{cfg_label}**")
            if sensor not in source.columns:
                st.info("Sensor data unavailable.")
                continue

            value = pd.to_numeric(source[sensor], errors="coerce").mean()
            state = _live_threshold_state(sensor, value, electrical)
            st.metric("Latest reading", "—" if pd.isna(value) else f"{value:.2f} {unit}")
            st.caption(f"Status: {state}")

            key = f"live_plot_{selected_vehicle}_{sensor}"
            points = st.session_state.get(key, [])
            if not pd.isna(value):
                points.append({
                    "time": datetime.datetime.now(datetime.timezone.utc),
                    "value": float(value),
                })
                points = points[-120:]
                st.session_state[key] = points

            if sensor == "battery_voltage_v":
                is24 = "24" in electrical
                wl, cl = (24.4, 23.6) if is24 else (12.2, 11.8)
                wh, ch = (28.8, 29.6) if is24 else (14.7, 15.0)
                thresholds = [
                    (wl, "Warning low"), (cl, "Critical low"),
                    (wh, "Warning high"), (ch, "Critical high"),
                ]
            else:
                cfg = LIVE_SENSOR_CONFIG[sensor]
                thresholds = [(cfg["warning"], "Warning threshold"),
                              (cfg["critical"], "Critical threshold")]

            if not points:
                st.info("Waiting for the next numeric reading.")
                continue

            chart = pd.DataFrame(points).sort_values("time")
            if go is not None:
                fig = go.Figure()
                for i, row in chart.iterrows():
                    point_state = _live_threshold_state(sensor, row["value"], electrical)
                    fig.add_trace(go.Scatter(
                        x=[row["time"]], y=[row["value"]], mode="markers",
                        marker={"color": colors[point_state], "size": 7},
                        name=point_state,
                        showlegend=point_state not in [trace.name for trace in fig.data],
                        hovertemplate="Time: %{x}<br>Value: %{y:.2f}<extra>" + point_state + "</extra>",
                    ))
                    if i > 0:
                        prev = chart.iloc[i - 1]
                        fig.add_trace(go.Scatter(
                            x=[prev["time"], row["time"]],
                            y=[prev["value"], row["value"]],
                            mode="lines",
                            line={"color": colors[point_state], "width": 2},
                            showlegend=False,
                            hoverinfo="skip",
                        ))
                for limit, label in thresholds:
                    fig.add_hline(
                        y=limit, line_dash="dash",
                        line_color="#f59e0b" if "Warning" in label else "#dc2626",
                        annotation_text=label,
                    )
                fig.update_layout(
                    height=330,
                    margin={"l": 8, "r": 8, "t": 25, "b": 8},
                    xaxis_title="Time",
                    yaxis_title=cfg_label,
                    legend={"orientation": "h", "yanchor": "bottom", "y": 1.02, "xanchor": "left", "x": 0},
                )
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.line_chart(chart.set_index("time"), y="value", x_label="Time")

render_live_threshold_monitor(chosen_brand, chosen_status, chosen_hub)

st.markdown("---")

# 6. AI-Generated Maintenance Report Section
st.subheader("📑 AI Maintenance Report Generator")
st.caption("Select any vehicle from the fleet to view its instant diagnostic work order:")

if not df_fleet_init.empty:
    veh_list = sorted(list(df_fleet_init["vehicle_id"].unique()))
    selected_vid = st.selectbox("Select Vehicle ID", veh_list, index=0, key="maintenance_vehicle_id_select")

    if selected_vid:
        # Fetch current latest state of this vehicle
        df_current, _, _ = load_fleet_data()
        v_match = df_current[df_current["vehicle_id"] == selected_vid]
        v_info = v_match.iloc[0] if not v_match.empty else df_fleet_init[df_fleet_init["vehicle_id"] == selected_vid].iloc[0]
        
        rep_gen = MaintenanceReportGenerator()
        report_text = rep_gen.generate_report(selected_vid, v_info.to_dict(), v_info.to_dict())
        
        col_rep_left, col_rep_right = st.columns([3, 1])
        
        with col_rep_left:
            st.text_area("Generated Diagnostic Work Order", report_text, height=340)
            
        with col_rep_right:
            st.markdown(f"**Vehicle:** `{selected_vid}`")
            st.markdown(f"**Model:** {v_info.get('vehicle_year', '')} {v_info.get('brand', '')} {v_info.get('model', '')}")
            st.markdown(f"**Configuration:** Profile {v_info.get('profile', 'A')} | {v_info.get('electrical_system', '12V')} | {v_info.get('brake_configuration', 'Hydraulic Disc')}")
            st.markdown(f"**Base Hub:** {v_info.get('base_hub', 'N/A')}")
            st.markdown(f"**Failure Risk:** `{v_info['failure_probability']*100:.0f}%`")
            st.markdown(f"**Predicted Fault:** `{v_info['predicted_failure_type']}`")
            
            if v_info.get("configuration_status") == "UNKNOWN_CONFIGURATION":
                st.error("🚨 Configuration Warning: Unknown electrical or brake system detected! Standard thresholds suspended.")
            
            dq_stat = v_info.get("data_quality_status", "HIGH")
            st.markdown(f"**Telemetry Quality:** `{dq_stat}`")
            if v_info.get("imputed_count", 0) > 0:
                imp_list = v_info.get("imputed_sensors", [])
                st.warning(f"⚠️ {v_info.get('imputed_count')} sensor(s) causally imputed from history: {', '.join(imp_list[:3])}")
            if v_info.get("sensor_errors"):
                st.error(f"🚨 Sensor error flagged: {', '.join(v_info.get('sensor_errors'))}")
            
            st.download_button(
                label="📥 Download Work Order (.txt)",
                data=report_text,
                file_name=f"Maintenance_Work_Order_{selected_vid}.txt",
                mime="text/plain",
                width="stretch"
            )
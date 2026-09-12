import os
import tempfile
import cv2
import numpy as np
import pandas as pd
import rasterio
import streamlit as st
import xarray as xr

# --- SEAMLESS DATA ADAPTERS ---
def load_metocean_data(file_path):
    ds = xr.open_dataset(file_path)
    var_map = {
        "u10": "u_wind", "v10": "v_wind",
        "u_surface": "u_current", "v_surface": "v_current",
        "eastward_wind": "u_wind", "northward_wind": "v_wind",
        "uo": "u_current", "vo": "v_current",
    }
    for raw_key, standard_key in var_map.items():
        if raw_key in ds.data_vars and standard_key not in ds.data_vars:
            ds = ds.rename({raw_key: standard_key})
    
    for fallback in ["u_current", "v_current", "u_wind", "v_wind"]:
        if fallback not in ds.data_vars:
            ds[fallback] = xr.zeros_like(ds[list(ds.data_vars.keys())[0]])
    return ds

def load_ais_data(file_path, spill_x, spill_y):
    df = pd.read_csv(file_path)
    col_map = {
        "LAT": "latitude", "Latitude": "latitude", "lat": "latitude",
        "LON": "longitude", "Longitude": "longitude", "lon": "longitude",
        "BaseDateTime": "timestamp", "Timestamp": "timestamp",
        "SOG": "SOG_knots", "Speed": "SOG_knots", "VesselName": "VesselName",
    }
    df = df.rename(columns=col_map)
    if "X" not in df.columns and "latitude" in df.columns:
        ref_lat = 24.5
        df["X"] = spill_x + ((df["longitude"] - df["longitude"].mean()) * (111000 * np.cos(np.radians(ref_lat))))
        df["Y"] = spill_y + ((df["latitude"] - df["latitude"].mean()) * 111000)
    if "SOG_knots" not in df.columns:
        df["SOG_knots"] = 10.0
    return df

# --- SIDEBAR TOGGLE ---
st.sidebar.header("⚙️ Data Environment")
data_source = st.sidebar.radio(
    "Select Dataset",
    ["Simulated Scenario (Dynamic)", "Real Dataset (Copernicus / Marine Cadastre)"]
)

if data_source == "Real Dataset (Copernicus / Marine Cadastre)":
    metocean_path = "real_era5.nc"
    ais_path = "real_marine_cadastre.csv"
else:
    metocean_path = "ocean_currents.nc"
    ais_path = "marine_cadastre_ais.csv"

st.set_page_config(page_title="Oil Spill Tracker", page_icon="🌊", layout="wide")

st.title("🌊 Maritime Oil Spill Detection & Vessel Tracking System")
st.caption("Production GIS Pipeline — Real GeoTIFF Spatial Extraction")

tab1, tab2, tab3 = st.tabs(["1. GeoTIFF Detection", "2. Drift Simulation", "3. Suspect Match"])

with tab1:
    st.subheader("1. Satellite GeoTIFF Analysis")
    uploaded_file = st.file_uploader("Upload Real Satellite GeoTIFF (.tif)", type=["tif", "tiff"])
    
    if uploaded_file is not None:
        # Save uploaded buffer to a temp file so rasterio can read spatial headers
        with tempfile.NamedTemporaryFile(delete=False, suffix=".tif") as tmp_file:
            tmp_file.write(uploaded_file.read())
            tmp_path = tmp_file.name

        try:
            with rasterio.open(tmp_path) as dataset:
                # Read Band 1 for image processing
                band1 = dataset.read(1)
                
                # Normalize band data to 8-bit for OpenCV thresholding
                band1_norm = cv2.normalize(band1, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
                
                col1, col2 = st.columns(2)
                with col1:
                    st.image(band1_norm, caption=f"Raw Band 1 ({dataset.width}x{dataset.height} px)", use_container_width=True)

                # OpenCV Noise Reduction & Dark Patch Extraction
                blurred = cv2.GaussianBlur(band1_norm, (5, 5), 0)
                _, binary_mask = cv2.threshold(blurred, 60, 255, cv2.THRESH_BINARY_INV)
                
                with col2:
                    st.image(binary_mask, caption="Processed Oil Spill Mask", use_container_width=True)

                # Extract center pixel of largest dark patch
                contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                
                if contours:
                    largest_spill = max(contours, key=cv2.contourArea)
                    M = cv2.moments(largest_spill)
                    if M["m00"] != 0:
                        px = int(M["m10"] / M["m00"])
                        py = int(M["m01"] / M["m00"])

                        # REAL RASTERIO SPATIAL TRANSFORM
                        real_x, real_y = dataset.xy(py, px)

                        # Store values across tabs
                        st.session_state['real_x'] = real_x
                        st.session_state['real_y'] = real_y
                        st.session_state['crs'] = str(dataset.crs)

                        st.success(f"🎯 Real Spatial Coordinates Extracted! Pixel: ({px}, {py}) | Coordinates: ({real_x:.2f}, {real_y:.2f}) | CRS: {dataset.crs}")

        finally:
            # Clean up temp file
            if os.path.exists(tmp_path):
                os.remove(tmp_path)

import xarray as xr

import matplotlib.pyplot as plt
import xarray as xr

with tab2:
    st.subheader("2. Hydrodynamic Drift Prediction (INCOIS / ERA5 NetCDF Engine)")

    # Guard clause: Ensure Tab 1 has processed a GeoTIFF image
    if "real_x" not in st.session_state or "real_y" not in st.session_state:
        st.warning(
            "⚠️ No spatial data detected. Please upload a GeoTIFF in **Tab 1** first to extract spill coordinates!"
        )
    else:
        rx = st.session_state["real_x"]
        ry = st.session_state["real_y"]

        try:
            # 1. Load Metocean Data dynamically (via Adapter)
            ds = load_metocean_data(metocean_path)

            # Extract average vectors for UI metric cards
            ds_sub = ds.sel(x=rx, y=ry, method="nearest")
            u_c = float(ds_sub["u_current"].mean())
            v_c = float(ds_sub["v_current"].mean())
            u_w = float(ds_sub["u_wind"].mean())
            v_w = float(ds_sub["v_wind"].mean())

            # Display key vector metrics (KEPT)
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric(
                    "🌊 Ocean Current (u, v)", f"{u_c:.2f} m/s E, {v_c:.2f} m/s N"
                )
            with col2:
                st.metric("💨 Surface Wind (u, v)", f"{u_w:.1f} m/s E, {v_w:.1f} m/s N")
            with col3:
                net_u_avg = u_c + (0.03 * u_w)
                net_v_avg = v_c + (0.03 * v_w)
                net_speed = np.sqrt(net_u_avg**2 + net_v_avg**2)
                st.metric("⚡ Combined Drift Speed", f"{net_speed:.2f} m/s")

            st.markdown("---")

            # Interactive timeline slider & mode selector (KEPT)
            hours = st.slider("⏱️ Forecast/Hindcast Timeline (Hours)", 1, 48, 24)
            drift_mode = st.radio(
                "🔄 Drift Engine Mode",
                ["Forecasting (Future Drift)", "Hindcasting (Trace Back to Origin)"],
                horizontal=True,
            )
            direction = -1.0 if "Hindcasting" in drift_mode else 1.0

            # 2. DYNAMIC TIME-VARYING TRAJECTORY ENGINE (Curved Path)
            trajectory = []
            curr_x, curr_y = rx, ry

            # Determine available time steps in NetCDF
            max_steps = min(hours + 1, len(ds_sub["time"]))

            for h in range(max_steps):
                # Pull 1D vector arrays across time explicitly
                u_curr_series = ds_sub["u_current"].values
            v_curr_series = ds_sub["v_current"].values
            u_wind_series = ds_sub["u_wind"].values
            v_wind_series = ds_sub["v_wind"].values

            trajectory = []
            curr_x, curr_y = rx, ry
            max_steps = min(hours + 1, len(u_curr_series))

            for h in range(max_steps):
                # Pull scalar for hour h
                u_c_h = float(u_curr_series[h])
                v_c_h = float(v_curr_series[h])
                u_w_h = float(u_wind_series[h])
                v_w_h = float(v_wind_series[h])

                # Combined vector calculation
                net_u = u_c_h + (0.03 * u_w_h)
                net_v = v_c_h + (0.03 * v_w_h)

                step_dx = net_u * 3600 * direction
                step_dy = net_v * 3600 * direction

                if h == 0:
                    dist_km = 0.0
                else:
                    curr_x += step_dx
                    curr_y += step_dy
                    dist_km = np.sqrt((curr_x - rx)**2 + (curr_y - ry)**2) / 1000.0

                trajectory.append({
                    "Hour": -h if direction < 0 else h,
                    "Predicted_X": curr_x,
                    "Predicted_Y": curr_y,
                    "Drift_Distance_km": dist_km
                })

            traj_df = pd.DataFrame(trajectory)

            # --- VISUALIZATION BLOCK ---
            col_left, col_right = st.columns([3, 2])

            with col_left:
                st.markdown("### 🗺️ 2D Spatial Particle Drift Path")

                fig, ax = plt.subplots(figsize=(7, 5))
                fig.patch.set_facecolor("#0e1117")
                ax.set_facecolor("#161b22")

                # Plot path line
                ax.plot(
                    traj_df["Predicted_X"],
                    traj_df["Predicted_Y"],
                    color="#00d4ff",
                    linestyle="--",
                    linewidth=2,
                    label="Drift Trajectory",
                )

                # Mark Origin Point
                ax.scatter(
                    rx,
                    ry,
                    color="#ff4b4b",
                    s=120,
                    zorder=5,
                    label="Spill Origin (0h)",
                )

                # Mark Projected Endpoint
                end_x = traj_df.iloc[-1]["Predicted_X"]
                end_y = traj_df.iloc[-1]["Predicted_Y"]
                ax.scatter(
                    end_x,
                    end_y,
                    color="#ffaa00",
                    s=120,
                    zorder=5,
                    label=f"Predicted ({hours}h)",
                )

                # Annotate End Point
                ax.annotate(
                    f"+{hours}h ({traj_df.iloc[-1]['Drift_Distance_km']:.1f} km)",
                    (end_x, end_y),
                    textcoords="offset points",
                    xytext=(10, 10),
                    ha="left",
                    color="#ffffff",
                    fontsize=9,
                    weight="bold",
                )

                # Styling
                ax.set_xlabel("Spatial X (UTM Easting)", color="white")
                ax.set_ylabel("Spatial Y (UTM Northing)", color="white")
                ax.tick_params(colors="white")
                for spine in ax.spines.values():
                    spine.set_color("#30363d")
                ax.grid(True, linestyle=":", color="#30363d", alpha=0.6)
                ax.legend(
                    facecolor="#0e1117", edgecolor="#30363d", labelcolor="white"
                )

                st.pyplot(fig)

            with col_right:
                st.markdown("### 📊 Forecast Data")

                # Summary Callout
                total_drift = traj_df.iloc[-1]["Drift_Distance_km"]
                st.warning(
                    f"⚠️ **Slick Dispersion:** Predicted to travel **{total_drift:.2f} km** over **{hours} hours**."
                )

                # Display compact table
                st.dataframe(
                    traj_df[["Hour", "Predicted_X", "Predicted_Y", "Drift_Distance_km"]].rename(
                        columns={"Drift_Distance_km": "Drift (km)"}
                    ),
                    height=320,
                    use_container_width=True,
                )

        except FileNotFoundError:
            st.error(
                "Missing 'ocean_currents.nc'. Run `python generate_metocean.py` first!"
            )


with tab3:
    st.subheader("3. Spatio-Temporal AIS Vessel Attribution & Anomaly Ranking")

    # Check if Tab 1 has generated spatial coordinates
    if "real_x" not in st.session_state or "real_y" not in st.session_state:
        st.warning(
            "⚠️ No spatial data detected. Please upload a GeoTIFF in **Tab 1** first to extract spill coordinates!"
        )
    else:
        rx = st.session_state["real_x"]
        ry = st.session_state["real_y"]

        # ADAPTIVE AIS SCORING ENGINE
        try:
            ais_df = load_ais_data(ais_path, rx, ry)

            # Calculate Spatial Distance
            ais_df["Distance_m"] = np.sqrt((ais_df["X"] - rx) ** 2 + (ais_df["Y"] - ry) ** 2)

            # 60/40 Weighted MCDA Scoring Logic
            ais_df["Proximity_Score"] = ais_df["Distance_m"].apply(lambda d: max(0, 100 - (d / 100.0)))
            ais_df["Behavior_Score"] = ais_df["SOG_knots"].apply(lambda sog: 100.0 if sog < 3.0 else 20.0)
            ais_df["Suspect_Risk_%"] = (0.6 * ais_df["Proximity_Score"]) + (0.4 * ais_df["Behavior_Score"])

            ais_df = ais_df.sort_values(by="Suspect_Risk_%", ascending=False)

            # Prime Suspect Callout Banner
            top_vessel = ais_df.iloc[0]
            if top_vessel["Suspect_Risk_%"] > 70:
                st.error(
                    f"🚨 **PRIME SUSPECT IDENTIFIED:** **{top_vessel['VesselName']}**\n\n"
                    f"• **Risk Score:** {top_vessel['Suspect_Risk_%']:.1f}%\n"
                    f"• **Distance to Origin:** {top_vessel['Distance_m']:.0f} meters\n"
                    f"• **Speed Anomaly:** {top_vessel['SOG_knots']} knots (Low-speed loitering detected)"
                )

            st.markdown("### 📋 Multi-Factor Suspect Vessel Ranking")
            st.dataframe(
                ais_df[["VesselName", "MMSI", "SOG_knots", "Distance_m", "Suspect_Risk_%"]].rename(
                    columns={
                        "SOG_knots": "Speed (kts)",
                        "Distance_m": "Distance (m)",
                        "Suspect_Risk_%": "Risk Score (%)",
                    }
                ),
                use_container_width=True,
            )

        except FileNotFoundError:
            st.error(f"Missing '{ais_path}'. Run generator scripts first!")
import numpy as np
import pandas as pd
import xarray as xr

# 1. Spatial & Temporal Dimensions
x = np.linspace(200000, 240000, 50)
y = np.linspace(2700000, 2740000, 50)
time = pd.date_range("2026-09-11 00:00", periods=48, freq="h")

# 2. Build 3D Grids (time x y x)
T, Y, X = np.meshgrid(np.arange(len(time)), y, x, indexing="ij")

# 3. HIGH-VARIANCE VECTOR MATH (Strong Tidal Oscillations + Shifting Winds)
# Strong sine/cosine swings force the trajectory to curve/S-shape noticeably
u_current = 0.15 + 0.60 * np.sin(2 * np.pi * T / 12.0)  # Strong 12h tidal shift
v_current = 0.10 + 0.55 * np.cos(2 * np.pi * T / 12.0)
u_wind = 3.0 + 4.0 * np.sin(2 * np.pi * T / 24.0)       # Diurnal wind swing
v_wind = 1.0 + 3.5 * np.cos(2 * np.pi * T / 24.0)

# 4. Construct Dataset
ds = xr.Dataset(
    data_vars={
        "u_current": (("time", "y", "x"), u_current),
        "v_current": (("time", "y", "x"), v_current),
        "u_wind": (("time", "y", "x"), u_wind),
        "v_wind": (("time", "y", "x"), v_wind),
    },
    coords={"time": time, "y": y, "x": x},
    attrs={"description": "High-Variance Oceanographic Grid"},
)

ds.to_netcdf("ocean_currents.nc", mode="w")
print("✅ Created high-variance ocean_currents.nc!")
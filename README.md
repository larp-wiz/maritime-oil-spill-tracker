# 🌊 Maritime Oil Spill Detection & Vessel Tracking System

An end-to-end GIS maritime intelligence platform built for the **Smart India Hackathon (SIH)**. 

It parses Sentinel-1 SAR satellite GeoTIFF files, models hydrodynamic oil slick dispersion using NetCDF oceanographic data, and performs spatial proximity matches against Marine Cadastre AIS vessel logs.

---

## 🛠️ Tech Stack & Architecture
- **Language & UI:** Python, Streamlit
- **GIS & Satellite Parsing:** `rasterio`, OpenCV
- **Hydrodynamic Modeling:** `xarray`, NetCDF4, Matplotlib
- **Vessel Tracking:** `pandas` (Marine Cadastre AIS Format)

---

## 🚀 Key Modules
1. **GeoTIFF Spatial Extraction:** Reads native `.tif` map headers (CRS, UTM bounds) and isolates oil slick center pixels using computer vision thresholding.
2. **Hydrodynamic Drift Engine:** Parses $u, v$ ocean current and surface wind vector arrays from `.nc` weather files to model Lagrangian particle dispersion over a 48-hour timeline.
3. **Marine Cadastre AIS Matcher:** Runs Euclidean spatial proximity queries against historical vessel logs to isolate high-risk suspect vessels near the spill origin.

---

## 💻 How to Run Locally

```bash
# 1. Clone repository
git clone [https://github.com/YOUR_USERNAME/maritime-oil-spill-tracker.git](https://github.com/YOUR_USERNAME/maritime-oil-spill-tracker.git)
cd maritime-oil-spill-tracker

# 2. Install dependencies
pip install streamlit rasterio OpenCV-python xarray netcdf4 pandas matplotlib

# 3. Generate mock spatial databases
python generate_metocean.py
python generate_ais.py

# 4. Launch Streamlit Dashboard
streamlit run app.py
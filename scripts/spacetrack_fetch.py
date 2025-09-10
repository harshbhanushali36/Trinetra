"""
spacetrack_to_czml.py

Requirements:
    pip install requests pandas openpyxl sgp4 numpy tqdm

What it does:
    1) Logs into Space-Track, fetches gp catalog JSON (your working fetch).
    2) Saves Excel with multiple sheets (raw & filtered).
    3) Uses TLE (TLE_LINE1 / TLE_LINE2) to propagate object positions with sgp4.
    4) Converts TEME positions to approximate ECEF using GMST rotation.
    5) Exports a CZML file where each object has a 'position' with epoch (IST offset)
       and cartesian coordinates at each time step for visualization in Cesium.
"""

import requests
import pandas as pd
import json
from sgp4.api import Satrec, jday
import numpy as np
from datetime import datetime, timedelta, timezone
import math
from tqdm import tqdm
import sys
import os

# -----------------------
# User / runtime params
# -----------------------
USERNAME = "harshbhanushali36@gmail.com"
PASSWORD = "Trinetra_12345678"
LOGIN_URL = "https://www.space-track.org/ajaxauth/login"
CATALOG_URL = "https://www.space-track.org/basicspacedata/query/class/gp/format/json"

# Output filenames
OUTPUT_EXCEL = "spacetrack_data.xlsx"
OUTPUT_CZML = "orbital_objects.czml"

# Propagation window (change as needed)
DURATION_HOURS = 1           # how many hours from start to propagate
STEP_SECONDS = 60            # time-step in seconds between samples

# Filters: keep as-is or change
TLE_LINE1_FIELD = "TLE_LINE1"
TLE_LINE2_FIELD = "TLE_LINE2"
TYPE_FIELD = "OBJECT_TYPE"

# Only include object types we want in CZML (to reduce size set to PAYLOAD/ROCKET/DEBRIS)
WANTED_TYPES = {"PAYLOAD", "ROCKET BODY", "DEBRIS"}

# -----------------------
# Utility functions
# -----------------------
def gmst_from_jd(jd_ut1):
    """
    Compute Greenwich Mean Sidereal Time (radians) from Julian date (UT1).
    Uses approximate IAU 1982 expression (Meeus).
    Returns angle in radians in [0, 2*pi).
    """
    # Reference epoch J2000
    T = (jd_ut1 - 2451545.0) / 36525.0
    # GMST in seconds
    gmst_sec = 67310.54841 + (876600.0 * 3600 + 8640184.812866) * T \
               + 0.093104 * T**2 - 6.2e-6 * T**3
    # reduce to range 0..86400
    gmst_sec = gmst_sec % 86400.0
    gmst_rad = (gmst_sec / 86400.0) * 2.0 * math.pi
    return gmst_rad

def rotate_z(vec, angle_rad):
    """Rotate 3-vector about Z axis by angle_rad (right-hand rule)."""
    c = math.cos(angle_rad)
    s = math.sin(angle_rad)
    x, y, z = vec
    xr = c*x - s*y
    yr = s*x + c*y
    zr = z
    return (xr, yr, zr)

def teme_to_ecef_km(r_teme_km, dt_utc):
    """
    Convert TEME position (km) to an approximate ECEF position (km) using GMST rotation.
    dt_utc should be a datetime in UTC.
    This is an approximation adequate for visualization.
    """
    # Compute Julian date for dt_utc
    jd, jd_frac = jday(dt_utc.year, dt_utc.month, dt_utc.day,
                       dt_utc.hour, dt_utc.minute, dt_utc.second + dt_utc.microsecond*1e-6)
    jd_ut1 = jd + jd_frac
    gmst = gmst_from_jd(jd_ut1)  # radians
    # Rotate TEME -> PEF/ECEF approx by GMST
    # Many references use rotation by +GMST: r_ecef = R3(gmst) * r_teme
    return rotate_z(r_teme_km, gmst)

def datetime_to_iso_with_ist_offset(dt_utc):
    """
    Return an ISO string representing dt_utc but with IST offset (+05:30).
    Example: "2025-09-10T18:30:00+05:30"
    Cesium/CZML can accept ISO with offset.
    """
    ist_offset = timezone(timedelta(hours=5, minutes=30))
    dt_ist = dt_utc.astimezone(ist_offset)
    # Use isoformat() which includes offset
    return dt_ist.isoformat()

# -----------------------
# Fetch & Save (your working fetch)
# -----------------------
session = requests.Session()
login_payload = {"identity": USERNAME, "password": PASSWORD}
resp = session.post(LOGIN_URL, data=login_payload)
resp.raise_for_status()
print("✅ Logged into Space-Track successfully!")

response = session.get(CATALOG_URL)
response.raise_for_status()
data = response.json()

# Convert to DataFrame
df = pd.DataFrame(data)

# Rename columns to be clear & consistent (keeps your mapping)
rename_map = {
    "OBJECT_NAME": "Name",
    "OBJECT_ID": "ID",
    "OBJECT_TYPE": "Type",
    "LAUNCH_DATE": "Launch Date",
    "PERIOD": "Period (min)",
    "INCLINATION": "Inclination (deg)",
    "APOGEE": "Apogee (km)",
    "PERIGEE": "Perigee (km)",
    "ECCENTRICITY": "Eccentricity",
    "MEAN_MOTION": "Mean Motion",
    "RA_OF_ASC_NODE": "RAAN (deg)",
    "ARG_OF_PERICENTER": "Arg Perigee (deg)",
    "TLE_LINE1": "TLE Line 1",
    "TLE_LINE2": "TLE Line 2"
}
df = df.rename(columns=rename_map)
columns_to_keep = list(rename_map.values())
df_filtered = df[[col for col in columns_to_keep if col in df.columns]]

df_satellites = df_filtered[df_filtered.get("Type") == "PAYLOAD"]
df_rockets = df_filtered[df_filtered.get("Type") == "ROCKET BODY"]
df_debris = df_filtered[df_filtered.get("Type") == "DEBRIS"]

# Save to Excel
with pd.ExcelWriter(OUTPUT_EXCEL, engine="openpyxl") as writer:
    df.to_excel(writer, sheet_name="Raw Data", index=False)
    df_filtered.to_excel(writer, sheet_name="Filtered Data", index=False)
    df_satellites.to_excel(writer, sheet_name="Satellites", index=False)
    df_rockets.to_excel(writer, sheet_name="Rockets", index=False)
    df_debris.to_excel(writer, sheet_name="Debris", index=False)

print(f"✅ Data saved to {OUTPUT_EXCEL} with multiple sheets including TLEs!")

# -----------------------
# Build CZML
# -----------------------

# Time window setup: start now (UTC)
start_utc = datetime.now(timezone.utc)
end_utc = start_utc + timedelta(hours=DURATION_HOURS)

# We'll set CZML epoch as an ISO with IST offset, per your request.
czml_epoch_iso_ist = datetime_to_iso_with_ist_offset(start_utc)

# For CZML, Cesium accepts an 'epoch' with offset ISO; we will also store time offsets in seconds from that epoch.
epoch_for_czml = czml_epoch_iso_ist

print("CZML epoch (IST):", epoch_for_czml)
print(f"Propagating from {start_utc.isoformat()} UTC to {end_utc.isoformat()} UTC "
      f"({DURATION_HOURS} hours) with step {STEP_SECONDS}s")

# Create base CZML array
czml = []

# Document packet
doc_packet = {
    "id": "document",
    "name": "Space-Track Objects (propagated)",
    "version": "1.0",
    "clock": {
        # Set clock interval in UTC for Cesium; Cesium will display local time based on viewer.
        "interval": f"{start_utc.isoformat()}/{end_utc.isoformat()}",
        "currentTime": start_utc.isoformat(),
        "multiplier": 1,
        "range": "LOOP_STOP"
    }
}
czml.append(doc_packet)

# Prepare time samples: list of datetimes and offsets in seconds from epoch (IST-ISO offset is converted to UTC seconds)
num_steps = int((end_utc - start_utc).total_seconds() // STEP_SECONDS) + 1
time_offsets = [i * STEP_SECONDS for i in range(num_steps)]
time_datetimes = [start_utc + timedelta(seconds=off) for off in time_offsets]

# Helper to create position array for CZML given list of (x,y,z) in meters.
def build_czml_cartesian_array(positions_m, offsets_s):
    """
    builds [offset0, x0, y0, z0, offset1, x1, y1, z1, ...]
    CZML allows either absolute ISO timestamps or epoch + offsets — we'll provide epoch + offsets.
    """
    arr = []
    for off, (x, y, z) in zip(offsets_s, positions_m):
        arr.append(off)
        arr.append(float(x))
        arr.append(float(y))
        arr.append(float(z))
    return arr

# Iterate objects and propagate
objects_written = 0
bad_tle_count = 0

# We'll iterate only objects with TLEs present and of wanted types
# If your dataset is huge and you want to restrict (e.g., only PAYLOAD), update WANTED_TYPES
candidates = []
for idx, row in df_filtered.iterrows():
    obj_type = row.get("Type", "")
    tle1 = row.get("TLE Line 1") or row.get("TLE_LINE1") or row.get("TLE Line1")
    tle2 = row.get("TLE Line 2") or row.get("TLE_LINE2") or row.get("TLE Line2")
    if pd.isna(tle1) or pd.isna(tle2):
        continue
    if WANTED_TYPES and obj_type not in WANTED_TYPES:
        continue
    # Use Name or ID or fallback
    name = str(row.get("Name") or row.get("ID") or f"object_{idx}")
    candidates.append((name, tle1.strip(), tle2.strip(), obj_type))

print(f"Found {len(candidates)} objects with TLEs matching types {WANTED_TYPES}.")

# Propagate each candidate - heavy operation for many objects
for name, tle1, tle2, obj_type in tqdm(candidates, desc="Propagating objects", unit="obj"):
    try:
        # Create Satrec from TLE
        sat = Satrec.twoline2rv(tle1, tle2)
    except Exception as e:
        bad_tle_count += 1
        # skip invalid TLE
        continue

    # For each timestep, propagate and convert to ECEF
    pos_ecef_m = []
    valid_any = False
    for dt in time_datetimes:
        # convert dt to jd + fraction used by sgp4.jday interface
        jd, fr = jday(dt.year, dt.month, dt.day, dt.hour, dt.minute, dt.second + dt.microsecond*1e-6)
        e, r, v = sat.sgp4(jd, fr)
        if e != 0:
            # propagation error (e.g., satrec error code) - append NaNs and continue
            pos_ecef_m.append((float('nan'), float('nan'), float('nan')))
            continue
        # r is TEME position in km
        # Convert TEME->ECEF (approx) using gmst-based rotation
        r_ecef_km = teme_to_ecef_km(r, dt)
        # convert to meters
        r_ecef_m = (r_ecef_km[0]*1000.0, r_ecef_km[1]*1000.0, r_ecef_km[2]*1000.0)
        pos_ecef_m.append(r_ecef_m)
        valid_any = True

    if not valid_any:
        bad_tle_count += 1
        continue

    # Build CZML packet for this object
    packet = {
        "id": name.replace(" ", "_")[:64],
        "name": name,
        # show only during availability
        "availability": f"{start_utc.isoformat()}/{end_utc.isoformat()}",
        "properties": {
            "object_type": obj_type
        },
        "position": {
            "epoch": epoch_for_czml,   # IST-offset ISO (Cesium accepts ISO+offset)
            "cartesian": build_czml_cartesian_array(pos_ecef_m, time_offsets)
        },
        # optionally add a billboard or point for visualization:
        "point": {
            "pixelSize": 3,
            "outlineWidth": 0
        }
    }

    czml.append(packet)
    objects_written += 1

print(f"Propagation done. Objects written to CZML: {objects_written}. Skipped/invalid: {bad_tle_count}")

# Write CZML to file
with open(OUTPUT_CZML, "w") as f:
    json.dump(czml, f, indent=2)

print(f"✅ CZML saved to {OUTPUT_CZML} — ready for Cesium visualization.")
print("Tip: Load the CZML in Cesium ion or CesiumJS viewer. The CZML packets use an epoch with IST offset; "
      "Cesium will convert timestamps to viewer time but labels/descriptions will reflect IST epoch.")

# Provide some quick information for user
print("\nSummary:")
print(" - Excel file:", os.path.abspath(OUTPUT_EXCEL))
print(" - CZML file:", os.path.abspath(OUTPUT_CZML))
print(f" - Start (UTC): {start_utc.isoformat()}")
print(f" - CZML epoch (IST): {epoch_for_czml}")
print(f" - Steps: {num_steps} (every {STEP_SECONDS} s for {DURATION_HOURS} hours)")


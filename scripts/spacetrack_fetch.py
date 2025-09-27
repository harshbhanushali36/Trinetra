#!/usr/bin/env python3
#working till 26 sept
"""
spacetrack_to_json_modes.py (Refactored for Trinetra API)

Modes:
 - HIGH_ACCURACY: 1 hr propagation, 60s steps
 - BALANCED: 30 min propagation, 300s steps
"""

import requests
import pandas as pd
import json
from sgp4.api import Satrec, jday
import numpy as np
from datetime import datetime, timedelta, timezone
import math
from tqdm import tqdm
from collections import Counter
import statistics

# -----------------------
# User / runtime params
# -----------------------
USERNAME = "harshbhanushali36@gmail.com"
PASSWORD = "Trinetra_12345678"
LOGIN_URL = "https://www.space-track.org/ajaxauth/login"
CATALOG_URL = "https://www.space-track.org/basicspacedata/query/class/gp/format/json"

# --- MODES ---
MODES = {
    "HIGH_ACCURACY": {"DURATION_HOURS": 1, "STEP_SECONDS": 60},
    "BALANCED": {"DURATION_HOURS": 0.5, "STEP_SECONDS": 300}
}

# Object filter
WANTED_TYPES = {"PAYLOAD", "ROCKET BODY", "DEBRIS"}
TLE_MAX_AGE_DAYS = 30   # skip TLEs older than 30 days

# -----------------------
# Utility functions
# -----------------------
def gmst_from_jd(jd_ut1):
    T = (jd_ut1 - 2451545.0) / 36525.0
    gmst_sec = 67310.54841 + (876600.0 * 3600 + 8640184.812866) * T \
               + 0.093104 * T**2 - 6.2e-6 * T**3
    gmst_sec = gmst_sec % 86400.0
    return (gmst_sec / 86400.0) * 2.0 * math.pi

def rotate_z(vec, angle_rad):
    c = math.cos(angle_rad); s = math.sin(angle_rad)
    x, y, z = vec
    return (c*x - s*y, s*x + c*y, z)

def teme_to_ecef_km(r_teme_km, dt_utc):
    jd, jd_frac = jday(dt_utc.year, dt_utc.month, dt_utc.day,
                       dt_utc.hour, dt_utc.minute, dt_utc.second + dt_utc.microsecond*1e-6)
    jd_ut1 = jd + jd_frac
    gmst = gmst_from_jd(jd_ut1)
    return rotate_z(r_teme_km, gmst)

def xyz_to_lat_lon_alt(x_m, y_m, z_m):
    """Convert ECEF XYZ (meters) to latitude, longitude, altitude"""
    a = 6378137.0
    f = 1 / 298.257223563
    e2 = 2 * f - f * f
    p = math.sqrt(x_m**2 + y_m**2)
    lat = math.atan2(z_m, p * (1 - e2))
    for _ in range(3):
        N = a / math.sqrt(1 - e2 * math.sin(lat)**2)
        alt = p / math.cos(lat) - N
        lat = math.atan2(z_m, p * (1 - e2 * N / (N + alt)))
    lon = math.atan2(y_m, x_m)
    return math.degrees(lat), math.degrees(lon), alt / 1000.0

def satrec_epoch_to_datetime(sat):
    try:
        jd = float(sat.jdsatepoch) + float(sat.jdsatepochF)
    except Exception:
        return None
    jd += 0.5
    F, I = math.modf(jd)
    I = int(I)
    A = int((I - 1867216.25) / 36524.25)
    if I > 2299160:
        B = I + 1 + A - int(A / 4)
    else:
        B = I
    C = B + 1524
    D = int((C - 122.1) / 365.25)
    E = int(365.25 * D)
    G = int((C - E) / 30.6001)
    day = C - E + F - int(30.6001 * G)
    if G < 13.5:
        month = G - 1
    else:
        month = G - 13
    if month > 2.5:
        year = D - 4716
    else:
        year = D - 4715
    day_frac, day_int = math.modf(day)
    day_int = int(day_int)
    secs = day_frac * 86400.0
    hour = int(secs // 3600)
    minute = int((secs % 3600) // 60)
    second = secs - hour*3600 - minute*60
    microsecond = int((second - int(second)) * 1e6)
    second = int(second)
    try:
        return datetime(year, month, day_int, hour, minute, second, microsecond, tzinfo=timezone.utc)
    except Exception:
        return None

# -----------------------
# Main function
# -----------------------
def generate_satellite_json(run_mode="BALANCED"):
    # -----------------------
    # 1) Fetch
    # -----------------------
    session = requests.Session()
    resp = session.post(LOGIN_URL, data={"identity": USERNAME, "password": PASSWORD})
    resp.raise_for_status()
    print("✅ Logged into Space-Track successfully!")

    response = session.get(CATALOG_URL)
    response.raise_for_status()
    data = response.json()
    df = pd.DataFrame(data)

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

    # -----------------------
    # 2) Mode selection
    # -----------------------
    DURATION_HOURS = MODES[run_mode]["DURATION_HOURS"]
    STEP_SECONDS = MODES[run_mode]["STEP_SECONDS"]

    start_utc = datetime.now(timezone.utc)
    end_utc = start_utc + timedelta(hours=DURATION_HOURS)
    num_steps = int((end_utc - start_utc).total_seconds() // STEP_SECONDS) + 1
    time_offsets = [i * STEP_SECONDS for i in range(num_steps)]
    time_datetimes = [start_utc + timedelta(seconds=off) for off in time_offsets]
    epoch_for_json = start_utc.isoformat()

    print(f"\nMode: {run_mode}")
    print(f"JSON epoch (UTC): {epoch_for_json}")
    print(f"Propagating {DURATION_HOURS} hr with step {STEP_SECONDS}s ({num_steps} steps)")

    # -----------------------
    # 3) Build candidate list
    # -----------------------
    candidates = []
    for idx, row in df_filtered.iterrows():
        obj_type = row.get("Type", "")
        tle1 = row.get("TLE Line 1")
        tle2 = row.get("TLE Line 2")
        if pd.isna(tle1) or pd.isna(tle2):
            continue
        if WANTED_TYPES and obj_type not in WANTED_TYPES:
            continue
        name = str(row.get("Name") or row.get("ID") or f"object_{idx}")
        candidates.append((name, tle1.strip(), tle2.strip(), obj_type))

    print(f"Found {len(candidates)} TLE candidates.")

    # -----------------------
    # 4) Propagation
    # -----------------------
    satellites = []
    objects_written = 0
    objects_skipped_tle_age = 0
    objects_skipped_no_valid = 0
    sgp4_error_counts = Counter()
    type_counts = Counter()

    TYPE_STYLES = {
        "PAYLOAD":  {"color": "#00FF00", "symbol": "🛰️", "size": 4},
        "ROCKET BODY": {"color": "#FFA500", "symbol": "🚀", "size": 3},
        "DEBRIS":   {"color": "#FF0000", "symbol": "💥", "size": 2}
    }

    for name, tle1, tle2, obj_type in tqdm(candidates, desc="Propagating", unit="obj"):
        try:
            sat = Satrec.twoline2rv(tle1, tle2)
        except:
            objects_skipped_no_valid += 1
            continue

        tle_epoch_dt = satrec_epoch_to_datetime(sat)
        if tle_epoch_dt is None:
            objects_skipped_no_valid += 1
            continue

        age_days = (start_utc - tle_epoch_dt).total_seconds() / 86400.0
        if (TLE_MAX_AGE_DAYS is not None) and (age_days > TLE_MAX_AGE_DAYS):
            objects_skipped_tle_age += 1
            continue

        jd_epoch, fr_epoch = jday(tle_epoch_dt.year, tle_epoch_dt.month, tle_epoch_dt.day,
                                  tle_epoch_dt.hour, tle_epoch_dt.minute,
                                  tle_epoch_dt.second + tle_epoch_dt.microsecond*1e-6)
        e0, r0, v0 = sat.sgp4(jd_epoch, fr_epoch)
        if e0 != 0:
            objects_skipped_no_valid += 1
            sgp4_error_counts[e0] += 1
            continue

        valid_samples = []
        for dt in time_datetimes:
            jd, fr = jday(dt.year, dt.month, dt.day,
                          dt.hour, dt.minute, dt.second + dt.microsecond*1e-6)
            e, r, v = sat.sgp4(jd, fr)
            if e != 0:
                sgp4_error_counts[e] += 1
                continue
            r_ecef_km = teme_to_ecef_km(r, dt)
            r_ecef_m = tuple(x * 1000.0 for x in r_ecef_km)
            lat, lon, alt = xyz_to_lat_lon_alt(*r_ecef_m)
            valid_samples.append({
                "time": dt.isoformat(),
                "timestamp": dt.timestamp(),
                "latitude": lat,
                "longitude": lon,
                "altitude": alt,
                "x": r_ecef_m[0],
                "y": r_ecef_m[1],
                "z": r_ecef_m[2]
            })

        if not valid_samples:
            objects_skipped_no_valid += 1
            continue

        display_type = obj_type if obj_type in TYPE_STYLES else "UNKNOWN"
        style = TYPE_STYLES.get(display_type, {"color": "#C8C8C8", "symbol": "❓", "size": 2})

        satellite_data = {
            "id": name.replace(" ", "_")[:64],
            "name": name,
            "type": obj_type,
            "display_type": display_type,
            "tle_age_days": round(age_days, 2),
            "style": {
                "color": style["color"],
                "symbol": style["symbol"],
                "size": style["size"]
            },
            "epoch": epoch_for_json,
            "positions": valid_samples
        }

        satellites.append(satellite_data)
        objects_written += 1
        type_counts[display_type] += 1

    # -----------------------
    # 5) Build final JSON structure
    # -----------------------
    json_data = {
        "metadata": {
            "generated": datetime.now(timezone.utc).isoformat(),
            "mode": run_mode,
            "epoch": epoch_for_json,
            "duration_hours": DURATION_HOURS,
            "step_seconds": STEP_SECONDS,
            "num_time_steps": num_steps,
            "start_time": start_utc.isoformat(),
            "end_time": end_utc.isoformat(),
            "tle_max_age_days": TLE_MAX_AGE_DAYS
        },
        "statistics": {
            "total_candidates": len(candidates),
            "objects_written": objects_written,
            "objects_skipped_old_tle": objects_skipped_tle_age,
            "objects_skipped_errors": objects_skipped_no_valid,
            "type_counts": {
                "PAYLOAD": type_counts.get("PAYLOAD", 0),
                "ROCKET_BODY": type_counts.get("ROCKET BODY", 0),
                "DEBRIS": type_counts.get("DEBRIS", 0),
                "UNKNOWN": type_counts.get("UNKNOWN", 0)
            },
            "sgp4_error_counts": dict(sgp4_error_counts)
        },
        "satellites": satellites
    }

    return json_data

# -----------------------
# Example usage
# -----------------------
RUN_MODE = "HIGH_ACCURACY"  # or "HIGH_ACCURACY"

if __name__ == "__main__":
    result = generate_satellite_json(RUN_MODE)
    print("\n✅ JSON generation completed")
    print("Mode:", RUN_MODE)
    print("Total objects:", len(result["satellites"]))
    print("Objects written:", result["statistics"]["objects_written"])
    print("Objects skipped (old TLE):", result["statistics"]["objects_skipped_old_tle"])
    print("Objects skipped (errors):", result["statistics"]["objects_skipped_errors"])                  
    print("Type counts:", result["statistics"]["type_counts"])
    print("SGP4 error counts:", result["statistics"]["sgp4_error_counts"])  
    
    

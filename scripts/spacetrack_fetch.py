import requests
import pandas as pd

# Space-Track credentials
USERNAME = "harshbhanushali36@gmail.com"
PASSWORD = "Trinetra_12345678"

# Space-Track API endpoints
LOGIN_URL = "https://www.space-track.org/ajaxauth/login"
CATALOG_URL = "https://www.space-track.org/basicspacedata/query/class/gp/format/json"

# Create a session
session = requests.Session()

# Login payload
login_payload = {
    "identity": USERNAME,
    "password": PASSWORD
}

# Perform login
resp = session.post(LOGIN_URL, data=login_payload)
resp.raise_for_status()
print("✅ Logged into Space-Track successfully!")

# Fetch satellite catalog data
response = session.get(CATALOG_URL)
response.raise_for_status()
data = response.json()

# Convert to DataFrame
df = pd.DataFrame(data)

# Rename columns to be clear & consistent
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

# Keep only useful columns
columns_to_keep = list(rename_map.values())
df_filtered = df[[col for col in columns_to_keep if col in df.columns]]

# Split into categories
df_satellites = df_filtered[df_filtered["Type"] == "PAYLOAD"]
df_rockets = df_filtered[df_filtered["Type"] == "ROCKET BODY"]
df_debris = df_filtered[df_filtered["Type"] == "DEBRIS"]

# Save to single Excel file with multiple sheets
output_file = "spacetrack_data.xlsx"
with pd.ExcelWriter(output_file, engine="openpyxl") as writer:
    df.to_excel(writer, sheet_name="Raw Data", index=False)             # Full raw dump with TLEs
    df_filtered.to_excel(writer, sheet_name="Filtered Data", index=False) # Cleaned data with TLEs
    df_satellites.to_excel(writer, sheet_name="Satellites", index=False)
    df_rockets.to_excel(writer, sheet_name="Rockets", index=False)
    df_debris.to_excel(writer, sheet_name="Debris", index=False)

print(f"✅ Data saved to {output_file} with multiple sheets including TLEs!")

import requests
import pandas as pd
import time
import os

API_KEY = "SEsoH1p8IBnZg44ePhFNPLKtcIHXgIQy3uriPjrc"
BASE_URL = "https://api.nasa.gov/neo/rest/v1"

class NASAAllAsteroidsFetcher:
    def __init__(self, api_key):
        self.api_key = api_key
        self.base_url = f"{BASE_URL}/neo/browse"

    def fetch_all_pages(self, max_pages=50, page_size=20, delay=1):
        all_data = []
        for page in range(max_pages):
            params = {"api_key": self.api_key, "page": page, "size": page_size}
            print(f"Fetching page {page+1}/{max_pages}...")
            try:
                response = requests.get(self.base_url, params=params)
                if response.status_code == 200:
                    data = response.json()
                    all_data.extend(data.get("near_earth_objects", []))
                    if len(data.get("near_earth_objects", [])) < page_size:
                        break
                else:
                    print(f"Error: Status {response.status_code}")
            except Exception as e:
                print(f"Exception on page {page}: {e}")
            time.sleep(delay)
        return all_data

    def process_data(self, neos):
        processed = []
        for neo in neos:
            orbital = neo.get("orbital_data", {})
            for approach in neo.get("close_approach_data", []):
                record = {
                    "id": neo["id"],
                    "name": neo["name"],
                    "is_potentially_hazardous": neo["is_potentially_hazardous_asteroid"],
                    "absolute_magnitude": neo["absolute_magnitude_h"],
                    "estimated_diameter_min_km": neo["estimated_diameter"]["kilometers"]["estimated_diameter_min"],
                    "estimated_diameter_max_km": neo["estimated_diameter"]["kilometers"]["estimated_diameter_max"],
                    "orbiting_body": approach.get("orbiting_body"),
                    "approach_date": approach.get("close_approach_date"),
                    "relative_velocity_kmh": float(approach.get("relative_velocity", {}).get("kilometers_per_hour", 0)),
                    "miss_distance_km": float(approach.get("miss_distance", {}).get("kilometers", 0)),
                    "nasa_jpl_url": neo.get("nasa_jpl_url"),
                    # Orbital elements
                    "eccentricity": float(orbital.get("eccentricity", 0)),
                    "semi_major_axis_au": float(orbital.get("semi_major_axis", 0)),
                    "inclination_deg": float(orbital.get("inclination", 0)),
                    "ascending_node_longitude_deg": float(orbital.get("ascending_node_longitude", 0)),
                    "argument_of_perihelion_deg": float(orbital.get("argument_of_perihelion", 0)),
                    "mean_anomaly_deg": float(orbital.get("mean_anomaly", 0)),
                    "perihelion_distance_au": float(orbital.get("perihelion_distance", 0)),
                    "aphelion_distance_au": float(orbital.get("aphelion_distance", 0)),
                    "orbital_period_days": float(orbital.get("orbital_period", 0))
                }
                processed.append(record)
        df = pd.DataFrame(processed)
        return df

    def filter_hazardous(self, df):
        return df[df["is_potentially_hazardous"] == True].copy()

    def filter_close(self, df, max_distance_km=10_000_000):
        return df[df["miss_distance_km"] <= max_distance_km].copy()

    def filter_largest(self, df, top_n=50):
        return df.nlargest(top_n, "estimated_diameter_max_km").copy()

    def save_excel(self, df_all, df_hazardous, df_close, df_largest, filename="nasa_asteroids.xlsx"):
        with pd.ExcelWriter(filename, engine="openpyxl") as writer:
            df_all.to_excel(writer, sheet_name="All_Asteroids", index=False)
            df_hazardous.to_excel(writer, sheet_name="Hazardous", index=False)
            df_close.to_excel(writer, sheet_name="Close_Approaches", index=False)
            df_largest.to_excel(writer, sheet_name="Largest", index=False)
        print(f"✅ All data saved to {os.path.abspath(filename)}")


def main():
    fetcher = NASAAllAsteroidsFetcher(API_KEY)
    neos = fetcher.fetch_all_pages(max_pages=100, page_size=20, delay=1)
    print(f"Total NEOs fetched: {len(neos)}")

    df_all = fetcher.process_data(neos)
    print(f"Total asteroid approaches processed: {len(df_all)}")

    df_hazardous = fetcher.filter_hazardous(df_all)
    df_close = fetcher.filter_close(df_all)
    df_largest = fetcher.filter_largest(df_all, top_n=50)

    fetcher.save_excel(df_all, df_hazardous, df_close, df_largest)

if __name__ == "__main__":
    main()

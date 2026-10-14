import os
import json
import urllib.request
from pathlib import Path

import pandas as pd

# Sensor we discovered in Step 13
SENSOR_ID = 23534

# Get API key securely from the terminal environment
api_key = os.getenv("OPENAQ_API_KEY")

if not api_key:
    print("API key missing. Set it again in this terminal.")
    raise SystemExit(1)

# Request up to 1000 measurements
url = (
    f"https://api.openaq.org/v3/sensors/"
    f"{SENSOR_ID}/measurements?limit=1000&page=1"
)

request = urllib.request.Request(
    url,
    headers={"X-API-Key": api_key}
)

try:
    with urllib.request.urlopen(request, timeout=30) as response:
        data = json.loads(response.read().decode("utf-8"))

    results = data.get("results", [])

    if not results:
        print("No measurements returned. We will investigate next.")
        raise SystemExit(0)

    rows = []

    for item in results:
        period = item.get("period") or {}
        start = period.get("datetimeFrom") or {}
        end = period.get("datetimeTo") or {}
        parameter = item.get("parameter") or {}

        rows.append({
            "sensor_id": SENSOR_ID,
            "datetime_local": (
                start.get("local") if isinstance(start, dict)
                else start
            ),
            "datetime_utc": (
                start.get("utc") if isinstance(start, dict)
                else None
            ),
            "datetime_to_local": (
                end.get("local") if isinstance(end, dict)
                else end
            ),
            "pm25": item.get("value"),
            "unit": parameter.get("units"),
        })

    df = pd.DataFrame(rows)

    # Create a CSV file in data/raw
    output = Path("data/raw/pm25_new_delhi.csv")
    output.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output, index=False)

    print("Download successful!")
    print("Records received:", len(df))
    print("Valid PM2.5 values:", df["pm25"].notna().sum())
    print("Saved to:", output)
    print("\nFirst five records:")
    print(df.head().to_string(index=False))

except Exception as error:
    print("Download failed:", error)
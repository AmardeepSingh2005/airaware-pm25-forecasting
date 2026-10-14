
import json
import urllib.request
from urllib.parse import urlencode
from pathlib import Path

import pandas as pd

# New Delhi monitoring station coordinates
LATITUDE = 28.63576
LONGITUDE = 77.22445

params = {
    "latitude": LATITUDE,
    "longitude": LONGITUDE,
    "start_date": "2026-07-15",
    "end_date": "2026-10-13",
    "hourly": (
        "temperature_2m,"
        "relative_humidity_2m,"
        "precipitation,"
        "wind_speed_10m,"
        "wind_direction_10m"
    ),
    "timezone": "Asia/Kolkata",
}

url = (
    "https://archive-api.open-meteo.com/v1/archive?"
    + urlencode(params)
)

request = urllib.request.Request(
    url,
    headers={"User-Agent": "AirAware-Student-Research/1.0"}
)

try:
    with urllib.request.urlopen(request, timeout=60) as response:
        weather_data = json.loads(
            response.read().decode("utf-8")
        )

    hourly = weather_data["hourly"]
    df = pd.DataFrame(hourly)

    output = Path("data/raw/weather_new_delhi.csv")
    output.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output, index=False)

    print("Weather download successful!")
    print("Records received:", len(df))
    print("Weather columns:", list(df.columns))
    print("Saved to:", output)

    print("\nFirst five records:")
    print(df.head().to_string(index=False))

except Exception as error:
    print("Weather download failed:", error)


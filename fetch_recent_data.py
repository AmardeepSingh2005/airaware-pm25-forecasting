import os
import json
import urllib.request
from urllib.parse import urlencode
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd

SENSOR_ID = 23534
LIMIT = 1000

api_key = os.getenv("OPENAQ_API_KEY")

if not api_key:
    print("API key missing. Use your existing terminal.")
    raise SystemExit(1)

# Define the last 90 days
end = datetime.now(timezone.utc)
start = end - timedelta(days=90)

base_params = {
    "datetime_from": start.isoformat(timespec="seconds"),
    "datetime_to": end.isoformat(timespec="seconds"),
    "limit": LIMIT,
}

all_rows = []
page = 1

try:
    while True:
        params = {**base_params, "page": page}

        url = (
            f"https://api.openaq.org/v3/sensors/"
            f"{SENSOR_ID}/measurements?{urlencode(params)}"
        )

        request = urllib.request.Request(
            url,
            headers={"X-API-Key": api_key}
        )

        with urllib.request.urlopen(request, timeout=30) as response:
            data = json.loads(response.read().decode("utf-8"))

        results = data.get("results", [])

        if not results:
            break

        print(f"Downloaded page {page}: {len(results)} records")

        for item in results:
            period = item.get("period") or {}
            start_time = period.get("datetimeFrom") or {}
            end_time = period.get("datetimeTo") or {}
            parameter = item.get("parameter") or {}

            def get_time(value, field):
                if isinstance(value, dict):
                    return value.get(field)
                return value if field == "local" else None

            all_rows.append({
                "sensor_id": SENSOR_ID,
                "datetime_local": get_time(start_time, "local"),
                "datetime_utc": get_time(start_time, "utc"),
                "datetime_to_local": get_time(end_time, "local"),
                "pm25": item.get("value"),
                "unit": parameter.get("units"),
            })

        # Stop when there are no more full pages
        if len(results) < LIMIT:
            break

        page += 1

    if not all_rows:
        print("No records found for this date range.")
        raise SystemExit(0)

    df = pd.DataFrame(all_rows)

    output = Path("data/raw/pm25_new_delhi_recent_90d.csv")
    output.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output, index=False)

    dates = pd.to_datetime(df["datetime_utc"], utc=True, errors="coerce")
    values = pd.to_numeric(df["pm25"], errors="coerce")

    print("\n----- DOWNLOAD SUMMARY -----")
    print("Total records:", len(df))
    print("Valid PM2.5 readings:", values.notna().sum())
    print("Missing PM2.5 readings:", values.isna().sum())
    print("Earliest record:", dates.min())
    print("Latest record:", dates.max())
    print("Saved to:", output)

except Exception as error:
    print("Download failed:", error)
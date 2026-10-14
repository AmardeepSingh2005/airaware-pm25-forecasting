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
    print("API key missing. Use the terminal where you saved it.")
    raise SystemExit(1)

end = datetime.now(timezone.utc)
start = end - timedelta(days=365)

all_rows = []
page = 1

try:
    while True:
        params = {
            "datetime_from": start.isoformat(timespec="seconds"),
            "datetime_to": end.isoformat(timespec="seconds"),
            "limit": LIMIT,
            "page": page
        }

        url = (
            f"https://api.openaq.org/v3/sensors/"
            f"{SENSOR_ID}/measurements?{urlencode(params)}"
        )

        request = urllib.request.Request(
            url,
            headers={"X-API-Key": api_key}
        )

        with urllib.request.urlopen(request, timeout=45) as response:
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

            all_rows.append({
                "sensor_id": SENSOR_ID,
                "datetime_local": (
                    start_time.get("local")
                    if isinstance(start_time, dict) else None
                ),
                "datetime_utc": (
                    start_time.get("utc")
                    if isinstance(start_time, dict) else None
                ),
                "datetime_to_local": (
                    end_time.get("local")
                    if isinstance(end_time, dict) else None
                ),
                "pm25": item.get("value"),
                "unit": parameter.get("units")
            })

        if len(results) < LIMIT:
            break

        page += 1

    if not all_rows:
        print("No measurements returned for this date range.")
        raise SystemExit(0)

    df = pd.DataFrame(all_rows)

    df = df.drop_duplicates(
        subset=["sensor_id", "datetime_utc"],
        keep="first"
    )

    df["datetime_utc"] = pd.to_datetime(
        df["datetime_utc"], utc=True, errors="coerce"
    )

    df = df.sort_values("datetime_utc")

    output = Path("data/raw/pm25_new_delhi_365d.csv")
    output.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output, index=False)

    print("\n===== ANNUAL DATASET SUMMARY =====")
    print("Total records:", len(df))
    print("Valid PM2.5 readings:", df["pm25"].notna().sum())
    print("Earliest timestamp:", df["datetime_utc"].min())
    print("Latest timestamp:", df["datetime_utc"].max())
    print("Saved to:", output)

except Exception as error:
    print("Download failed:", error)

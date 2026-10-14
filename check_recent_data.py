import os
import json
import urllib.request
import urllib.error

from datetime import datetime, timedelta, timezone
from urllib.parse import urlencode

SENSOR_ID = 23534
api_key = os.getenv("OPENAQ_API_KEY")

if not api_key:
    print("API key missing. Use the same terminal session.")
    raise SystemExit(1)

# Check the last 90 days
end = datetime.now(timezone.utc)
start = end - timedelta(days=90)

params = {
    "datetime_from": start.isoformat(timespec="seconds"),
    "datetime_to": end.isoformat(timespec="seconds"),
    "limit": 1000,
    "page": 1
}

url = (
    f"https://api.openaq.org/v3/sensors/"
    f"{SENSOR_ID}/measurements?{urlencode(params)}"
)

request = urllib.request.Request(
    url,
    headers={"X-API-Key": api_key}
)

try:
    with urllib.request.urlopen(request, timeout=30) as response:
        data = json.loads(response.read().decode("utf-8"))

    results = data.get("results", [])

    print("Recent records found:", len(results))

    timestamps = []
    valid_values = 0

    for item in results:
        period = item.get("period") or {}
        start_time = period.get("datetimeFrom") or {}

        if isinstance(start_time, dict):
            timestamp = start_time.get("utc")
        else:
            timestamp = start_time

        if timestamp:
            timestamps.append(timestamp)

        if item.get("value") is not None:
            valid_values += 1

    print("Valid measurements:", valid_values)

    if timestamps:
        print("Earliest returned:", min(timestamps))
        print("Latest returned:", max(timestamps))
    else:
        print("No timestamps returned.")

except urllib.error.HTTPError as error:
    print("API error:", error.code)
    print(error.read().decode("utf-8")[:500])

except Exception as error:
    print("Request failed:", error)
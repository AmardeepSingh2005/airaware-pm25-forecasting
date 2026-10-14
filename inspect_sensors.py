import os
import json
import urllib.request

api_key = os.getenv("OPENAQ_API_KEY")

if not api_key:
    print("API key missing. Check your terminal.")
else:
    url = "https://api.openaq.org/v3/locations/8118/sensors"

    request = urllib.request.Request(
        url,
        headers={"X-API-Key": api_key}
    )

    try:
        with urllib.request.urlopen(request, timeout=20) as response:
            data = json.loads(response.read().decode("utf-8"))

        sensors = data.get("results", [])
        print("Sensors found:", len(sensors))

        for sensor in sensors:
            parameter = sensor.get("parameter") or {}
            latest = sensor.get("latest") or {}

            name = parameter.get("displayName") or parameter.get("name")
            units = parameter.get("units", "unknown")
            latest_date = (latest.get("datetime") or {}).get("local")

            print(
                f"Sensor ID: {sensor.get('id')} | "
                f"Pollutant: {name} | Units: {units}"
            )

            print(
                f"Latest value: {latest.get('value')} | "
                f"Date: {latest_date}"
            )

    except Exception as error:
        print("Error:", error)
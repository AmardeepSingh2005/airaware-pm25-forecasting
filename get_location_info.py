import os
import json
import urllib.request

LOCATION_ID = 8118
api_key = os.getenv("OPENAQ_API_KEY")

if not api_key:
    print("API key missing. Use your existing terminal.")
    raise SystemExit(1)

url = f"https://api.openaq.org/v3/locations/{LOCATION_ID}"

request = urllib.request.Request(
    url,
    headers={"X-API-Key": api_key}
)

try:
    with urllib.request.urlopen(request, timeout=20) as response:
        data = json.loads(response.read().decode("utf-8"))

    location = data["results"][0]
    coordinates = location.get("coordinates") or {}

    print("Location:", location.get("name"))
    print("Timezone:", location.get("timezone"))
    print("Latitude:", coordinates.get("latitude"))
    print("Longitude:", coordinates.get("longitude"))

except Exception as error:
    print("Request failed:", error)
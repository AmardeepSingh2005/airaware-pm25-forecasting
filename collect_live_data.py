"""Collect AirAware forecasts and OpenAQ observations.

Keeps the existing central New Delhi forecast rows (location='New Delhi') and
also stores sensor-specific forecast snapshots under locations 'sensor-<ID>'.
The latter allow a later fair comparison of each sensor with a forecast requested
for that sensor's coordinates. All timestamps are saved in UTC.
"""
from __future__ import annotations

import json
import os
import tomllib
from datetime import datetime, timedelta, timezone
from pathlib import Path
from urllib.parse import urlencode
from urllib.request import Request, urlopen
from zoneinfo import ZoneInfo

from supabase import create_client

SENSOR_ID = 23534
LOCATION = "New Delhi"
CENTER_LATITUDE = 28.6139
CENTER_LONGITUDE = 77.2090
OPENAQ_LIMIT = 1000
BATCH_SIZE = 500
FORECAST_HOURS_TO_SAVE = 24
OPENAQ_BASE_URL = "https://api.openaq.org/v3"

SENSORS = {
    "23534": {"name": "New Delhi (current sensor)", "distance_km": 2.9},
    "15289075": {"name": "Air Check", "distance_km": 2.7},
    "9320367": {"name": "GK1 (Oberoi Terrace)", "distance_km": 7.8},
    "14745878": {"name": "Anand Lok", "distance_km": 6.2},
    "13273790": {"name": "Santushti Apartments, Vasant Kunj", "distance_km": 11.3},
}


def load_settings():
    """Load credentials from environment variables or local secrets."""
    root = Path(__file__).resolve().parent
    secrets_path = root / ".streamlit" / "secrets.toml"
    local_secrets = {}
    if secrets_path.exists():
        with secrets_path.open("rb") as file:
            local_secrets = tomllib.load(file)

    supabase_url = os.getenv("SUPABASE_URL") or local_secrets.get("SUPABASE_URL")
    supabase_key = (
        os.getenv("SUPABASE_SECRET_KEY")
        or os.getenv("SUPABASE_SERVICE_ROLE_KEY")
        or local_secrets.get("SUPABASE_SECRET_KEY")
        or local_secrets.get("SUPABASE_SERVICE_ROLE_KEY")
    )
    openaq_key = os.getenv("OPENAQ_API_KEY") or local_secrets.get("OPENAQ_API_KEY")

    missing = []
    if not supabase_url:
        missing.append("SUPABASE_URL")
    if not supabase_key:
        missing.append("SUPABASE_SECRET_KEY")
    if not openaq_key:
        missing.append("OPENAQ_API_KEY")
    if missing:
        raise RuntimeError("Missing configuration: " + ", ".join(missing))
    if not str(supabase_url).startswith(("https://", "http://")):
        raise RuntimeError("SUPABASE_URL does not look like a valid URL.")
    return str(supabase_url), str(supabase_key), str(openaq_key)


def read_json(url: str, headers: dict[str, str] | None = None, timeout: int = 30):
    request = Request(url, headers=headers or {"User-Agent": "AirAware-Research/1.0"})
    with urlopen(request, timeout=timeout) as response:
        if response.status != 200:
            raise RuntimeError(f"HTTP {response.status} returned for request")
        return json.loads(response.read().decode("utf-8"))


def fetch_open_meteo_forecast(latitude: float, longitude: float):
    """Fetch hourly PM2.5 forecasts for a requested coordinate pair."""
    params = {
        "latitude": f"{latitude:.6f}",
        "longitude": f"{longitude:.6f}",
        "current": "pm2_5",
        "hourly": "pm2_5",
        "forecast_hours": 48,
        "timezone": "Asia/Kolkata",
    }
    url = "https://air-quality-api.open-meteo.com/v1/air-quality?" + urlencode(params)
    payload = read_json(url)
    hourly = payload.get("hourly", {})
    if not hourly.get("time") or not hourly.get("pm2_5"):
        raise RuntimeError("Open-Meteo returned no hourly PM2.5 data.")
    return payload


def build_forecast_rows(payload, issued_at=None, location: str = LOCATION):
    """Prepare future hourly forecast rows for one named location."""
    zone_name = payload.get("timezone", "Asia/Kolkata")
    local_zone = ZoneInfo(zone_name)
    if issued_at is None:
        issued_at = datetime.now(timezone.utc)
    elif issued_at.tzinfo is None:
        issued_at = issued_at.replace(tzinfo=timezone.utc)
    else:
        issued_at = issued_at.astimezone(timezone.utc)

    hourly = payload["hourly"]
    rows = []
    for raw_time, raw_value in zip(hourly.get("time", []), hourly.get("pm2_5", [])):
        if raw_value is None:
            continue
        try:
            valid_local = datetime.fromisoformat(str(raw_time))
            if valid_local.tzinfo is None:
                valid_local = valid_local.replace(tzinfo=local_zone)
            predicted = float(raw_value)
            if predicted < 0:
                continue
            valid_utc = valid_local.astimezone(timezone.utc)
            # Do not store a forecast whose target time has already passed.
            if valid_utc <= issued_at:
                continue
            horizon = (valid_utc - issued_at).total_seconds() / 3600.0
            rows.append({
                "provider": "open-meteo",
                "location": location,
                "issued_at": issued_at.isoformat(),
                "valid_at": valid_utc.isoformat(),
                "forecast_horizon_hours": max(0.0, horizon),
                "predicted_pm25": predicted,
            })
            if len(rows) >= FORECAST_HOURS_TO_SAVE:
                break
        except (TypeError, ValueError, OverflowError):
            continue
    if not rows:
        raise RuntimeError(f"No future forecast rows could be prepared for {location}.")
    return rows


def fetch_sensor_coordinates(sensor_id: str, api_key: str) -> tuple[float, float]:
    """Read a sensor's latest coordinates from OpenAQ sensor metadata."""
    url = f"{OPENAQ_BASE_URL}/sensors/{sensor_id}"
    payload = read_json(url, headers={
        "X-API-Key": api_key,
        "Accept": "application/json",
        "User-Agent": "AirAware-Research/1.0",
    })
    results = payload.get("results") or []
    if not results:
        raise RuntimeError(f"OpenAQ returned no metadata for sensor {sensor_id}.")
    sensor = results[0]
    latest = sensor.get("latest") or {}
    coordinates = latest.get("coordinates") or sensor.get("coordinates") or {}
    latitude = coordinates.get("latitude")
    longitude = coordinates.get("longitude")
    if latitude is None or longitude is None:
        raise RuntimeError(f"OpenAQ has no usable coordinates for sensor {sensor_id}.")
    latitude = float(latitude)
    longitude = float(longitude)
    if not (-90 <= latitude <= 90 and -180 <= longitude <= 180):
        raise RuntimeError(f"OpenAQ returned invalid coordinates for sensor {sensor_id}.")
    return latitude, longitude


def fetch_recent_openaq_observations(api_key):
    """Fetch recent observations for the primary sensor, with metadata supplement."""
    end = datetime.now(timezone.utc)
    start = end - timedelta(days=7)
    base_params = {
        "datetime_from": start.isoformat(timespec="seconds"),
        "datetime_to": end.isoformat(timespec="seconds"),
        "limit": OPENAQ_LIMIT,
    }
    observations = {}
    page = 1
    while True:
        url = f"{OPENAQ_BASE_URL}/sensors/{SENSOR_ID}/measurements?" + urlencode({**base_params, "page": page})
        payload = read_json(url, headers={
            "X-API-Key": api_key,
            "Accept": "application/json",
            "User-Agent": "AirAware-Research/1.0",
        })
        results = payload.get("results", [])
        if not results:
            break
        for item in results:
            period = item.get("period") or {}
            start_time = period.get("datetimeFrom") or {}
            parameter = item.get("parameter") or {}
            timestamp_raw = start_time.get("utc") if isinstance(start_time, dict) else start_time
            if not timestamp_raw or item.get("value") is None:
                continue
            try:
                observed_at = datetime.fromisoformat(str(timestamp_raw).replace("Z", "+00:00"))
                if observed_at.tzinfo is None:
                    observed_at = observed_at.replace(tzinfo=timezone.utc)
                observed_at = observed_at.astimezone(timezone.utc)
                observed_value = float(item["value"])
                if observed_value < 0 or not start <= observed_at <= end:
                    continue
                observations[observed_at.isoformat()] = {
                    "source": "OpenAQ",
                    "sensor_id": str(SENSOR_ID),
                    "location": LOCATION,
                    "observed_at": observed_at.isoformat(),
                    "observed_pm25": observed_value,
                    "unit": parameter.get("units") or "µg/m³",
                }
            except (TypeError, ValueError, OverflowError):
                continue
        print(f"Downloaded OpenAQ page {page}: {len(results)} records")
        if len(results) < OPENAQ_LIMIT:
            break
        page += 1

    # Supplemental latest value can be newer than the historical measurements feed.
    try:
        latest_payload = read_json(f"{OPENAQ_BASE_URL}/sensors/{SENSOR_ID}", headers={
            "X-API-Key": api_key,
            "Accept": "application/json",
            "User-Agent": "AirAware-Research/1.0",
        })
        meta_results = latest_payload.get("results") or []
        if meta_results:
            sensor = meta_results[0]
            latest = sensor.get("latest") or {}
            dt_obj = latest.get("datetime") or {}
            raw_time = dt_obj.get("utc") if isinstance(dt_obj, dict) else dt_obj
            raw_value = latest.get("value")
            units = (sensor.get("parameter") or {}).get("units") or "µg/m³"
            if raw_time and raw_value is not None:
                at = datetime.fromisoformat(str(raw_time).replace("Z", "+00:00"))
                if at.tzinfo is None:
                    at = at.replace(tzinfo=timezone.utc)
                at = at.astimezone(timezone.utc)
                value = float(raw_value)
                if value >= 0 and start <= at <= end:
                    key = at.isoformat()
                    if key not in observations:
                        observations[key] = {
                            "source": "OpenAQ", "sensor_id": str(SENSOR_ID),
                            "location": LOCATION, "observed_at": key,
                            "observed_pm25": value, "unit": units,
                        }
                        print("Supplemental sensor metadata reading added:", key, f"({value:g} µg/m³)")
                    else:
                        print("Sensor metadata latest already exists in measurements:", key)
    except (OSError, ValueError, TypeError, KeyError, RuntimeError) as exc:
        print("Warning: sensor metadata latest could not be added:", f"{type(exc).__name__}: {exc}")

    if not observations:
        raise RuntimeError("OpenAQ returned no valid observations in the last seven days.")
    return sorted(observations.values(), key=lambda row: row["observed_at"])


def upsert_batches(client, table, rows, conflict_columns):
    """Upload records in batches using each table's unique constraint."""
    for start in range(0, len(rows), BATCH_SIZE):
        client.table(table).upsert(rows[start:start + BATCH_SIZE], on_conflict=conflict_columns).execute()


def main():
    supabase_url, supabase_key, openaq_key = load_settings()
    client = create_client(supabase_url, supabase_key)

    # Preserve the central New Delhi series used by the existing dashboard.
    center_issued_at = datetime.now(timezone.utc)
    center_payload = fetch_open_meteo_forecast(CENTER_LATITUDE, CENTER_LONGITUDE)
    center_rows = build_forecast_rows(center_payload, issued_at=center_issued_at, location=LOCATION)
    upsert_batches(client, "forecast_snapshots", center_rows, "provider,location,issued_at,valid_at")
    print(f"Forecast snapshots saved: {len(center_rows)} (central New Delhi series)")

    # Start collecting forecasts issued for each sensor's own metadata coordinates.
    sensor_forecast_total = 0
    snapped_grids: dict[tuple[float, float], list[str]] = {}
    for sensor_id, spec in SENSORS.items():
        try:
            latitude, longitude = fetch_sensor_coordinates(sensor_id, openaq_key)
            issued_at = datetime.now(timezone.utc)
            payload = fetch_open_meteo_forecast(latitude, longitude)
            sensor_location_key = f"sensor-{sensor_id}"
            rows = build_forecast_rows(payload, issued_at=issued_at, location=sensor_location_key)
            upsert_batches(client, "forecast_snapshots", rows, "provider,location,issued_at,valid_at")
            sensor_forecast_total += len(rows)
            grid_lat = float(payload.get("latitude", latitude))
            grid_lon = float(payload.get("longitude", longitude))
            grid_key = (round(grid_lat, 4), round(grid_lon, 4))
            snapped_grids.setdefault(grid_key, []).append(sensor_id)
            print(
                f"Sensor-specific forecast saved: sensor {sensor_id} ({spec['name']}), "
                f"{len(rows)} snapshots; requested=({latitude:.5f},{longitude:.5f}); "
                f"model_grid=({grid_lat:.4f},{grid_lon:.4f}); stored location={sensor_location_key}"
            )
        except Exception as exc:
            # Keep the central forecast and observation collection running if one sensor fails.
            print(f"WARNING: sensor-specific forecast failed for {sensor_id} ({spec['name']}): {type(exc).__name__}: {exc}")

    shared_grids = [ids for ids in snapped_grids.values() if len(ids) > 1]
    if shared_grids:
        print("Note: these sensors resolved to the same Open-Meteo model grid:", "; ".join(",".join(ids) for ids in shared_grids))
    print(f"Sensor-specific forecast snapshots saved: {sensor_forecast_total} across {len(snapped_grids)} model grids")

    observation_rows = fetch_recent_openaq_observations(openaq_key)
    upsert_batches(client, "pm25_observations", observation_rows, "source,sensor_id,observed_at")
    print(f"OpenAQ observations synced: {len(observation_rows)}")
    print("Latest observation:", max(row["observed_at"] for row in observation_rows))
    print("AirAware collection run completed successfully.")


if __name__ == "__main__":
    main()

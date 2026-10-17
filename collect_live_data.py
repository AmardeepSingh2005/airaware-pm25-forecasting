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
OPENAQ_LIMIT = 1000
BATCH_SIZE = 500
FORECAST_HOURS_TO_SAVE = 24


def load_settings():
    """Load credentials from environment variables or local secrets."""
    root = Path(__file__).resolve().parent
    secrets_path = root / ".streamlit" / "secrets.toml"
    local_secrets = {}

    if secrets_path.exists():
        with secrets_path.open("rb") as file:
            local_secrets = tomllib.load(file)

    supabase_url = (
        os.getenv("SUPABASE_URL")
        or local_secrets.get("SUPABASE_URL")
    )
    supabase_key = (
        os.getenv("SUPABASE_SECRET_KEY")
        or os.getenv("SUPABASE_SERVICE_ROLE_KEY")
        or local_secrets.get("SUPABASE_SECRET_KEY")
        or local_secrets.get("SUPABASE_SERVICE_ROLE_KEY")
    )
    openaq_key = os.getenv("OPENAQ_API_KEY")

    missing = []
    if not supabase_url:
        missing.append("SUPABASE_URL")
    if not supabase_key:
        missing.append("SUPABASE_SECRET_KEY")
    if not openaq_key:
        missing.append("OPENAQ_API_KEY")

    if missing:
        raise RuntimeError(
            "Missing configuration: " + ", ".join(missing)
        )

    return supabase_url, supabase_key, openaq_key


def fetch_open_meteo_forecast():
    """Fetch a fresh hourly PM2.5 forecast for New Delhi."""
    params = {
        "latitude": 28.6139,
        "longitude": 77.2090,
        "current": "pm2_5",
        "hourly": "pm2_5",
        "forecast_hours": 48,
        "timezone": "Asia/Kolkata",
    }

    url = (
        "https://air-quality-api.open-meteo.com/v1/air-quality?"
        + urlencode(params)
    )
    request = Request(
        url,
        headers={"User-Agent": "AirAware-Research/1.0"},
    )

    with urlopen(request, timeout=30) as response:
        if response.status != 200:
            raise RuntimeError(
                f"Open-Meteo returned HTTP {response.status}"
            )
        payload = json.loads(response.read().decode("utf-8"))

    hourly = payload.get("hourly", {})
    if not hourly.get("time") or not hourly.get("pm2_5"):
        raise RuntimeError("Open-Meteo returned no hourly PM2.5 data.")

    return payload


def build_forecast_rows(payload, issued_at=None):
    """Prepare 24 future forecast hours for Supabase.

    `issued_at` should be captured immediately before requesting the forecast,
    so the saved lead time reflects when the forecast request was made.
    """
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

    for raw_time, raw_value in zip(
        hourly.get("time", []),
        hourly.get("pm2_5", []),
    ):
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

            # Do not save forecast hours that have already passed.
            if valid_utc <= issued_at:
                continue

            horizon = (
                valid_utc - issued_at
            ).total_seconds() / 3600.0

            rows.append({
                "provider": "open-meteo",
                "location": LOCATION,
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
        raise RuntimeError("No future forecast rows could be prepared.")

    return rows


def fetch_recent_openaq_observations(api_key):
    """Fetch the most recent seven days of observations for the sensor."""
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
        params = {**base_params, "page": page}
        url = (
            f"https://api.openaq.org/v3/sensors/{SENSOR_ID}/measurements?"
            + urlencode(params)
        )
        request = Request(
            url,
            headers={
                "X-API-Key": api_key,
                "Accept": "application/json",
                "User-Agent": "AirAware-Research/1.0",
            },
        )

        with urlopen(request, timeout=30) as response:
            if response.status != 200:
                raise RuntimeError(
                    f"OpenAQ returned HTTP {response.status}"
                )
            payload = json.loads(response.read().decode("utf-8"))

        results = payload.get("results", [])
        if not results:
            break

        for item in results:
            period = item.get("period") or {}
            start_time = period.get("datetimeFrom") or {}
            parameter = item.get("parameter") or {}

            if isinstance(start_time, dict):
                timestamp_raw = start_time.get("utc")
            else:
                timestamp_raw = start_time

            if not timestamp_raw or item.get("value") is None:
                continue

            try:
                observed_at = datetime.fromisoformat(
                    str(timestamp_raw).replace("Z", "+00:00")
                )
                if observed_at.tzinfo is None:
                    observed_at = observed_at.replace(
                        tzinfo=timezone.utc
                    )
                observed_at = observed_at.astimezone(timezone.utc)

                observed_value = float(item["value"])
                if observed_value < 0:
                    continue

                if not start <= observed_at <= end:
                    continue

                key = observed_at.isoformat()
                observations[key] = {
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

    if not observations:
        raise RuntimeError(
            "OpenAQ returned no valid observations in the last seven days."
        )

    return sorted(
        observations.values(),
        key=lambda row: row["observed_at"],
    )


def upsert_batches(client, table, rows, conflict_columns):
    """Upload records in batches using each table's unique constraint."""
    for start in range(0, len(rows), BATCH_SIZE):
        batch = rows[start:start + BATCH_SIZE]
        client.table(table).upsert(
            batch,
            on_conflict=conflict_columns,
        ).execute()


def main():
    supabase_url, supabase_key, openaq_key = load_settings()
    client = create_client(supabase_url, supabase_key)

    # Save forecasts first so a temporary OpenAQ issue doesn't discard them.
    # Capture issue time before the API call, not after the response returns.
    forecast_issued_at = datetime.now(timezone.utc)
    forecast_payload = fetch_open_meteo_forecast()
    forecast_rows = build_forecast_rows(
        forecast_payload,
        issued_at=forecast_issued_at,
    )

    upsert_batches(
        client,
        "forecast_snapshots",
        forecast_rows,
        "provider,location,issued_at,valid_at",
    )
    print(f"Forecast snapshots saved: {len(forecast_rows)}")

    observation_rows = fetch_recent_openaq_observations(openaq_key)

    upsert_batches(
        client,
        "pm25_observations",
        observation_rows,
        "source,sensor_id,observed_at",
    )
    print(f"OpenAQ observations synced: {len(observation_rows)}")
    print(
        "Latest observation:",
        max(row["observed_at"] for row in observation_rows),
    )
    print("AirAware collection run completed successfully.")


if __name__ == "__main__":
    main()
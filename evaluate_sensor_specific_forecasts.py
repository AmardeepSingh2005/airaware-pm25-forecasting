"""Read-only cross-sensor diagnostic for AirAware's Open-Meteo forecasts.

Compares one latest-issued (pre-target) forecast per matured target hour with
nearby OpenAQ PM2.5 sensor measurements at the same target times. It only reads
Supabase/OpenAQ and writes local CSV reports; it does not modify database rows.

Run from the AirAware project root:
    python .\\cross_sensor_forecast_check.py
"""
from __future__ import annotations

import json
import math
import os
import sys
import tomllib
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
REPORTS_DIR = ROOT / "reports"
BASE_URL = "https://api.openaq.org/v3"
PAGE_SIZE = 1000
MATCH_TOLERANCE_MINUTES = 45
DEFAULT_LOCATION = "New Delhi"
DEFAULT_PROVIDER = "open-meteo"
# Nearby PM2.5 sensors selected from the read-only discovery run.
SENSORS = {
    "23534": {"name": "New Delhi (current sensor)", "distance_km": 2.9},
    "15289075": {"name": "Air Check", "distance_km": 2.7},
    "9320367": {"name": "GK1 (Oberoi Terrace)", "distance_km": 7.8},
    "14745878": {"name": "Anand Lok", "distance_km": 6.2},
    "13273790": {"name": "Santushti Apartments, Vasant Kunj", "distance_km": 11.3},
}


def load_settings() -> tuple[str, str, str]:
    """Load the keys from environment or the existing local secrets file."""
    url = os.getenv("SUPABASE_URL", "").strip()
    key = (
        os.getenv("SUPABASE_SECRET_KEY", "").strip()
        or os.getenv("SUPABASE_SERVICE_ROLE_KEY", "").strip()
    )
    openaq_key = os.getenv("OPENAQ_API_KEY", "").strip()

    secrets_path = ROOT / ".streamlit" / "secrets.toml"
    if secrets_path.exists() and (not url or not key or not openaq_key):
        with secrets_path.open("rb") as f:
            secrets = tomllib.load(f)
        url = url or str(secrets.get("SUPABASE_URL", "")).strip()
        key = key or str(
            secrets.get("SUPABASE_SECRET_KEY", "")
            or secrets.get("SUPABASE_SERVICE_ROLE_KEY", "")
        ).strip()
        openaq_key = openaq_key or str(secrets.get("OPENAQ_API_KEY", "")).strip()

    missing = []
    if not url:
        missing.append("SUPABASE_URL")
    if not key:
        missing.append("SUPABASE_SECRET_KEY")
    if not openaq_key:
        missing.append("OPENAQ_API_KEY")
    if missing:
        raise RuntimeError(
            "Missing configuration: " + ", ".join(missing)
            + ". Keep keys in .streamlit/secrets.toml or environment variables; do not paste them into chat."
        )
    if not url.startswith(("https://", "http://")):
        raise RuntimeError("SUPABASE_URL does not look like a valid URL.")
    return url, key, openaq_key


def get_supabase_client(url: str, key: str):
    try:
        from supabase import create_client
    except ImportError as exc:
        raise RuntimeError("Missing Supabase package. Run: python -m pip install supabase") from exc
    return create_client(url, key)


def fetch_all_forecasts(client) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    start = 0
    while True:
        response = (
            client.table("forecast_snapshots")
            .select("provider,location,issued_at,valid_at,predicted_pm25")
            .range(start, start + PAGE_SIZE - 1)
            .execute()
        )
        batch = response.data or []
        rows.extend(batch)
        if len(batch) < PAGE_SIZE:
            break
        start += PAGE_SIZE
    return rows


def api_get(path: str, api_key: str, params: dict[str, Any]) -> dict[str, Any]:
    url = f"{BASE_URL}{path}?{urlencode(params)}"
    request = Request(url, headers={
        "X-API-Key": api_key,
        "Accept": "application/json",
        "User-Agent": "AirAware-CrossSensorCheck/1.0",
    })
    try:
        with urlopen(request, timeout=30) as response:
            return json.loads(response.read().decode("utf-8"))
    except HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")[:250]
        raise RuntimeError(f"OpenAQ returned HTTP {exc.code} for sensor {path}: {detail}") from exc
    except URLError as exc:
        raise RuntimeError(f"Could not reach OpenAQ: {exc.reason}") from exc


def parse_timestamp(value: Any) -> pd.Timestamp | pd.NaT:
    if isinstance(value, dict):
        value = value.get("utc") or value.get("datetime") or value.get("value")
    if value is None:
        return pd.NaT
    return pd.to_datetime(value, utc=True, errors="coerce")


def fetch_sensor_measurements(sensor_id: str, api_key: str, start: datetime, end: datetime) -> list[dict[str, Any]]:
    """Fetch last 7 days for one sensor, including pagination if needed."""
    result_rows: list[dict[str, Any]] = []
    page = 1
    params_base = {
        "datetime_from": start.isoformat(timespec="seconds"),
        "datetime_to": end.isoformat(timespec="seconds"),
        "limit": 1000,
    }
    while True:
        payload = api_get(
            f"/sensors/{sensor_id}/measurements",
            api_key,
            {**params_base, "page": page},
        )
        results = payload.get("results") or []
        for item in results:
            period = item.get("period") or {}
            period_start = period.get("datetimeFrom") or {}
            raw_time = parse_timestamp(period_start)
            raw_value = item.get("value")
            if pd.isna(raw_time) or raw_value is None:
                continue
            try:
                value = float(raw_value)
            except (ValueError, TypeError):
                continue
            if not math.isfinite(value) or value < 0:
                continue
            parameter = item.get("parameter") or {}
            unit = str(parameter.get("units") or "µg/m³").strip()
            result_rows.append({
                "sensor_id": sensor_id,
                "sensor_name": SENSORS[sensor_id]["name"],
                "distance_km": SENSORS[sensor_id]["distance_km"],
                "observed_at": raw_time,
                "observed_pm25": value,
                "unit": unit,
            })
        print(f"  Sensor {sensor_id}: page {page}, {len(results)} API records")
        if len(results) < params_base["limit"]:
            break
        page += 1
    return result_rows


def select_latest_valid_forecast(rows: list[dict[str, Any]], now: pd.Timestamp, location: str) -> pd.DataFrame:
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    df["provider"] = df["provider"].astype(str).str.strip().str.lower()
    df["location"] = df["location"].astype(str).str.strip().str.lower()
    df["issued_at"] = pd.to_datetime(df["issued_at"], utc=True, errors="coerce")
    df["valid_at"] = pd.to_datetime(df["valid_at"], utc=True, errors="coerce")
    df["predicted_pm25"] = pd.to_numeric(df["predicted_pm25"], errors="coerce")
    df = df[
        (df["provider"] == DEFAULT_PROVIDER)
        & (df["location"] == location.strip().lower())
        & df["issued_at"].notna()
        & df["valid_at"].notna()
        & df["predicted_pm25"].notna()
        & (df["valid_at"] > df["issued_at"])
        & (df["valid_at"] <= now - pd.Timedelta(minutes=MATCH_TOLERANCE_MINUTES))
    ].copy()
    if df.empty:
        return df
    df = df.sort_values(["valid_at", "issued_at"]).drop_duplicates("valid_at", keep="last")
    return df.sort_values("valid_at").reset_index(drop=True)


def fetch_current_sensor_observations_from_supabase(
    client, start: datetime, end: datetime
) -> list[dict[str, Any]]:
    """Read recent sensor 23534 observations already stored in Supabase.

    This includes supplemental metadata readings saved by collect_live_data.py.
    The query is read-only; it does not insert or update any rows.
    """
    result_rows: list[dict[str, Any]] = []
    offset = 0
    while True:
        response = (
            client.table("pm25_observations")
            .select("sensor_id,location,observed_at,observed_pm25,unit")
            .eq("sensor_id", "23534")
            .gte("observed_at", start.isoformat(timespec="seconds"))
            .lte("observed_at", end.isoformat(timespec="seconds"))
            .order("observed_at")
            .range(offset, offset + PAGE_SIZE - 1)
            .execute()
        )
        batch = response.data or []
        for item in batch:
            observed_at = parse_timestamp(item.get("observed_at"))
            raw_value = item.get("observed_pm25")
            if pd.isna(observed_at) or raw_value is None:
                continue
            try:
                value = float(raw_value)
            except (TypeError, ValueError):
                continue
            if not math.isfinite(value) or value < 0:
                continue
            result_rows.append({
                "sensor_id": "23534",
                "sensor_name": SENSORS["23534"]["name"],
                "distance_km": SENSORS["23534"]["distance_km"],
                "observed_at": observed_at,
                "observed_pm25": value,
                "unit": str(item.get("unit") or "µg/m³").strip(),
            })
        if len(batch) < PAGE_SIZE:
            break
        offset += PAGE_SIZE
    return result_rows


def closest_measurement(sensor_obs: pd.DataFrame, target: pd.Timestamp) -> tuple[dict[str, Any] | None, float | None]:
    if sensor_obs.empty:
        return None, None
    deltas = (sensor_obs["observed_at"] - target).dt.total_seconds() / 60.0
    within = deltas.abs() <= MATCH_TOLERANCE_MINUTES
    if not within.any():
        return None, float(deltas.abs().min()) if len(deltas) else None
    choices = sensor_obs.loc[within].copy()
    choices["_offset_min"] = (choices["observed_at"] - target).dt.total_seconds() / 60.0
    # Prefer minimum time gap; on a tie, prefer a measurement at/before the target.
    choices["_abs_offset"] = choices["_offset_min"].abs()
    choices["_is_future"] = choices["_offset_min"] > 0
    chosen = choices.sort_values(["_abs_offset", "_is_future", "observed_at"]).iloc[0]
    row = chosen.drop(labels=["_abs_offset", "_is_future", "_offset_min"]).to_dict()
    return row, float(chosen["_offset_min"])


def score_pairs(pairs: pd.DataFrame) -> dict[str, Any]:
    if pairs.empty:
        return {"n_pairs": 0, "distinct_targets": 0, "mean_actual": np.nan,
                "mean_forecast": np.nan, "MAE": np.nan, "RMSE": np.nan, "bias": np.nan}
    error = pairs["predicted_pm25"].astype(float) - pairs["observed_pm25"].astype(float)
    return {
        "n_pairs": int(len(pairs)),
        "distinct_targets": int(pairs["valid_at"].nunique()),
        "mean_actual": float(pairs["observed_pm25"].mean()),
        "mean_forecast": float(pairs["predicted_pm25"].mean()),
        "MAE": float(error.abs().mean()),
        "RMSE": float(np.sqrt(np.square(error).mean())),
        "bias": float(error.mean()),
    }


def main() -> int:
    print("AirAware sensor-specific forecast evaluation")
    print("-------------------------------------------")
    print("Each sensor is scored only against forecasts requested for its own coordinates.")
    print("Read-only check: no Supabase rows will be changed.")
    try:
        supabase_url, supabase_key, openaq_key = load_settings()
        client = get_supabase_client(supabase_url, supabase_key)
        raw_forecasts = fetch_all_forecasts(client)
    except Exception as exc:
        print(f"ERROR loading forecast data: {exc}", file=sys.stderr)
        return 1

    now = pd.Timestamp(datetime.now(timezone.utc))
    forecast_sets: dict[str, pd.DataFrame] = {}
    print(f"Forecast rows read: {len(raw_forecasts):,}")
    total_matured_targets = 0
    for sensor_id, spec in SENSORS.items():
        sensor_forecasts = select_latest_valid_forecast(
            raw_forecasts, now, location=f"sensor-{sensor_id}"
        )
        forecast_sets[sensor_id] = sensor_forecasts
        n_targets = int(sensor_forecasts["valid_at"].nunique()) if not sensor_forecasts.empty else 0
        total_matured_targets += n_targets
        print(f"Matured sensor-specific forecast targets for {sensor_id} ({spec['name']}): {n_targets}")

    if total_matured_targets == 0:
        print("\nNo matured sensor-specific forecasts are available yet.")
        print("First replace collect_live_data.py with the version that saves locations 'sensor-<ID>'.")
        print("Run the collector, then allow its target hours to pass before evaluating.")
        print("Old central New Delhi forecasts are intentionally not used as substitutes.")
        return 0

    end = datetime.now(timezone.utc) + timedelta(minutes=1)
    start = end - timedelta(days=7)
    all_observations: list[dict[str, Any]] = []
    print(f"\nDownloading the last 7 days of observations for {len(SENSORS)} sensors...")
    for sensor_id in SENSORS:
        try:
            all_observations.extend(fetch_sensor_measurements(sensor_id, openaq_key, start, end))
        except RuntimeError as exc:
            print(f"WARNING: {exc}")

    # OpenAQ's measurements listing can lag metadata for sensor 23534. Reuse
    # stored observations (including metadata supplements) for that sensor.
    try:
        stored_current = fetch_current_sensor_observations_from_supabase(client, start, end)
        all_observations.extend(stored_current)
        print(f"Stored readings reused for sensor 23534: {len(stored_current):,}")
    except Exception as exc:
        print(f"WARNING: could not read stored sensor 23534 observations: {exc}")

    if not all_observations:
        print("No usable sensor measurements were downloaded.")
        return 1

    observation_df = pd.DataFrame(all_observations)
    observation_df["observed_at"] = pd.to_datetime(observation_df["observed_at"], utc=True, errors="coerce")
    observation_df = observation_df.dropna(subset=["observed_at", "observed_pm25"])
    unit_norm = observation_df["unit"].astype(str).str.lower().str.replace(" ", "", regex=False)
    ok_units = unit_norm.isin({"µg/m³", "μg/m³", "ug/m3", "µg/m3", "μg/m3", "micrograms/m3", "microgram/m3"})
    skipped_unit_rows = int((~ok_units).sum())
    observation_df = observation_df[ok_units].copy()
    # Deduplicate identical sensor+timestamp entries; stored metadata supplements
    # are appended last and therefore win only for duplicate timestamp keys.
    observation_df = (
        observation_df.sort_values("observed_at")
        .drop_duplicates(["sensor_id", "observed_at"], keep="last")
        .reset_index(drop=True)
    )

    detail_rows: list[dict[str, Any]] = []
    sensor_stats: list[dict[str, Any]] = []
    for sensor_id, spec in SENSORS.items():
        sensor_forecasts = forecast_sets[sensor_id]
        if sensor_forecasts.empty:
            sensor_stats.append({
                "sensor_id": sensor_id,
                "sensor_name": spec["name"],
                "location_key": f"sensor-{sensor_id}",
                "forecast_targets_available": 0,
                "matched_pairs": 0,
                "distinct_matched_targets": 0,
                "mean_actual": np.nan,
                "mean_forecast": np.nan,
                "MAE": np.nan,
                "RMSE": np.nan,
                "bias": np.nan,
                "latest_matched_actual_time_utc": "",
                "status": "No matured forecast rows",
            })
            continue

        sensor_obs = (
            observation_df[observation_df["sensor_id"].astype(str) == str(sensor_id)]
            .sort_values("observed_at")
            .reset_index(drop=True)
        )
        matched_for_sensor = 0
        for _, forecast in sensor_forecasts.iterrows():
            match, gap = closest_measurement(sensor_obs, forecast["valid_at"])
            if match is None:
                continue
            matched_for_sensor += 1
            detail_rows.append({
                "provider": forecast["provider"],
                "forecast_location_key": forecast["location"],
                "sensor_id": sensor_id,
                "sensor_name": spec["name"],
                "issued_at": forecast["issued_at"].isoformat(),
                "valid_at": forecast["valid_at"].isoformat(),
                "forecast_horizon_hours": (forecast["valid_at"] - forecast["issued_at"]).total_seconds() / 3600.0,
                "predicted_pm25": float(forecast["predicted_pm25"]),
                "observed_pm25": float(match["observed_pm25"]),
                "observed_at": pd.Timestamp(match["observed_at"]).isoformat(),
                "observation_offset_minutes": gap,
                "unit": match["unit"],
            })

        sensor_detail = pd.DataFrame([r for r in detail_rows if r["sensor_id"] == sensor_id])
        metrics = score_pairs(sensor_detail) if not sensor_detail.empty else {
            "n_pairs": 0, "distinct_targets": 0, "mean_actual": np.nan,
            "mean_forecast": np.nan, "MAE": np.nan, "RMSE": np.nan, "bias": np.nan,
        }
        sensor_stats.append({
            "sensor_id": sensor_id,
            "sensor_name": spec["name"],
            "location_key": f"sensor-{sensor_id}",
            "forecast_targets_available": int(sensor_forecasts["valid_at"].nunique()),
            "matched_pairs": int(metrics["n_pairs"]),
            "distinct_matched_targets": int(metrics["distinct_targets"]),
            "mean_actual": metrics["mean_actual"],
            "mean_forecast": metrics["mean_forecast"],
            "MAE": metrics["MAE"],
            "RMSE": metrics["RMSE"],
            "bias": metrics["bias"],
            "latest_matched_actual_time_utc": sensor_detail["observed_at"].max() if not sensor_detail.empty else "",
            "status": "Matched" if matched_for_sensor else "No sensor reading within time tolerance",
        })

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    detail_path = REPORTS_DIR / "sensor_specific_forecast_evaluation.csv"
    summary_path = REPORTS_DIR / "sensor_specific_forecast_summary.csv"
    detail = pd.DataFrame(detail_rows)
    summary = pd.DataFrame(sensor_stats)
    detail.to_csv(detail_path, index=False)
    summary.to_csv(summary_path, index=False)

    print(f"\nCompatible-unit observation rows available: {len(observation_df):,}")
    if skipped_unit_rows:
        print(f"Rows skipped due to unsupported/different unit labels: {skipped_unit_rows:,}")
    print(f"Matched forecast/sensor pairs: {len(detail):,}")
    print("\nSensor-specific forecast accuracy (only the matching sensor forecast is scored):")
    print("sensor_id | sensor | targets available | matched | distinct targets | actual mean | forecast mean | MAE | RMSE | bias | latest actual | status")
    for row in sensor_stats:
        def show_num(value):
            return "n/a" if pd.isna(value) else f"{float(value):.2f}"
        print(
            f"{row['sensor_id']} | {row['sensor_name']} | {row['forecast_targets_available']} | "
            f"{row['matched_pairs']} | {row['distinct_matched_targets']} | {show_num(row['mean_actual'])} | "
            f"{show_num(row['mean_forecast'])} | {show_num(row['MAE'])} | {show_num(row['RMSE'])} | "
            f"{show_num(row['bias'])} | {row['latest_matched_actual_time_utc'] or 'n/a'} | {row['status']}"
        )
    print("\nInterpretation notes:")
    print("- Rows are evaluated only against forecasts requested for the same sensor ID.")
    print("- Each table location key is 'sensor-<ID>'; central 'New Delhi' forecasts are deliberately excluded.")
    print("- Different sensors can differ because they measure different sites, and several coordinates may resolve to the same model grid cell.")
    print("- Scores are preliminary until enough distinct target timestamps have matured and matched.")
    print(f"Detailed matches: {detail_path}")
    print(f"Summary by sensor: {summary_path}")
    print("Read-only evaluation complete; no Supabase rows were changed.")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (RuntimeError, OSError, ValueError) as exc:
        raise SystemExit(f"ERROR: {exc}")

"""Evaluate AirAware's stored Open-Meteo PM2.5 forecasts against OpenAQ readings.

Run from the project root:
    python evaluate_forecasts.py

Credentials are read from environment variables first, then from
.streamlit/secrets.toml. No credentials are printed or written to output files.
"""

from __future__ import annotations

import argparse
import os
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
REPORTS_DIR = ROOT / "reports"
FORECAST_TABLE = "forecast_snapshots"
OBSERVATION_TABLE = "pm25_observations"
DEFAULT_LOCATION = "New Delhi"
DEFAULT_PROVIDER = "open-meteo"
DEFAULT_SOURCE = "OpenAQ"
DEFAULT_SENSOR_ID = "23534"
PAGE_SIZE = 1000


def load_credentials() -> tuple[str, str]:
    """Read Supabase credentials without displaying them."""
    url = os.getenv("SUPABASE_URL", "").strip()
    key = (
        os.getenv("SUPABASE_SECRET_KEY", "").strip()
        or os.getenv("SUPABASE_SERVICE_ROLE_KEY", "").strip()
    )

    secrets_path = ROOT / ".streamlit" / "secrets.toml"
    if (not url or not key) and secrets_path.exists():
        try:
            import tomllib

            with secrets_path.open("rb") as file:
                secrets = tomllib.load(file)
            url = url or str(secrets.get("SUPABASE_URL", "")).strip()
            key = key or str(
                secrets.get("SUPABASE_SECRET_KEY", "")
                or secrets.get("SUPABASE_SERVICE_ROLE_KEY", "")
            ).strip()
        except Exception as exc:
            raise RuntimeError(
                f"Could not read {secrets_path}. Check its TOML formatting."
            ) from exc

    if not url or not key:
        raise RuntimeError(
            "Supabase credentials are missing. Set SUPABASE_URL and "
            "SUPABASE_SECRET_KEY, or keep them in .streamlit/secrets.toml."
        )
    if not url.startswith(("https://", "http://")):
        raise RuntimeError("SUPABASE_URL does not look like a valid URL.")
    return url, key


def get_client():
    try:
        from supabase import create_client
    except ImportError as exc:
        raise RuntimeError(
            "The Supabase Python package is missing. Run: "
            "python -m pip install supabase"
        ) from exc

    url, key = load_credentials()
    return create_client(url, key)


def fetch_all_rows(client: Any, table: str, columns: str) -> list[dict[str, Any]]:
    """Fetch every row using PostgREST range pagination."""
    collected: list[dict[str, Any]] = []
    start = 0
    while True:
        response = (
            client.table(table)
            .select(columns)
            .range(start, start + PAGE_SIZE - 1)
            .execute()
        )
        batch = response.data or []
        collected.extend(batch)
        if len(batch) < PAGE_SIZE:
            break
        start += PAGE_SIZE
    return collected


def parse_utc(series: pd.Series) -> pd.Series:
    return pd.to_datetime(series, utc=True, errors="coerce")


def datetime_series_to_ns(series: pd.Series) -> np.ndarray:
    """Return UTC datetimes as integer nanoseconds, independent of pandas resolution.

    Pandas may store parsed timestamps at microsecond resolution. Converting that
    backing array directly to int64 and treating the result as nanoseconds causes
    timestamp comparisons to be wrong by a factor of 1,000. Normalize explicitly.
    """
    return series.dt.as_unit("ns").astype("int64").to_numpy()


def metrics_for(actual: pd.Series, predicted: pd.Series) -> dict[str, float]:
    errors = pd.to_numeric(predicted, errors="coerce") - pd.to_numeric(
        actual, errors="coerce"
    )
    errors = errors.replace([np.inf, -np.inf], np.nan).dropna()
    if errors.empty:
        return {"MAE": np.nan, "RMSE": np.nan, "Bias": np.nan}
    return {
        "MAE": float(errors.abs().mean()),
        "RMSE": float(np.sqrt(np.square(errors).mean())),
        # Bias = prediction - observation; positive means overprediction.
        "Bias": float(errors.mean()),
    }


def add_actual_matches(
    forecasts: pd.DataFrame,
    observations: pd.DataFrame,
    tolerance_minutes: int,
) -> pd.DataFrame:
    """Match each forecast to the nearest observation within a time tolerance."""
    forecasts = forecasts.copy()
    forecasts["observed_pm25"] = np.nan
    forecasts["matched_observed_at"] = pd.Series(pd.NaT, index=forecasts.index, dtype="datetime64[ns, UTC]")
    forecasts["observation_offset_minutes"] = np.nan

    if forecasts.empty or observations.empty:
        return forecasts

    obs = observations.sort_values("observed_at").dropna(
        subset=["observed_at", "observed_pm25"]
    )
    obs_times_ns = datetime_series_to_ns(obs["observed_at"])
    obs_values = obs["observed_pm25"].astype(float).to_numpy()
    tolerance_ns = int(tolerance_minutes * 60 * 1_000_000_000)

    valid_rows = forecasts[forecasts["valid_at"].notna()].copy()
    for row_index, valid_at in valid_rows["valid_at"].items():
        target_ns = int(valid_at.value)
        insertion = int(np.searchsorted(obs_times_ns, target_ns))
        candidates = []
        if insertion > 0:
            candidates.append(insertion - 1)
        if insertion < len(obs_times_ns):
            candidates.append(insertion)
        if not candidates:
            continue

        # On an exact tie, prefer the observation at/before the forecast valid time.
        best = min(
            candidates,
            key=lambda idx: (
                abs(int(obs_times_ns[idx]) - target_ns),
                int(obs_times_ns[idx]) > target_ns,
            ),
        )
        delta_ns = int(obs_times_ns[best]) - target_ns
        if abs(delta_ns) <= tolerance_ns:
            forecasts.at[row_index, "observed_pm25"] = float(obs_values[best])
            forecasts.at[row_index, "matched_observed_at"] = pd.Timestamp(
                int(obs_times_ns[best]), tz="UTC"
            )
            forecasts.at[row_index, "observation_offset_minutes"] = delta_ns / 60_000_000_000

    return forecasts



def print_matching_diagnostics(
    forecasts: pd.DataFrame, observations: pd.DataFrame, tolerance_minutes: int
) -> None:
    """Explain why timestamp ranges may overlap without any target matches."""
    if forecasts.empty or observations.empty:
        return

    target_times = forecasts["valid_at"].dropna().sort_values()
    obs = observations.dropna(subset=["observed_at", "observed_pm25"]).sort_values("observed_at")
    if target_times.empty or obs.empty:
        return

    target_start = target_times.min() - pd.Timedelta(minutes=tolerance_minutes)
    target_end = target_times.max() + pd.Timedelta(minutes=tolerance_minutes)
    in_window = obs[(obs["observed_at"] >= target_start) & (obs["observed_at"] <= target_end)]

    obs_ns = datetime_series_to_ns(obs["observed_at"])
    rows = []
    for idx, target in forecasts["valid_at"].dropna().items():
        target_ns = int(target.value)
        pos = int(np.searchsorted(obs_ns, target_ns))
        candidates = []
        if pos > 0:
            candidates.append(pos - 1)
        if pos < len(obs_ns):
            candidates.append(pos)
        if not candidates:
            continue
        best = min(candidates, key=lambda j: abs(int(obs_ns[j]) - target_ns))
        delta_minutes = (int(obs_ns[best]) - target_ns) / 60_000_000_000
        rows.append((abs(delta_minutes), target, obs.iloc[best]["observed_at"], delta_minutes))

    print("\nNearest-observation diagnostics")
    print("-------------------------------")
    print(f"Observations within matured-target span (±{tolerance_minutes} min): {len(in_window):,}")
    if rows:
        distances = np.array([r[0] for r in rows], dtype=float)
        print(f"Forecast target rows with a nearest observation: {len(rows):,}")
        print(f"Closest timestamp gap: {distances.min():.1f} minutes")
        print(f"Median nearest timestamp gap: {np.median(distances):.1f} minutes")
        print(f"Targets with a nearest observation within tolerance: {int((distances <= tolerance_minutes).sum()):,}")
        print("Five closest target/observation pairs:")
        for _, target, observed_at, delta in sorted(rows, key=lambda item: item[0])[:5]:
            print(
                f"  target={target.isoformat()} | observation={observed_at.isoformat()} "
                f"| offset={delta:+.1f} min"
            )

def add_persistence_baseline(
    forecasts: pd.DataFrame,
    observations: pd.DataFrame,
    max_age_hours: float,
) -> pd.DataFrame:
    """Use the most recent observation available when each forecast was issued."""
    forecasts = forecasts.copy()
    forecasts["persistence_pm25"] = np.nan
    forecasts["persistence_observed_at"] = pd.Series(pd.NaT, index=forecasts.index, dtype="datetime64[ns, UTC]")
    forecasts["persistence_age_hours"] = np.nan

    obs = observations.sort_values("observed_at").dropna(
        subset=["observed_at", "observed_pm25"]
    )
    if forecasts.empty or obs.empty:
        return forecasts

    obs_times_ns = datetime_series_to_ns(obs["observed_at"])
    obs_values = obs["observed_pm25"].astype(float).to_numpy()
    max_age_ns = int(max_age_hours * 3600 * 1_000_000_000)

    for row_index, issued_at in forecasts["issued_at"].items():
        if pd.isna(issued_at):
            continue
        target_ns = int(issued_at.value)
        insertion = int(np.searchsorted(obs_times_ns, target_ns, side="right")) - 1
        if insertion < 0:
            continue
        age_ns = target_ns - int(obs_times_ns[insertion])
        if age_ns < 0 or age_ns > max_age_ns:
            continue
        forecasts.at[row_index, "persistence_pm25"] = float(obs_values[insertion])
        forecasts.at[row_index, "persistence_observed_at"] = pd.Timestamp(
            int(obs_times_ns[insertion]), tz="UTC"
        )
        forecasts.at[row_index, "persistence_age_hours"] = age_ns / 3_600_000_000_000
    return forecasts


def horizon_group(value: Any) -> str:
    try:
        hours = float(value)
    except (TypeError, ValueError):
        return "Unknown"
    if not np.isfinite(hours) or hours < 0:
        return "Unknown"
    if hours < 3:
        return "0–3 h"
    if hours < 6:
        return "3–6 h"
    if hours < 9:
        return "6–9 h"
    if hours < 12:
        return "9–12 h"
    if hours < 24:
        return "12–24 h"
    return "24+ h"


def latest_issue_per_target(detail: pd.DataFrame) -> pd.DataFrame:
    """Select one forecast per valid hour: the latest issue before that target."""
    if detail.empty:
        return detail.copy()
    return (
        detail.sort_values(["valid_at", "issued_at"])
        .drop_duplicates(subset=["valid_at"], keep="last")
        .copy()
    )


def build_metrics(detail: pd.DataFrame) -> pd.DataFrame:
    """Score a deduplicated forecast sample overall and within lead-time bands.

    Each valid_at contributes no more than one forecast to a given row: the
    latest-issued forecast in the target's overall set or in that horizon band.
    This prevents targets with many saved snapshots from dominating the metrics.
    """
    rows: list[dict[str, Any]] = []
    groups = [("Latest forecast per target (overall)", latest_issue_per_target(detail))]
    ordered_groups = ["0–3 h", "3–6 h", "6–9 h", "9–12 h", "12–24 h", "24+ h", "Unknown"]
    for group_name in ordered_groups:
        subset = detail[detail["horizon_group"] == group_name]
        if not subset.empty:
            groups.append((group_name, latest_issue_per_target(subset)))

    for group_name, subset in groups:
        matched = subset.dropna(subset=["observed_pm25", "predicted_pm25"]).copy()
        forecast_metrics = metrics_for(matched["observed_pm25"], matched["predicted_pm25"])
        baseline_matched = matched.dropna(subset=["persistence_pm25"])
        baseline_metrics = metrics_for(
            baseline_matched["observed_pm25"], baseline_matched["persistence_pm25"]
        )
        # Compare the forecast and persistence on exactly the same selected targets.
        paired_forecast_metrics = metrics_for(
            baseline_matched["observed_pm25"], baseline_matched["predicted_pm25"]
        )
        rows.append({
            "Evaluation group": group_name,
            "Forecast targets scored (n)": int(matched["valid_at"].nunique()),
            "Distinct actual timestamps (n)": int(matched["matched_observed_at"].nunique()),
            "Forecast MAE (µg/m³)": forecast_metrics["MAE"],
            "Forecast RMSE (µg/m³)": forecast_metrics["RMSE"],
            "Forecast bias (µg/m³)": forecast_metrics["Bias"],
            "Paired forecast MAE (µg/m³)": paired_forecast_metrics["MAE"],
            "Paired forecast RMSE (µg/m³)": paired_forecast_metrics["RMSE"],
            "Persistence targets scored (n)": int(baseline_matched["valid_at"].nunique()),
            "Persistence MAE (µg/m³)": baseline_metrics["MAE"],
            "Persistence RMSE (µg/m³)": baseline_metrics["RMSE"],
            "Persistence bias (µg/m³)": baseline_metrics["Bias"],
            "MAE improvement vs persistence (%)": (
                100.0 * (baseline_metrics["MAE"] - paired_forecast_metrics["MAE"]) / baseline_metrics["MAE"]
                if len(baseline_matched) and pd.notna(baseline_metrics["MAE"]) and baseline_metrics["MAE"] != 0
                and pd.notna(paired_forecast_metrics["MAE"])
                else np.nan
            ),
        })
    return pd.DataFrame(rows)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Evaluate saved AirAware PM2.5 forecasts against OpenAQ observations."
    )
    parser.add_argument(
        "--match-tolerance-minutes", type=int, default=45,
        help="Maximum absolute time difference for matching an observation to forecast valid_at (default: 45).",
    )
    parser.add_argument(
        "--persistence-max-age-hours", type=float, default=6.0,
        help="Maximum age of the last reading available at forecast issue time (default: 6).",
    )
    parser.add_argument(
        "--location", default=DEFAULT_LOCATION,
        help=f"Location recorded in Supabase (default: {DEFAULT_LOCATION}).",
    )
    parser.add_argument(
        "--provider", default=DEFAULT_PROVIDER,
        help=f"Forecast provider in Supabase (default: {DEFAULT_PROVIDER}).",
    )
    parser.add_argument(
        "--source", default=DEFAULT_SOURCE,
        help=f"Observation source in Supabase (default: {DEFAULT_SOURCE}).",
    )
    parser.add_argument(
        "--sensor-id", default=DEFAULT_SENSOR_ID,
        help=f"OpenAQ sensor ID to evaluate (default: {DEFAULT_SENSOR_ID}); use all to skip sensor filtering.",
    )
    args = parser.parse_args()

    if args.match_tolerance_minutes < 0:
        parser.error("--match-tolerance-minutes must be zero or greater.")
    if args.persistence_max_age_hours < 0:
        parser.error("--persistence-max-age-hours must be zero or greater.")

    print("AirAware live forecast evaluation")
    print(f"Location: {args.location} | Provider: {args.provider} | Observation source: {args.source}")
    print("Connecting to Supabase and reading stored rows...")

    try:
        client = get_client()
        forecast_rows = fetch_all_rows(
            client,
            FORECAST_TABLE,
            "provider,location,issued_at,valid_at,forecast_horizon_hours,predicted_pm25",
        )
        observation_rows = fetch_all_rows(
            client,
            OBSERVATION_TABLE,
            "source,sensor_id,location,observed_at,observed_pm25,unit",
        )
    except Exception as exc:
        print(f"ERROR: Could not load data from Supabase: {exc}", file=sys.stderr)
        return 1

    print(f"Forecast rows downloaded: {len(forecast_rows):,}")
    print(f"Observation rows downloaded: {len(observation_rows):,}")

    forecasts = pd.DataFrame(forecast_rows)
    observations = pd.DataFrame(observation_rows)

    if forecasts.empty:
        print("No rows were found in forecast_snapshots. Run the collection workflow first.")
        return 0
    if observations.empty:
        print("No rows were found in pm25_observations. Run the observation sync first.")
        return 0

    forecasts["provider"] = forecasts["provider"].astype(str).str.strip().str.lower()
    forecasts["location"] = forecasts["location"].astype(str).str.strip().str.lower()
    forecasts["issued_at"] = parse_utc(forecasts["issued_at"])
    forecasts["valid_at"] = parse_utc(forecasts["valid_at"])
    forecasts["predicted_pm25"] = pd.to_numeric(forecasts["predicted_pm25"], errors="coerce")
    forecasts["forecast_horizon_hours"] = pd.to_numeric(
        forecasts["forecast_horizon_hours"], errors="coerce"
    )

    observations["source"] = observations["source"].astype(str).str.strip().str.lower()
    observations["location"] = observations["location"].astype(str).str.strip().str.lower()
    observations["sensor_id"] = observations["sensor_id"].astype(str).str.strip()
    observations["observed_at"] = parse_utc(observations["observed_at"])
    observations["observed_pm25"] = pd.to_numeric(observations["observed_pm25"], errors="coerce")

    forecasts = forecasts[
        (forecasts["provider"] == args.provider.strip().lower())
        & (forecasts["location"] == args.location.strip().lower())
        & forecasts["issued_at"].notna()
        & forecasts["valid_at"].notna()
        & forecasts["predicted_pm25"].notna()
    ].copy()

    # Older collector runs may have saved an hourly target that was already
    # past by the time the forecast was recorded. Exclude these legacy rows;
    # clipping their lead time to zero would make them look like valid nowcasts.
    invalid_timing_mask = forecasts["valid_at"] <= forecasts["issued_at"]
    invalid_timing_count = int(invalid_timing_mask.sum())
    forecasts = forecasts.loc[~invalid_timing_mask].copy()
    print(
        "Legacy forecast rows excluded (valid_at <= issued_at): "
        f"{invalid_timing_count:,}"
    )

    # Recompute lead time from timestamps so old or clipped stored values cannot
    # misclassify rows into the wrong horizon group.
    forecasts["forecast_horizon_hours"] = (
        forecasts["valid_at"] - forecasts["issued_at"]
    ).dt.total_seconds() / 3600.0

    observation_mask = (
        (observations["source"] == args.source.strip().lower())
        & (observations["location"] == args.location.strip().lower())
        & observations["observed_at"].notna()
        & observations["observed_pm25"].notna()
    )
    if args.sensor_id.strip().lower() != "all":
        observation_mask &= observations["sensor_id"] == args.sensor_id.strip()
    observations = observations[observation_mask].copy()

    print("\nTimestamp diagnostics (UTC)")
    print("---------------------------")
    if not forecasts.empty:
        print(
            "Stored forecast targets: "
            f"{forecasts['valid_at'].min().isoformat()} to {forecasts['valid_at'].max().isoformat()}"
        )
        print(
            "Forecast issue times:    "
            f"{forecasts['issued_at'].min().isoformat()} to {forecasts['issued_at'].max().isoformat()}"
        )
    else:
        print("Stored forecast targets: none remain after provider/location filters")
    if not observations.empty:
        print(
            "OpenAQ observation times: "
            f"{observations['observed_at'].min().isoformat()} to {observations['observed_at'].max().isoformat()}"
        )
        print(f"Filtered OpenAQ observations: {len(observations):,}")
    else:
        print("OpenAQ observation times: none remain after source/location/sensor filters")

    # Restrict to forecasts whose target times have matured. Matching still requires
    # an actual observation within the defined tolerance below.
    now_utc = pd.Timestamp(datetime.now(timezone.utc))
    maturity_cutoff = now_utc - pd.Timedelta(minutes=args.match_tolerance_minutes)
    before_maturity_count = len(forecasts)
    forecasts = forecasts[forecasts["valid_at"] <= maturity_cutoff].copy()
    matured_excluded = before_maturity_count - len(forecasts)
    if not forecasts.empty:
        print(
            "Matured forecast targets: "
            f"{forecasts['valid_at'].min().isoformat()} to {forecasts['valid_at'].max().isoformat()}"
        )

    if forecasts.empty:
        print("No matured forecasts yet. Wait for target hours to pass, then run this again.")
        return 0
    if observations.empty:
        print("No observations remain after source/location/sensor filters. Check the table values.")
        print("Hint: the script defaults to source=OpenAQ and sensor_id=23534; use --sensor-id all if needed.")
        return 0

    print_matching_diagnostics(forecasts, observations, args.match_tolerance_minutes)

    forecasts["horizon_group"] = forecasts["forecast_horizon_hours"].map(horizon_group)

    observations = observations.sort_values("observed_at").drop_duplicates(
        subset=["observed_at"], keep="last"
    )
    matched = add_actual_matches(forecasts, observations, args.match_tolerance_minutes)
    matched = add_persistence_baseline(matched, observations, args.persistence_max_age_hours)

    # Keep only rows with an actual match for the evaluation report; the full match
    # rate is separately reported so missing observations are visible.
    total_matured = len(matched)
    evaluated = matched.dropna(subset=["observed_pm25", "predicted_pm25"]).copy()
    match_rate = (100.0 * len(evaluated) / total_matured) if total_matured else 0.0

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    detail_path = REPORTS_DIR / "live_forecast_evaluation.csv"
    metrics_path = REPORTS_DIR / "live_forecast_metrics.csv"

    detail_columns = [
        "provider", "location", "issued_at", "valid_at", "forecast_horizon_hours",
        "horizon_group", "predicted_pm25", "observed_pm25", "matched_observed_at",
        "observation_offset_minutes", "persistence_pm25", "persistence_observed_at",
        "persistence_age_hours",
    ]
    matched[detail_columns].sort_values(["valid_at", "issued_at"]).to_csv(
        detail_path, index=False, date_format="%Y-%m-%dT%H:%M:%SZ"
    )

    if evaluated.empty:
        print("\nNo forecast rows matched observations within the tolerance yet.")
        print(f"Matured forecasts checked: {total_matured:,}")
        print(f"Allowed matching difference: ±{args.match_tolerance_minutes} minutes")
        print("Try again after readings arrive for the forecast target times. Compare the UTC timestamp ranges printed above.")
        print("If the ranges overlap, review the source/location/sensor filters and matching tolerance.")
        print(f"Match detail written to: {detail_path}")
        return 0

    metrics = build_metrics(evaluated)
    metrics.to_csv(metrics_path, index=False)

    all_row = metrics.iloc[0]
    print("\nEvaluation summary")
    print("------------------")
    print(f"Matured forecast rows checked: {total_matured:,} (excluded future/not-yet-matured: {matured_excluded:,})")
    unique_target_count = int(evaluated["valid_at"].nunique())
    unique_observation_count = int(evaluated["matched_observed_at"].nunique())
    print(f"Raw matched forecast rows: {len(evaluated):,} / {total_matured:,} ({match_rate:.1f}%)")
    print(f"Distinct forecast target timestamps matched: {unique_target_count:,}")
    print(f"Distinct observation timestamps used: {unique_observation_count:,}")
    print(f"Matching tolerance: ±{args.match_tolerance_minutes} minutes")
    print(f"Persistence reading age limit: {args.persistence_max_age_hours:g} hours")
    print("Primary score uses one latest-issued forecast per target timestamp.")
    if pd.notna(all_row["Forecast MAE (µg/m³)"]):
        print(f"Unique-target forecast MAE: {all_row['Forecast MAE (µg/m³)']:.2f} µg/m³")
        print(f"Unique-target forecast RMSE: {all_row['Forecast RMSE (µg/m³)']:.2f} µg/m³")
        print(f"Unique-target forecast bias: {all_row['Forecast bias (µg/m³)']:+.2f} µg/m³")

    if pd.notna(all_row["Persistence MAE (µg/m³)"]):
        print(f"Paired persistence MAE: {all_row['Persistence MAE (µg/m³)']:.2f} µg/m³")
        print(f"Paired persistence RMSE: {all_row['Persistence RMSE (µg/m³)']:.2f} µg/m³")
        improvement = all_row["MAE improvement vs persistence (%)"]
        if pd.notna(improvement):
            comparison = "better" if improvement > 0 else "worse" if improvement < 0 else "equal to"
            print(f"Unique-target forecast MAE vs persistence: {abs(improvement):.1f}% {comparison}")
    else:
        print("Persistence metrics unavailable: no recent enough observation existed at issue time for the matched targets.")

    print("\nMetrics by forecast horizon (one forecast per target per group)")
    print(metrics.to_string(index=False, formatters={
        "Forecast MAE (µg/m³)": lambda x: "—" if pd.isna(x) else f"{x:.2f}",
        "Forecast RMSE (µg/m³)": lambda x: "—" if pd.isna(x) else f"{x:.2f}",
        "Forecast bias (µg/m³)": lambda x: "—" if pd.isna(x) else f"{x:+.2f}",
        "Paired forecast MAE (µg/m³)": lambda x: "—" if pd.isna(x) else f"{x:.2f}",
        "Paired forecast RMSE (µg/m³)": lambda x: "—" if pd.isna(x) else f"{x:.2f}",
        "Persistence MAE (µg/m³)": lambda x: "—" if pd.isna(x) else f"{x:.2f}",
        "Persistence RMSE (µg/m³)": lambda x: "—" if pd.isna(x) else f"{x:.2f}",
        "Persistence bias (µg/m³)": lambda x: "—" if pd.isna(x) else f"{x:+.2f}",
        "MAE improvement vs persistence (%)": lambda x: "—" if pd.isna(x) else f"{x:+.1f}%",
    }))
    print(f"\nDetailed matches: {detail_path}")
    print(f"Metrics summary:  {metrics_path}")
    print("Note: positive bias means the forecast overpredicted PM2.5; negative bias means it underpredicted.")
    if evaluated["valid_at"].nunique() < 50:
        print("CAUTION: fewer than 50 distinct target timestamps. Treat these metrics as preliminary, not as a reliable model ranking.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

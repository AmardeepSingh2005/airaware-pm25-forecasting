import tomllib
from pathlib import Path

import pandas as pd
from supabase import create_client


ROOT = Path(__file__).resolve().parent
CSV_PATH = ROOT / "data/raw/pm25_new_delhi_recent_90d.csv"
SECRETS_PATH = ROOT / ".streamlit/secrets.toml"
BATCH_SIZE = 500


def main():
    if not CSV_PATH.exists():
        raise FileNotFoundError(f"CSV not found: {CSV_PATH}")

    if not SECRETS_PATH.exists():
        raise FileNotFoundError("Local Supabase secrets file not found.")

    with SECRETS_PATH.open("rb") as file:
        secrets = tomllib.load(file)

    url = secrets.get("SUPABASE_URL")
    key = (
        secrets.get("SUPABASE_SECRET_KEY")
        or secrets.get("SUPABASE_SERVICE_ROLE_KEY")
    )

    if not url or not key:
        raise ValueError("Supabase URL or backend secret key is missing.")

    df = pd.read_csv(CSV_PATH)

    required = {"datetime_utc", "pm25"}
    if not required.issubset(df.columns):
        raise ValueError(
            f"CSV must contain these columns: {sorted(required)}"
        )

    df["observed_at"] = pd.to_datetime(
        df["datetime_utc"], utc=True, errors="coerce"
    )
    df["observed_pm25"] = pd.to_numeric(
        df["pm25"], errors="coerce"
    )

    df = df.dropna(subset=["observed_at", "observed_pm25"])
    df = df[df["observed_pm25"] >= 0].copy()

    if "sensor_id" in df.columns:
        df["sensor_id"] = df["sensor_id"].fillna(23534).astype(str)
    else:
        df["sensor_id"] = "23534"

    if "unit" in df.columns:
        df["unit"] = df["unit"].fillna("µg/m³").astype(str)
    else:
        df["unit"] = "µg/m³"

    # Match the database's unique key and make reruns safe.
    df = (
        df.sort_values("observed_at")
        .drop_duplicates(
            subset=["sensor_id", "observed_at"], keep="last"
        )
    )

    df["observed_at"] = df["observed_at"].dt.strftime(
        "%Y-%m-%dT%H:%M:%SZ"
    )

    records = []
    for row in df.to_dict("records"):
        records.append({
            "source": "OpenAQ",
            "sensor_id": row["sensor_id"],
            "location": "New Delhi",
            "observed_at": row["observed_at"],
            "observed_pm25": float(row["observed_pm25"]),
            "unit": row["unit"],
        })

    client = create_client(url, key)

    for start in range(0, len(records), BATCH_SIZE):
        batch = records[start:start + BATCH_SIZE]
        client.table("pm25_observations").upsert(
            batch,
            on_conflict="source,sensor_id,observed_at",
        ).execute()
        print(
            f"Processed batch ending at "
            f"{min(start + len(batch), len(records)):,}"
            f" / {len(records):,}"
        )

    print(f"SUCCESS: {len(records):,} observations synced to Supabase.")


if __name__ == "__main__":
    main()
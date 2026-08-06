"""
P1 — Station eligibility pipeline.

Reads hourly PM10 and PM2.5 observations for the 2023–2025 period and
produces:
  data/metadata/station_eligibility_2023_2025.csv  (per station-pollutant-year)
  data/metadata/station_summary.csv                (aggregated per station)

Rules (all thresholds come from configs/station_eligibility.yaml):
  - Does NOT impute missing values.
  - Does NOT silently convert negatives or zeros to NaN.
  - Flags negative values and duplicate timestamps.
  - Records format differences between input files.
  - Never uses value-at-the-moment-of-running to infer historical continuity.

Input contract (per station file):
  Columns: datetime, station_id, pollutant, value
  datetime: ISO-8601 hourly, UTC or offset-aware
  pollutant: "PM10" or "PM2.5"
  value: float (µg m⁻³); NaN allowed for missing; may contain negatives

Usage:
  python scripts/p1_01_compute_station_eligibility.py \\
      --obs-dir data/raw/stations/gva \\
      --config configs/station_eligibility.yaml \\
      --manifest data/raw/stations/stations_manifest.csv \\
      --out-dir data/metadata
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]


# ---------------------------------------------------------------------------
# Core computation
# ---------------------------------------------------------------------------

def expected_hourly_records(year: int) -> int:
    """Number of expected hourly records in a given year (8760 or 8784)."""
    import calendar
    return 24 * sum(
        calendar.monthrange(year, m)[1] for m in range(1, 13)
    )


def compute_longest_gap_days(series: pd.Series) -> float:
    """
    Compute the longest consecutive run of NaN values in an hourly series,
    expressed in days.
    """
    is_missing = series.isna()
    max_run = 0
    current_run = 0
    for v in is_missing:
        if v:
            current_run += 1
            max_run = max(max_run, current_run)
        else:
            current_run = 0
    return max_run / 24.0


def analyse_station_pollutant_year(
    df_station: pd.DataFrame,
    station_id: str,
    pollutant: str,
    year: int,
    cfg: dict[str, Any],
    format_notes: list[str],
) -> dict[str, Any]:
    """
    Compute eligibility metrics for one (station, pollutant, year) triplet.

    Parameters
    ----------
    df_station:
        Hourly series for this station and pollutant, with a DatetimeIndex (UTC).
    station_id, pollutant, year:
        Group keys.
    cfg:
        Parsed station_eligibility.yaml.
    format_notes:
        List of format-change notes accumulated across files.
    """
    cov_cfg = cfg["coverage"]
    qual_cfg = cfg["quality"]
    q_cfg = cfg["quantile_records"]

    subset = df_station[
        (df_station.index.year == year) & ~df_station.index.duplicated(keep="first")
    ]

    # Expected records
    exp = expected_hourly_records(year)

    # Duplicate detection (before deduplication)
    dup_count = int(df_station[df_station.index.year == year].index.duplicated().sum())

    # Valid records: non-NaN, non-negative (per quality policy)
    valid_mask = subset["value"].notna()
    neg_count = int((subset["value"].dropna() < 0).sum())

    # Flag but do NOT remove negatives — caller sees them in the output
    valid_records = int(valid_mask.sum())
    coverage = valid_records / exp if exp > 0 else 0.0

    # Gap analysis
    full_index = pd.date_range(
        f"{year}-01-01", f"{year}-12-31 23:00", freq="h", tz="UTC"
    )
    reindexed = subset["value"].reindex(full_index)
    longest_gap_days = compute_longest_gap_days(reindexed)

    # Quantile record counts (observations strictly above train-candidate thresholds)
    values = subset["value"].dropna()
    q90_count = int((values >= values.quantile(0.90)).sum()) if len(values) > 0 else 0
    q95_count = int((values >= values.quantile(0.95)).sum()) if len(values) > 0 else 0

    # Eligibility decision
    eligible = (
        coverage >= cov_cfg["min_annual_per_pollutant"]
        and longest_gap_days <= cov_cfg["max_gap_days"]
    )
    exclusion_reason = ""
    if not eligible:
        reasons = []
        if coverage < cov_cfg["min_annual_per_pollutant"]:
            reasons.append(
                f"coverage={coverage:.3f} < {cov_cfg['min_annual_per_pollutant']}"
            )
        if longest_gap_days > cov_cfg["max_gap_days"]:
            reasons.append(
                f"gap={longest_gap_days:.1f}d > {cov_cfg['max_gap_days']}d"
            )
        exclusion_reason = "; ".join(reasons)

    return {
        "station_id": station_id,
        "pollutant": pollutant,
        "year": year,
        "expected_records": exp,
        "valid_records": valid_records,
        "coverage": round(coverage, 4),
        "longest_missing_gap_days": round(longest_gap_days, 2),
        "q90_train_candidate_count": q90_count,
        "q95_train_candidate_count": q95_count,
        "negative_value_count": neg_count,
        "duplicate_timestamp_count": dup_count,
        "eligible_pollutant_year": eligible,
        "exclusion_reason": exclusion_reason,
    }


def compute_station_summary(
    detail: pd.DataFrame,
    cfg: dict[str, Any],
) -> pd.DataFrame:
    """
    Aggregate per-station: coverage across pollutants and years, joint coverage,
    max gap, eligibility decisions.
    """
    cov_cfg = cfg["coverage"]
    years = cfg["period"]["years"]
    rows = []

    for station_id, grp in detail.groupby("station_id"):
        station_rows: dict[str, Any] = {"station_id": station_id}

        for year in years:
            for pollutant in cfg["pollutants"]:
                match = grp[
                    (grp["year"] == year) & (grp["pollutant"] == pollutant)
                ]
                key = f"coverage_{pollutant.lower().replace('.', '')}_{year}"
                station_rows[key] = (
                    float(match["coverage"].iloc[0]) if len(match) > 0 else np.nan
                )

        # Joint coverage (mean across pollutants and years)
        cov_cols = [
            c for c in station_rows
            if c.startswith("coverage_") and c != "station_id"
        ]
        station_rows["coverage_joint_mean"] = float(
            np.nanmean([station_rows[c] for c in cov_cols])
        )

        station_rows["max_gap_days"] = float(
            grp["longest_missing_gap_days"].max()
        )
        station_rows["total_neg_values"] = int(
            grp["negative_value_count"].sum()
        )
        station_rows["total_dup_timestamps"] = int(
            grp["duplicate_timestamp_count"].sum()
        )

        # Primary eligibility: all pollutant-year combos must pass
        all_pass = all(
            bool(grp[
                (grp["year"] == y) & (grp["pollutant"] == p)
            ]["eligible_pollutant_year"].iloc[0])
            if len(grp[(grp["year"] == y) & (grp["pollutant"] == p)]) > 0
            else False
            for y in years
            for p in cfg["pollutants"]
        )
        station_rows["eligible_primary"] = all_pass

        # Sensitivity eligibility
        sens_cfg = cfg["sensitivity"]
        sens_pass = (
            station_rows["coverage_joint_mean"] >= sens_cfg["min_joint_annual"]
            and station_rows["max_gap_days"] <= sens_cfg["max_gap_days"]
        )
        station_rows["eligible_sensitivity"] = sens_pass

        rows.append(station_rows)

    return pd.DataFrame(rows)


def regional_decision(n_eligible: int, cfg: dict[str, Any]) -> str:
    rd = cfg["regional_decision"]
    if n_eligible >= rd["REGIONAL_STRONG"]:
        return "REGIONAL_STRONG"
    if n_eligible >= rd["REGIONAL_BOUNDED"]:
        return "REGIONAL_BOUNDED"
    if n_eligible >= rd["PILOT_ONLY"]:
        return "PILOT_ONLY"
    return "STOP_OR_EXPAND_REGION"


# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------

def load_observations(obs_dir: Path, manifest: pd.DataFrame) -> pd.DataFrame:
    """
    Load all station CSV files found in obs_dir.

    Expected CSV columns: datetime, station_id, pollutant, value
    Records format changes (number of columns, column names) across files.
    Does NOT impute, does NOT convert negatives silently.
    """
    frames = []
    expected_cols = {"datetime", "station_id", "pollutant", "value"}
    format_notes = []
    prev_cols: set | None = None

    for fpath in sorted(obs_dir.glob("*.csv")):
        df = pd.read_csv(fpath, dtype={"station_id": str, "pollutant": str})
        cols = set(df.columns)

        # Detect format change
        if prev_cols is not None and cols != prev_cols:
            format_notes.append(
                f"Format change in {fpath.name}: "
                f"prev={sorted(prev_cols)}, now={sorted(cols)}"
            )
        prev_cols = cols

        missing_cols = expected_cols - cols
        if missing_cols:
            print(
                f"[WARN] {fpath.name} missing columns {missing_cols}, skipping.",
                file=sys.stderr,
            )
            continue

        df["datetime"] = pd.to_datetime(df["datetime"], utc=True, errors="coerce")
        df["value"] = pd.to_numeric(df["value"], errors="coerce")
        df["source_file"] = fpath.name
        frames.append(df)

    if format_notes:
        print("[INFO] Format changes detected:", file=sys.stderr)
        for note in format_notes:
            print(f"  {note}", file=sys.stderr)

    if not frames:
        raise FileNotFoundError(
            f"No valid observation CSV files found in {obs_dir}"
        )

    combined = pd.concat(frames, ignore_index=True)
    combined = combined.set_index("datetime").sort_index()
    return combined


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def run(
    obs_dir: Path,
    config_path: Path,
    manifest_path: Path,
    out_dir: Path,
) -> dict[str, Any]:
    with open(config_path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    manifest = pd.read_csv(manifest_path, dtype={"station_id": str})
    station_ids = manifest["station_id"].tolist()

    obs = load_observations(obs_dir, manifest)
    obs = obs[obs["station_id"].isin(station_ids)]
    obs = obs[obs["pollutant"].isin(cfg["pollutants"])]

    detail_rows = []
    for station_id in station_ids:
        for pollutant in cfg["pollutants"]:
            mask = (obs["station_id"] == station_id) & (obs["pollutant"] == pollutant)
            df_sp = obs[mask][["value"]].copy()
            if df_sp.empty:
                for year in cfg["period"]["years"]:
                    detail_rows.append(
                        {
                            "station_id": station_id,
                            "pollutant": pollutant,
                            "year": year,
                            "expected_records": expected_hourly_records(year),
                            "valid_records": 0,
                            "coverage": 0.0,
                            "longest_missing_gap_days": expected_hourly_records(year) / 24.0,
                            "q90_train_candidate_count": 0,
                            "q95_train_candidate_count": 0,
                            "negative_value_count": 0,
                            "duplicate_timestamp_count": 0,
                            "eligible_pollutant_year": False,
                            "exclusion_reason": "no data available",
                        }
                    )
                continue
            for year in cfg["period"]["years"]:
                row = analyse_station_pollutant_year(
                    df_sp, station_id, pollutant, year, cfg, []
                )
                detail_rows.append(row)

    detail = pd.DataFrame(detail_rows)
    summary = compute_station_summary(detail, cfg)

    out_dir.mkdir(parents=True, exist_ok=True)
    detail_path = out_dir / "station_eligibility_2023_2025.csv"
    summary_path = out_dir / "station_summary.csv"
    detail.to_csv(detail_path, index=False)
    summary.to_csv(summary_path, index=False)

    n_eligible = int(summary["eligible_primary"].sum())
    decision = regional_decision(n_eligible, cfg)

    result = {
        "stations_total": len(station_ids),
        "stations_eligible_primary": n_eligible,
        "stations_eligible_sensitivity": int(summary["eligible_sensitivity"].sum()),
        "regional_decision": decision,
        "detail_path": str(detail_path),
        "summary_path": str(summary_path),
    }
    print(json.dumps(result, indent=2))
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Station eligibility pipeline")
    parser.add_argument("--obs-dir", required=True, type=Path)
    parser.add_argument("--config", default=ROOT / "configs" / "station_eligibility.yaml", type=Path)
    parser.add_argument("--manifest", default=ROOT / "data" / "raw" / "stations" / "stations_manifest.csv", type=Path)
    parser.add_argument("--out-dir", default=ROOT / "data" / "metadata", type=Path)
    args = parser.parse_args()
    run(args.obs_dir, args.config, args.manifest, args.out_dir)


if __name__ == "__main__":
    main()

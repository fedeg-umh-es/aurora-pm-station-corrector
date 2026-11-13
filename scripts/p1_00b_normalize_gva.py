"""
P1 — Normalise GVA monthly hourly files into the tidy contract used by
`p1_01_compute_station_eligibility.py` and build the station manifest.

GVA format (observed 2023-2025): ';' separated, decimal comma, '-' = missing,
FECHA=YYYY-MM-DD, HORA=00..23, one column per pollutant, no validation flag.

Time convention (inferred, see docs/gva_source_notes.md): every day has exactly
24 rows, including DST-change days, so timestamps are a fixed offset. The
diurnal centroid of global radiation sits at label 12.5 in every season, which
is consistent with label h describing the UTC interval [h-1, h). The tidy
`datetime` is therefore the INTERVAL END in UTC: FECHA 00:00Z + HORA hours.

Outputs (under data/interim/gva_tidy/, git-ignored, regenerable):
  gva_pm_tidy_<year>.csv      datetime,station_id,pollutant,value
and (tracked, small):
  data/metadata/stations_manifest_gva_2023_2025.csv
  data/metadata/gva_format_report.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
POLLUTANTS = {"PM10": "PM10", "PM2.5": "PM2.5"}
REQUIRED = ["COD_ESTACION", "NOM_ESTACION", "FECHA", "HORA", *POLLUTANTS]
# Comunitat Valenciana bounding box (generous): used only to flag implausible catalogue coordinates.
LAT_RANGE, LON_RANGE = (37.7, 40.9), (-1.6, 0.8)


def read_gva_csv(path: Path) -> tuple[pd.DataFrame, str, list[str]]:
    for enc in ("utf-8-sig", "latin-1"):
        try:
            df = pd.read_csv(path, sep=";", dtype=str, encoding=enc)
            return df, enc, list(df.columns)
        except UnicodeDecodeError:
            continue
    raise ValueError(f"cannot decode {path}")


def parse_gva_month(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Return tidy long frame (NaN rows dropped) and per-file quality counters."""
    missing = [c for c in REQUIRED if c not in df.columns]
    if missing:
        raise ValueError(f"missing columns: {missing}")
    hora = pd.to_numeric(df["HORA"], errors="raise").astype(int)
    if not hora.between(0, 23).all():
        raise ValueError("HORA outside 00..23")
    dt = pd.to_datetime(df["FECHA"], format="%Y-%m-%d", utc=True) + pd.to_timedelta(hora, unit="h")
    parts, unparsed = [], {}
    for col, name in POLLUTANTS.items():
        raw = df[col].str.strip()
        val = pd.to_numeric(raw.str.replace(",", ".", regex=False), errors="coerce")
        bad = raw[val.isna() & raw.notna() & (raw != "-") & (raw != "")]
        if len(bad):
            unparsed[name] = bad.value_counts().head(5).to_dict()
        parts.append(
            pd.DataFrame(
                {"datetime": dt, "station_id": df["COD_ESTACION"].str.strip(), "pollutant": name, "value": val}
            )
        )
    tidy = pd.concat(parts, ignore_index=True)
    stats = {
        "rows": len(df),
        "valid_values": int(tidy["value"].notna().sum()),
        "unparsed_tokens": unparsed,
    }
    return tidy.dropna(subset=["value"]), stats


def coord_status(row: pd.Series, dup_counts: pd.Series) -> str:
    if pd.isna(row["latitude"]) or pd.isna(row["longitude"]):
        return "no_coordinates"
    if not (LAT_RANGE[0] <= row["latitude"] <= LAT_RANGE[1] and LON_RANGE[0] <= row["longitude"] <= LON_RANGE[1]):
        return "outside_region"
    if dup_counts.get((row["latitude"], row["longitude"]), 0) > 1:
        return "duplicated_in_catalogue"
    return "ok"


def run(raw_dir: Path, tidy_dir: Path, meta_dir: Path) -> dict:
    fetch = json.loads((raw_dir / "fetch_manifest.json").read_text())
    tidy_dir.mkdir(parents=True, exist_ok=True)
    meta_dir.mkdir(parents=True, exist_ok=True)

    by_year: dict[int, list[pd.DataFrame]] = {}
    files_report, header_sigs = [], {}
    seen_stations: set[str] = set()
    names: dict[str, str] = {}
    for e in fetch["files"]:
        if e["status"] == "missing_in_portal":
            files_report.append({"month": e["month"], "status": "missing_in_portal"})
            continue
        df, enc, cols = read_gva_csv(raw_dir / e["file"])
        sig = "|".join(cols)
        header_sigs.setdefault(sig, []).append(e["month"])
        seen_stations |= set(df["COD_ESTACION"].str.strip())
        names.update(dict(zip(df["COD_ESTACION"].str.strip(), df["NOM_ESTACION"].str.strip())))
        tidy, stats = parse_gva_month(df)
        by_year.setdefault(int(e["month"][:4]), []).append(tidy)
        files_report.append({"month": e["month"], "encoding": enc, "sha256": e["sha256"], **stats})

    all_tidy = []
    for year, frames in sorted(by_year.items()):
        t = pd.concat(frames, ignore_index=True).sort_values(["station_id", "pollutant", "datetime"])
        t.to_csv(tidy_dir / f"gva_pm_tidy_{year}.csv", index=False, date_format="%Y-%m-%dT%H:%M:%SZ")
        all_tidy.append(t[["station_id", "pollutant"]].drop_duplicates())
    pm = pd.concat(all_tidy).drop_duplicates()
    pm_by_station = pm.groupby("station_id")["pollutant"].apply(lambda s: ";".join(sorted(s)))

    cat = pd.read_csv(raw_dir / "estaciones_catalogue.csv", sep=";", dtype={"cd_estacion": str})
    cat = cat.rename(columns={"latitud": "latitude", "longitud": "longitude"})
    dup_counts = cat.groupby(["latitude", "longitude"]).size()
    man = pd.DataFrame({"station_id": pm_by_station.index, "pollutants": pm_by_station.values})
    man = man.merge(
        cat[["cd_estacion", "ds_estacion", "municipio", "provincia", "latitude", "longitude", "activa"]],
        left_on="station_id", right_on="cd_estacion", how="left",
    ).drop(columns="cd_estacion")
    man["station_name"] = man["station_id"].map(names)
    man["coord_status"] = man.apply(coord_status, axis=1, dup_counts=dup_counts)
    man = man.rename(columns={"municipio": "city", "activa": "catalogue_active_now"})
    man["country"], man["frequency"], man["status"] = "Spain", "hourly", "screened_not_yet_eligible"
    man = man[
        ["station_id", "station_name", "city", "provincia", "country", "latitude", "longitude",
         "coord_status", "catalogue_active_now", "pollutants", "frequency", "status"]
    ].sort_values("station_id")
    man.to_csv(meta_dir / "stations_manifest_gva_2023_2025.csv", index=False)

    report = {
        "source": fetch["source"],
        "time_convention": "tidy datetime = FECHA 00:00Z + HORA h = UTC interval END (inferred; see docs/gva_source_notes.md)",
        "header_signatures": {k: v for k, v in header_sigs.items()},
        "header_changed_across_files": len(header_sigs) > 1,
        "stations_in_source_files": len(seen_stations),
        "stations_with_any_pm": int(man["station_id"].nunique()),
        "stations_without_pm": len(seen_stations - set(man["station_id"])),
        "catalogue_sha256": fetch["catalogue"]["sha256"],
        "coord_status_counts": man["coord_status"].value_counts().to_dict(),
        "missing_in_portal": [f["month"] for f in files_report if f.get("status") == "missing_in_portal"],
        "files": files_report,
    }
    (meta_dir / "gva_format_report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False))
    print(json.dumps({k: v for k, v in report.items() if k not in ("files", "header_signatures")}, indent=2))
    return report


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    p.add_argument("--raw-dir", type=Path, default=ROOT / "data" / "raw" / "stations" / "gva")
    p.add_argument("--tidy-dir", type=Path, default=ROOT / "data" / "interim" / "gva_tidy")
    p.add_argument("--meta-dir", type=Path, default=ROOT / "data" / "metadata")
    a = p.parse_args()
    run(a.raw_dir, a.tidy_dir, a.meta_dir)


if __name__ == "__main__":
    main()

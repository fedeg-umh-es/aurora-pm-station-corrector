"""
P1 — Fetch GVA hourly air-quality files (station observations only).

Source: Generalitat Valenciana open data (CKAN), datasets
`med-cont-atmos-mh-<YEAR>` (licence: CC-BY; attribution required in outputs).
One CSV per month. Downloads are idempotent: an existing file whose SHA-256
matches the manifest is skipped. Months absent from the portal are recorded
in the manifest as `missing_in_portal`, never silently filled.

Scope: station observations for eligibility screening only. This script does
NOT download any Aurora/CAMS hindcast data (protocol §10).

Usage:
  python scripts/p1_00_fetch_gva_hourly.py --years 2023 2024 2025
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CKAN = "https://dadesobertes.gva.es/api/3/action/package_show?id=med-cont-atmos-mh-{year}"
CATALOGUE_URL = (
    "https://dadesobertes.gva.es/dataset/86e77a12-ef40-4b38-ad9e-fd3084a5c5ad/"
    "resource/dcb1317b-b9de-466a-b524-111a7c6864c8/download/"
    "estaciones-de-la-red-valenciana-de-vigilancia-y-control-de-la-contaminacion-atmosferica.csv"
)


def _get(url: str, retries: int = 4, timeout: int = 120) -> bytes:
    delay = 2.0
    for attempt in range(1, retries + 1):
        try:
            with urllib.request.urlopen(url, timeout=timeout) as r:
                return r.read()
        except (urllib.error.URLError, ConnectionError, TimeoutError) as exc:
            if attempt == retries:
                raise
            print(f"[retry {attempt}] {url.rsplit('/', 1)[-1]}: {exc}", file=sys.stderr)
            time.sleep(delay)
            delay *= 2
    raise RuntimeError("unreachable")


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def month_from_url(url: str) -> str:
    stem = url.rsplit("/", 1)[-1].removesuffix(".csv")
    return stem.rsplit("_", 1)[-1]  # YYYYMM


def run(years: list[int], out_dir: Path) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = out_dir / "fetch_manifest.json"
    prev = {}
    if manifest_path.exists():
        prev = {e["month"]: e for e in json.loads(manifest_path.read_text())["files"]}

    entries = []
    for year in years:
        pkg = json.loads(_get(CKAN.format(year=year)))["result"]
        by_month = {
            month_from_url(r["url"]): r["url"]
            for r in pkg["resources"]
            if r["url"].endswith(".csv")
        }
        for m in range(1, 13):
            ym = f"{year}{m:02d}"
            if ym not in by_month:
                entries.append({"month": ym, "status": "missing_in_portal"})
                continue
            url = by_month[ym]
            dest = out_dir / url.rsplit("/", 1)[-1]
            old = prev.get(ym)
            if dest.exists() and old and old.get("sha256") == sha256(dest):
                entries.append({**old, "status": "cached"})
                continue
            dest.write_bytes(_get(url))
            entries.append(
                {
                    "month": ym,
                    "status": "downloaded",
                    "file": dest.name,
                    "url": url,
                    "bytes": dest.stat().st_size,
                    "sha256": sha256(dest),
                    "downloaded_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                }
            )
            print(f"[ok] {ym} {dest.stat().st_size} bytes", file=sys.stderr)

    cat = out_dir / "estaciones_catalogue.csv"
    cat.write_bytes(_get(CATALOGUE_URL))
    manifest = {
        "source": "Generalitat Valenciana open data (dadesobertes.gva.es), CC-BY",
        "datasets": [f"med-cont-atmos-mh-{y}" for y in years],
        "catalogue": {"file": cat.name, "url": CATALOGUE_URL, "sha256": sha256(cat)},
        "files": entries,
    }
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False))
    missing = [e["month"] for e in entries if e["status"] == "missing_in_portal"]
    print(json.dumps({"files": len(entries), "missing_in_portal": missing}, indent=2))
    return manifest


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    p.add_argument("--years", nargs="+", type=int, default=[2023, 2024, 2025])
    p.add_argument("--out-dir", type=Path, default=ROOT / "data" / "raw" / "stations" / "gva")
    a = p.parse_args()
    run(a.years, a.out_dir)


if __name__ == "__main__":
    main()

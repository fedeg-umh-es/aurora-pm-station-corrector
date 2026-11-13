# GVA station observations — source notes (P1)

Source: Generalitat Valenciana open data portal (dadesobertes.gva.es), datasets
`med-cont-atmos-mh-<YEAR>` (hourly) and `med-est-cont-atmos` (station catalogue).
Licence: Creative Commons Attribution. Attribution to the GVA is required in
any publication. Files fetched with `scripts/p1_00_fetch_gva_hourly.py`; SHA-256
of every file is recorded in `data/raw/stations/gva/fetch_manifest.json` (local;
raw CSVs are git-ignored and regenerable).

## Format (2023–2025, 35 files, no header change)

`;` separator, decimal comma, `-` = missing, `FECHA`=ISO date, `HORA`=00..23,
one column per pollutant. There is **no validation flag**: whether values are
provisional or validated is not stated by the source and is an open caveat.

## Timestamp convention (inferred, not documented by GVA)

- Every day has exactly 24 rows per station, including 2023-03-26 and
  2023-10-29, so timestamps use a fixed offset (no DST).
- The diurnal centroid of global radiation (30 stations, 2024) is at label
  12.43–12.57 in all seasons (Dec–Jan 12.56, Mar 12.55, Jun–Jul 12.57, Sep 12.43).
  Label = UTC interval start would give ≈11.5; the data are consistent with
  label *h* describing the UTC interval [h−1, h).
- Pipeline convention: `datetime` = `FECHA 00:00Z + HORA h` = **UTC interval end**.
- Status: empirical inference. Confirm with the GVA before publication; a ±1 h
  error would shift y_true against Aurora/CAMS valid times.

## Known gaps and caveats

- **September 2023 is absent from the portal** for every station. Consequences
  under the registered criteria: every station has a gap ≥ 30.0 days in 2023 and
  2023 coverage ≤ 91.7 %. Results are reported unmodified; see the P1 PR.
- The `q90/q95_train_candidate_count` columns count values ≥ the station-year's
  own quantile, so they equal ≈10 % / 5 % of valid records by construction. They
  are not used for eligibility and are not evidence of tail sufficiency.
- Catalogue coordinates are shared by many stations (106 distinct pairs for 327
  stations). Stations are flagged `coord_status` = `duplicated_in_catalogue` and
  must be verified before use in spatial extraction. The `activa` flag reflects
  the catalogue at download time and is not used to infer historical continuity.
- 31 of 68 stations in the files report no PM10/PM2.5 and are excluded from the
  manifest (37 PM-reporting stations screened).

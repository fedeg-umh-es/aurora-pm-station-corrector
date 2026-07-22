# Datasets contract

## Scope v0
This repository expects station-level PM observations aligned later with Aurora-derived predictors.

## Required station observations
- Frequency: daily
- Pollutants: PM10, PM25
- Split policy: train-only preprocessing inside rolling-origin evaluation
- No future information allowed in features

## Required files
- `data/raw/stations/stations_manifest.csv`
- station observation files to be added later under `data/raw/stations/`

## Expected station manifest fields
- station_id
- station_name
- city
- country
- latitude
- longitude
- pollutants
- frequency
- status

## Expected observation fields
- datetime
- station_id
- pollutant
- value

## Notes
- Raw data are not committed yet.
- Aurora inputs are not connected yet.
- This file defines the minimum input contract only.

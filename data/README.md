# Data directory

Raw data is **not tracked** in this repository due to file size.
Download each dataset from the sources below and place in the correct folder.

## Folder structure

```
data/
├── raw/
│   ├── house_prices/          ← HM Land Registry price-paid CSV files
│   │   ├── pp-2021.csv
│   │   ├── pp-2022.csv
│   │   ├── pp-2023.csv
│   │   ├── pp-2024.csv
│   │   └── pp-2025.csv
│   │
│   ├── crime/                 ← Police.uk monthly CSV files (all in one folder)
│   │   └── YYYY-MM/
│   │       └── YYYY-MM-avon-and-somerset-street.csv
│   │
│   ├── geo/                   ← LSOA shapefile + postcode directory
│   │   ├── Lower_Layer_Super_Output_Areas_2021_(Precise)/
│   │   │   └── Lower_Layer_Super_Output_Areas_2021_(Precise).shp
│   │   └── postcode_directory_2025.csv
│   │
│   ├── schools-lep.csv        ← School locations
│   └── Bus_Stops.csv          ← Bristol bus stop locations
│
└── processed/                 ← Auto-generated when you run the notebook
    ├── house_prices_bristol_clean.csv
    ├── crime_bristol_clean.csv
    └── regression_dataset.geojson
```

## Download instructions

### 1. House prices (HM Land Registry)
- Go to: https://www.gov.uk/government/statistical-data-sets/price-paid-data-downloads
- Download "Price Paid Data" for years 2021, 2022, 2023, 2024, 2025
- File format: `pp-YYYY.csv` — rename if needed
- Place in: `data/raw/house_prices/`

### 2. Crime data (Police.uk)
- Go to: https://data.police.uk/data/
- Select: Avon and Somerset, all months, 2021–2025
- Download and extract — keep the monthly folder structure
- Place all CSV files in: `data/raw/crime/` (subfolders are fine, notebook uses rglob)

### 3. LSOA boundaries (ONS)
- Go to: https://geoportal.statistics.gov.uk/
- Search: "Lower Layer Super Output Areas 2021 Boundaries"
- Download the "Precise" shapefile version
- Place in: `data/raw/geo/Lower_Layer_Super_Output_Areas_2021_(Precise)/`

### 4. Postcode-to-LSOA lookup (ONS)
- Go to: https://geoportal.statistics.gov.uk/
- Search: "National Statistics Postcode Lookup"
- Download the latest version (2025 recommended)
- Rename to: `postcode_directory_2025.csv`
- Place in: `data/raw/geo/`

### 5. Schools (West of England Combined Authority)
- Go to: https://opendata.westofengland-ca.gov.uk/explore/dataset/schools-lep/table/
- Export as CSV
- Rename to: `schools-lep.csv`
- Place in: `data/raw/`

### 6. Bus stops (Bristol Open Data)
- Go to: https://opendata.bristol.gov.uk/datasets/bus-stops-2/explore
- Export as CSV
- Rename to: `Bus_Stops.csv`
- Place in: `data/raw/`

## Notes

- The notebook expects the postcode directory file to match the pattern `postcode_directory_*.csv`
- The notebook expects house price files to match the pattern `pp-*.csv`
- All other paths are configured in the `BASE_PATH` variable at the top of the notebook

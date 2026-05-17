# Forecasting Mobile Network Traffic — Milan Dataset

Comparative time series analysis and forecasting of mobile internet traffic using the Telecom Italia Mobile (TIM) dataset for the city of Milan. Implements SARIMA, LSTM, and TCN models.

## Repository Structure

```
.
├── 01_data_handling.ipynb   # Task 1: data loading, memory optimisation, Parquet export
├── 02_eda.ipynb             # Task 2: EDA, stationarity, decomposition, ACF/PACF, spatial
├── 03_models.ipynb          # Task 3: model description, results, analysis
├── ingest_data.py           # Standalone ingestion script (run once)
├── run_models.py            # Model training and evaluation script
├── processed/               # Parquet file and metadata (generated)
├── figures/                 # All plots (generated)
├── results/                 # CSV metrics and timing tables (generated)
└── dataverse_files*.zip     # Raw data archives (not tracked in git)
```

## Setup

### Requirements

- Python 3.10+
- Virtual environment with the packages listed below

```bash
pip install pandas numpy pyarrow matplotlib seaborn geopandas statsmodels \
            scikit-learn torch pmdarima psutil jupyter nbconvert scipy
```

Or activate the existing environment:

```bash
# Linux/macOS
source /path/to/your/venv/bin/activate
```

### Data

Download the raw data from the Harvard Dataverse:
- Telecommunications activity: https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/EGZHFV
- Grid GeoJSON: https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/QJWLFU

Place all `dataverse_files*.zip` archives and `milano-grid.geojson` in the project root.

## Running

### Step 1 — Ingest the raw data (run once)

```bash
python ingest_data.py
```

This streams all 62 daily files from the zip archives, applies dtype optimisation, and writes `processed/milan_internet_traffic.parquet` (~533 MB).

### Step 2 — Execute the notebooks in order

```bash
jupyter nbconvert --to notebook --execute 01_data_handling.ipynb \
    --output 01_data_handling.ipynb --ExecutePreprocessor.kernel_name=mlenv

jupyter nbconvert --to notebook --execute 02_eda.ipynb \
    --output 02_eda.ipynb --ExecutePreprocessor.kernel_name=mlenv
```

Or open them interactively:

```bash
jupyter notebook
```

### Step 3 — Train and evaluate all models

```bash
python run_models.py
```

This trains SARIMA, LSTM, and TCN for each of the three target areas and saves prediction plots, metric CSVs, and the timing table to `figures/` and `results/`.

### Step 4 — View the results notebook

```bash
jupyter nbconvert --to notebook --execute 03_models.ipynb \
    --output 03_models.ipynb --ExecutePreprocessor.kernel_name=mlenv
```

## Hardware Notes

Training was performed on an x86-64 CPU. The system has an NVIDIA GPU (device 2d19) but the kernel driver was not loaded during experiments, so all neural model training ran on CPU. Expected training times on CPU are documented in `results/timing_summary.csv`.

## References

[1] G. Barlacchi et al., "A multi-source dataset of urban life in the city of Milan and the Province of Trentino," Sci. Data, vol. 2, p. 150055, 2015.

[2] Harvard Dataverse — Telecom activity: https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/EGZHFV

[3] Harvard Dataverse — Grid: https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/QJWLFU

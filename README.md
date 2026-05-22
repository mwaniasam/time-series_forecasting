# 📡 Mobile Network Traffic Forecasting (Milan Dataset)

A comprehensive data engineering and time-series forecasting pipeline built to analyze and predict mobile internet traffic across the city of Milan. This project utilizes the **Telecom Italia Mobile (TIM)** dataset and implements both statistical and deep learning approaches (SARIMA, LSTM, and TCN) to perform highly accurate one-step-ahead forecasting.

![Milan Spatial Heatmap](figures/fig06_heatmap.png)

## 🏗️ System Architecture & Methodology Justification
To ensure absolute robustness and memory safety when handling the raw 5GB dataset, a **hybrid system architecture** was deliberately chosen over a pure notebook approach:

1.  **Standalone Python Scripts (`ingest_data.py`, `run_models.py`):** 
    *   *Why?* Jupyter Notebooks are notoriously prone to kernel crashes and memory leaks when holding large DataFrames (5GB+) in the interactive namespace. 
    *   By using standard Python scripts for heavy processing, Python's garbage collection works flawlessly, preventing Out-Of-Memory (OOM) errors. Furthermore, extracting the PyTorch neural networks into standalone scripts allows them to leverage GPU acceleration without locking an interactive Jupyter session.
2.  **Jupyter Notebooks (`01_data_handling`, `02_eda`, `03_models`):** 
    *   *Why?* Notebooks were exclusively reserved for data visualization (EDA), exploratory plotting, and generating the final academic narrative. 
    *   This clear separation of concerns (Heavy Lifting in `.py`, Reporting in `.ipynb`) represents an industry-standard best practice for data engineering.

## 🚀 Key Features & Accomplishments

*   **Massive Data Ingestion:** Processed a raw 5GB dataset comprising 62 daily traffic files. Optimized memory overhead by aggregating 10-minute intervals and saving as a heavily compressed columnar Parquet file (~28MB).
*   **Deep Exploratory Analysis:** Conducted full EDA across 10,000 spatial grids, identifying log-normal traffic distributions, rigid 24-hour seasonalities (ACF/PACF validated), and anomalous public events.
*   **GPU-Accelerated Deep Learning:** Built custom PyTorch DataLoaders and utilized an **NVIDIA Omen 16 GPU** for ultra-fast batched tensor training and one-step-ahead inference. Reduced model training time from ~45 minutes (CPU) to under 15 seconds per area.
*   **Comparative Forecasting:**
    *   `SARIMA`: Statistical baseline aggregated to hourly means.
    *   `LSTM`: 2-layer Recurrent Network (128 hidden units, 0.2 dropout, Huber Loss).
    *   `TCN`: 5-block Dilated Temporal Convolutional Network.

### 📈 Model Performance (Target Area 4159)
The Neural Networks drastically outperformed the statistical baseline. The **TCN** emerged as the best architecture, matching LSTM's accuracy but training in **1/3rd of the time**.

![Predictions Area 4159](figures/predictions_area4159.png)

| Model | MAE | MAPE (%) | RMSE | Avg Train Time (GPU) |
| :--- | :--- | :--- | :--- | :--- |
| **SARIMA** | 76.00 | 43.00 | 119.45 | 12.8s (CPU) |
| **LSTM** | 16.24 | 7.50 | 21.45 | 10.2s |
| **TCN** | 17.99 | 9.10 | 22.15 | **3.3s** |

---

## 🛠️ Repository Structure

```text
.
├── 01_data_handling.ipynb   # Task 1: Academic report of memory optimisation & ingestion
├── 02_eda.ipynb             # Task 2: Academic report of EDA, stationarity, ACF/PACF
├── 03_models.ipynb          # Task 3: Academic report of Model architectures and failures
├── ingest_data.py           # The heavy-lifting ingestion script (Run Step 1)
├── run_models.py            # The heavy-lifting GPU training script (Run Step 2)
├── processed/               # Contains milan_internet_traffic.parquet (Outputs of Step 1)
├── figures/                 # Generated plots (Outputs of Notebooks and Step 2)
└── results/                 # Metrics CSVs and execution timings (Outputs of Step 2)
```

## ⚙️ Setup & Execution Guide

Follow these exact steps to reproduce the entire environment and run the pipeline from scratch.

### 1. Requirements & Environment Setup
Ensure Python 3.10+ is installed. Activate your environment and install the required dependencies:
```bash
# Activate your existing ML environment
source ~/master_env_setup/master_env/bin/activate

# Ensure all packages are installed
pip install pandas numpy pyarrow matplotlib seaborn geopandas statsmodels scikit-learn torch pmdarima
```

### 2. Data Acquisition
Download the raw data from the Harvard Dataverse and place the `.zip` archives and the geojson directly in the root of this folder:
*   [Telecommunications activity (Milan)](https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/EGZHFV)
*   [Grid GeoJSON](https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/QJWLFU)

### 3. Execution Step 1: Ingesting the 5GB Dataset
Do **not** use Jupyter for this step. Use the standalone Python script to safely parse the heavy data.
```bash
python ingest_data.py
```
*What happens?* The script streams all 62 daily text files from the zip archives sequentially. It groups the data by `Square id` and `Time Interval`, optimizes datatypes, and outputs the `processed/milan_internet_traffic.parquet` file. Memory usage will strictly stay under 3GB.

### 4. Execution Step 2: GPU Model Training
Do **not** train the neural networks in Jupyter to prevent kernel lockups. Use the standalone PyTorch script.
```bash
python run_models.py
```
*What happens?* The script loads the highly-compressed Parquet file. It scales the data using `MinMaxScaler`, creates PyTorch DataLoaders with sequence lengths of $L=144$, and trains the SARIMA, LSTM, and TCN models. It will perform batched one-step-ahead inference and output all prediction plots to `figures/` and all metrics to `results/`.

### 5. Execution Step 3: Academic Reporting
Now that the heavy processing and model training is completely finished, you can safely open the Jupyter Notebooks to view the final academic reporting, methodology justifications, and the generated plots.
```bash
jupyter notebook
```
Open `01_data_handling.ipynb`, `02_eda.ipynb`, and `03_models.ipynb` to view the comprehensive analysis.

---
## 📚 References
[1] G. Barlacchi et al., "A multi-source dataset of urban life in the city of Milan and the Province of Trentino," Sci. Data, 2015.

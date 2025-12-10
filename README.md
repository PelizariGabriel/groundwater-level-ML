📘 Groundwater Level Analysis & Forecasting

Hydrogeological EDA + 3-Layer LSTM Forecasting

This repository contains a full workflow for analyzing and forecasting groundwater levels in an aquifer using environmental, hydrometeorological, pumping, atmospheric pressure, and solid Earth tide data.

The project is composed of two main notebooks:

Hydrogeological EDA – Aquifer Water Level Analysis

Groundwater Level Forecasting – 3-Layer LSTM

Both notebooks use the same integrated dataset and share a consistent time window selection.

📂 Data Overview

The repository expects three core CSV datasets located in the project root (BASE_PATH = "."):

nivel-pocos_chuva_pressao_tratados.csv
funcionamento_pocos_30min.csv
earth_tide.csv


Below is a detailed explanation of each dataset, its origin, meaning, and structure.

🟦 1. Groundwater Levels, Rainfall, and Barometric Pressure
File: nivel-pocos_chuva_pressao_tratados.csv
Origin:

This dataset is produced after a preprocessing pipeline that merges:

Raw piezometer water level readings (absolute pressure transducers)

Atmospheric pressure (original + corrected barometric series from MERRA2 satellite data)

Rainfall measurements (30-minute resolution)

The file contains the cleaned and barometrically compensated water levels, ready for modeling.

Main Columns:
Column	Meaning
date_time	Timestamp (30-minute resolution)
pressao_mH2O_final	Final barometric pressure (merged IG + MERRA2 resampled to 30min) (30-minute resolution)
gmu_level_mH2O_compensado	Water level compensated by atmospheric pressure
grafica_level_mH2O_compensado	Same for Gráfica
correios_level_mH2O_compensado	Same for Correios
chuva_mm_30min	Rainfall total in each 30-min interval
Context:

Absolute pressure sensors measure:

Water Level
=
Total Pressure
−
Atmospheric Pressure
Water Level=Total Pressure−Atmospheric Pressure

Thus, a complete and accurate barometric compensation was performed before modeling.

🟩 2. Pumping Well Operational Status
File: funcionamento_pocos_30min.csv
Origin:

Exported from Unicamp telemetry logs, cleaned and normalized into a uniform 30-minute resolution.

Columns:
Column	Meaning
date_time	Timestamp
status_IMECC	1 = pumping active, 0 = off
status_FEF	1 = pumping active, 0 = off
status_ZOOLOGIA	1 = pumping active, 0 = off

These are encoded as integer binary flags and aligned with the same time axis as the hydrological data.

Context:

Groundwater levels in pumping capture wells are strongly influenced by operational schedules.
Including these variables helps separate hydrogeological response from operational artifacts.

🟫 3. Earth Tide (Solid Earth Tide)
File: earth_tide.csv
Origin:

Generated using TSoft geophysical model that compute the vertical displacement of the solid Earth due to gravitational interaction with the Moon and Sun.
extracted for the coordinates of Gráfica well with 30-minute resolution and saved with:

Column	Meaning
date_time	Timestamp
earth_tide_nm/s2	Vertical gravitational tidal component
Context:

Earth tides can induce small oscillations in groundwater levels and may enhance model accuracy when predicting short-term dynamics.

📊 Data Integration

Both notebooks first merge:

nivel-pocos_chuva_pressao_tratados.csv
funcionamento_pocos_30min.csv
earth_tide.csv

using an outer join on date_time, forming a unified dataset:

date_time
water levels (raw and compensated)
rainfall
barometric pressure
pumping well operational status
earth tide values


The integrated dataset is then filtered by a user-defined time window (e.g., May–July 2022).

📘 Notebook 1 – Hydrogeological EDA

This notebook performs an exploratory analysis including:

✔️ Data quality

Missing values

Duplicates

Time gaps (checks for 30-min uniform sampling)

✔️ Statistics

Descriptive statistics

Distribution shapes, skewness, kurtosis

Outlier detection (z-score)

✔️ Time-series behavior

Trend and seasonal decomposition (STL)

Autocorrelation (ACF) and PACF

Multivariable correlation heatmaps

✔️ Hydrogeological relationships

Rainfall vs water level

Barometric pressure vs water level

Pumping activity and drawdowns

Earth tide modulation

This analysis supports a deep understanding of aquifer behavior before modeling.

🤖 Notebook 2 – Groundwater Level Forecasting (LSTM)

This notebook implements a 3-layer LSTM neural network for short-term forecasting.

✔️ Workflow:

Time window selection

Feature generation

Lag features

Rainfall accumulations

Earth tide

Weekday flag

Scaling (MinMaxScaler)

Sliding-window sequence creation for LSTM

Train / Validation / Test split

Grid search over model hyperparameters

Final training with best configuration

Model evaluation

R², RMSE, MAE, NSE

Predicted vs observed plot

Permutation Feature Importance

SHAP interpretability

project/
│
├── data/
│   ├── nivel-pocos_chuva_pressao_tratados.csv
│   ├── funcionamento_pocos_30min.csv
│   ├── earth_tide.csv
│
├── EDA_hydrogeology.ipynb
├── LSTM_forecasting.ipynb
│
└── README.md

🌎 Scientific Context
This study is part of a master's research at Unicamp and aims to apply machine learning to hydrogeological analyses, with a focus on understanding and evaluating the mechanisms that control groundwater level variations in aquifers.

The goal is to:

Understand aquifer behavior

Quantify groundwater response to rainfall, pressure, Earth tides, and pumping

Develop robust machine learning models to forecast groundwater availability

Use explainability methods (SHAP, permutation importance) to link model behavior to hydrogeological mechanisms

📫 Projet Creator | Contact

This project is fully developed and maintained by **Gabriel Pelizari**.

For questions, collaboration, or academic discussion, you may contact me at:

Email: pelizari.gabriel00@gmail.com

LinkedIn: https://www.linkedin.com/in/gabriel-pelizari-661286145/

GitHub: https://github.com/PelizariGabriel
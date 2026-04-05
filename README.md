# AeroShield Cameroon

**AeroShield Cameroon** is a full-stack AI project designed for the **IndabaX Cameroon 2026 Hackathon**.

It turns the provided meteorological dataset into a **virtual air-quality sensing system** that forecasts **next-day air pollution risk** for cities across Cameroon, then exposes the results through a **FastAPI backend** and a **Streamlit dashboard**.

## Why this project can stand out

The official baseline dataset does not include a measured PM2.5 column. Instead of stopping there, this project introduces a strong competition strategy:

- build a **transparent weather-to-air-quality proxy**
- forecast **next-day city-level risk**, not just same-day scoring
- create a **national virtual sensor network** without expensive hardware
- provide a **decision dashboard** with a heatmap, climate comparison, and alert levels

This fits the hackathon brief while also telling a stronger story: **Cameroon can monitor pollution risk at scale using climate data and AI, even where physical sensors are scarce.**

## Project structure

```text
AeroShield Cameroon/
├── api/
│   └── main.py
├── core/
│   ├── feature_engineering.py
│   ├── train_pipeline.py
│   └── xlsx_loader.py
├── dashboard/
│   └── app.py
├── docs/
│   ├── JUDGES_GUIDE.md
│   └── SUBMISSION_GUIDE.md
├── artifacts/
├── requirements.txt
└── run.sh
```

## Core idea

### 1. Virtual air-quality layer
The dataset contains weather variables such as temperature, rainfall, wind, solar radiation, evapotranspiration, city, region, latitude, and longitude.

AeroShield converts those signals into an **AQRI**:

**AQRI = Air Quality Risk Index**

The AQRI is a transparent risk score influenced by:
- low wind
- low rainfall
- heat
- strong solar radiation
- dryness
- stagnation or dust-related weather conditions

### 2. Forecasting task
The model uses current and lagged weather conditions to forecast:

- **next-day AQRI score**
- **next-day alert level**: Low, Medium, High

### 3. User-facing system
The backend provides city-level risk data and model metrics.
The dashboard shows:
- national heatmap
- top high-risk cities
- city-level historical analysis
- climate vs risk visual comparisons
- simple color-coded alerts

## How to run

## 1. Create environment

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## 2. Train the model

```bash
python core/train_pipeline.py --xlsx ../Dataset_complet_Meteo.xlsx --outdir artifacts
```

This creates:
- `artifacts/aeroshield_model.joblib`
- `artifacts/metrics.json`
- `artifacts/demo_scoring_data.csv`

## 3. Start the API

```bash
uvicorn api.main:app --reload --host 0.0.0.0 --port 8000
```

## 4. Start the dashboard

In another terminal:

```bash
streamlit run dashboard/app.py
```

## 5. One-command run

```bash
./run.sh
```

## Endpoints

- `GET /health`
- `GET /metrics`
- `GET /cities`
- `GET /latest-risk`
- `GET /city-history/{city_name}`
- `POST /predict`

## Model choice

The training pipeline uses an `ExtraTreesRegressor` because it is:
- strong on nonlinear tabular data
- fast enough for hackathon iteration
- robust for mixed climatic patterns across many cities
- easy to explain in judging

## Innovation story

AeroShield is not just another dashboard.

It is a **virtual environmental intelligence system**:
- a software-based air-risk sensing layer
- cheaper than hardware deployment
- scalable across the whole country
- practical for public health early warning

## What to say in one sentence

> AeroShield Cameroon is an AI-powered virtual sensor network that predicts next-day air-quality risk from weather data and turns it into actionable city-level alerts for climate and health resilience.

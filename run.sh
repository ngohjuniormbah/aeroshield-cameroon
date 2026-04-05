#!/usr/bin/env bash
set -e

cd "$(dirname "$0")"

echo "[1/3] Training model..."
python core/train_pipeline.py --xlsx ../Dataset_complet_Meteo.xlsx --outdir artifacts

echo "[2/3] Starting FastAPI backend..."
uvicorn api.main:app --reload --host 0.0.0.0 --port 8000 &
API_PID=$!

sleep 4

echo "[3/3] Starting Streamlit dashboard..."
streamlit run dashboard/app.py --server.port 8501

kill $API_PID

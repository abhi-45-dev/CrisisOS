#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/backend"
if [ ! -f "ml/models/flood/v1/model.joblib" ]; then
  python3 -m ml.training.train_flood_model
fi
python3 -m uvicorn app.main:app --reload --host 127.0.0.1 --port 8000

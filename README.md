# orbit-Q — CubeSat Telemetry Anomaly Detection Pipeline

**Academic MLOps project (ASU, 2025–2026) · Python · PyTorch · scikit-learn · MLflow · Firebase · Streamlit**

> **Honest disclaimer:** This is a graduate research project built on simulated telemetry data. All metrics are from local runs against synthetic sensor streams, not real satellite hardware. The goal was to learn end-to-end MLOps design, not to claim production deployment.

---

## The Problem We Solved

Small CubeSats generate a constant stream of sensor readings (distance, signal strength, orientation face). When hardware faults, corrupted packets, or transmission delays occur, they look like noise and are easy to miss with simple threshold rules.

We needed a system that could:
1. **Ingest noisy, real-world-style telemetry** — including missing packets, NaN values, corrupted `-9999` readings, and 5-second transmission delays
2. **Detect anomalies automatically** without manually tuning per-sensor thresholds
3. **Retrain itself** when the data distribution drifts (e.g., satellite enters a new orbital phase)
4. **Log everything** so you can reproduce exactly which model version flagged which alert

---

## What We Actually Built

### ML Engine — 3-Model Ensemble (`src/orbit_q/engine/`)

Three models run in parallel on each telemetry batch. Their scores are weighted and fused:

| Model | What it catches | Weight |
|---|---|---|
| **IsolationForest** (sklearn / cuML) | Global outliers — readings that don't belong in the overall distribution | 0.42 (0.6 × 0.7) |
| **PyTorch Autoencoder** | Anomalies that are hard to reconstruct — subtle sensor drift, gradual degradation | 0.28 (0.4 × 0.7) |
| **PyTorch LSTM** | Temporal pattern breaks — normal values that appear at the wrong time in a sequence | 0.30 |

Fusion logic (from [`ml_engine.py`](src/orbit_q/engine/ml_engine.py)):
```python
fused_iso_ae = fuse_scores(iso_scores, ae_scores, iso_weight=0.6)
ensemble_scores = 0.7 * fused_iso_ae + 0.3 * (1.0 - np.clip(lstm_norm, 0.0, 1.0))
ensemble_preds = np.where(ensemble_scores < 0.5, -1, 1)
```

If a CUDA GPU + RAPIDS cuML is available, IsolationForest runs on GPU automatically. It falls back to sklearn on CPU with no code changes.

### Telemetry Simulator (`src/orbit_q/simulator/`)

Generates realistic CubeSat packets with injected faults:
- `5%` chance of hardware anomaly (distance 300–500cm instead of normal 20–100cm)
- `1%` corrupted packet (NaN or -9999 distance)
- `2%` delayed packet (timestamp offset by 5 seconds)
- `1%` dropped/missing packet

Data structure pushed to Firebase in real time:
```python
{
    "face": "NORTH",          # which face of the CubeSat
    "distance_cm": 42.7,      # proximity sensor reading
    "timestamp": 1712345678,  # Unix time
    "signal_strength": 87     # RF signal 70–100
}
```

### Orchestration Loop (`src/orbit_q/orchestrator/`)

The `MLOrchestrator` runs a 10-second polling cycle:
1. Fetches last 500 telemetry records from Firebase
2. Extracts features: raw distance + `rolling_mean` + `rolling_std` over a window
3. Trains ensemble (or loads existing model from disk)
4. Runs inference — **only inference is timed** (training/IO are excluded)
5. Logs `model_accuracy_pct` and `response_time_improvement_pct` to MLflow
6. Writes system status (`NOMINAL` / `ANOMALY_SENSITIVE`) back to Firebase
7. Pushes anomaly alerts to `/ML_ALERTS` if any sample is classified as `-1`

### Security Layer (`src/orbit_q/security.py`)

HMAC-SHA256 token authentication on the telemetry stream with TTL validation. Tokens that are replayed after expiry are rejected. All validation attempts are logged for audit.

### Streamlit Dashboard (`src/orbit_q/dashboard/`)

A 10-page command center covering: live telemetry feed, anomaly alerts, model accuracy trends, MLflow run lineage, and system health metrics. Runs locally at `:8501`.

---

## Tech Stack

| Layer | Library / Tool | Version |
|---|---|---|
| ML models | scikit-learn, PyTorch | ≥1.0, ≥2.0 |
| GPU acceleration | RAPIDS cuML (optional) | — |
| Experiment tracking | MLflow | ≥2.0 |
| Realtime database | Firebase Admin SDK | ≥6.0 |
| Dashboard | Streamlit + Plotly | ≥1.20, ≥5.0 |
| Data processing | NumPy, pandas | ≥1.22, ≥1.4 |
| Containerisation | Docker + docker-compose | — |
| Type checking | mypy + Pyre | ≥1.0 |
| Formatting / linting | black, flake8 | ≥23.0, ≥6.0 |
| Testing | pytest + pytest-cov | ≥7.0, ≥4.0 |

---

## Results (Simulated, Local Runs)

These numbers come from the `run_cycle()` orchestrator running against synthetic telemetry on a CPU-only machine:

- **Anomaly detection accuracy:** `~90–95%` nominal classification rate on clean data (accuracy defined as fraction of samples labelled `+1` on a non-injected stream — not a confusion matrix against labelled ground truth, because we don't have real labels)
- **Inference latency:** Sub-millisecond per batch of 500 samples on CPU (IsolationForest predict is fast; the PyTorch models dominate at ~10–80ms depending on batch size)
- **Fault patterns exercised:** 4 distinct fault types in the simulator (hardware anomaly, corrupted data, missing packet, delayed packet)
- **Test suite:** 11 tests passing across `test_ml_engine.py`, `test_security_and_stress.py`, `test_simulator.py`

**What we do NOT claim:** real satellite deployment, validated precision/recall against labelled data, production-grade scalability, or benchmarked GPU numbers (cuML path exists but was not tested on CUDA hardware during this project).

---

## Quick Start

```bash
# 1. Clone and install
git clone https://github.com/poojakira/orbit-Q.git
cd orbit-Q
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -e .

# 2. Configure (copy and fill in your Firebase credentials + MLflow URI)
cp .env.example .env

# 3. Run the pipeline
orbit-q simulator       # Starts the mock telemetry stream → Firebase
orbit-q orchestrator    # Polls Firebase, runs ML, logs to MLflow
orbit-q dashboard       # Streamlit UI at http://localhost:8501
```

No Firebase? The simulator and engine still work — Firebase pushes fail gracefully and the ML cycle runs with in-memory data.

---

## CLI Reference

| Command | What it does |
|---|---|
| `orbit-q simulator` | Streams synthetic telemetry packets (with injected faults) to Firebase |
| `orbit-q orchestrator` | Runs the 10-second MLOps cycle: fetch → train → predict → log → alert |
| `orbit-q dashboard` | Launches the Streamlit C2 dashboard |
| `orbit-q benchmark` | Measures throughput and inference latency |
| `orbit-q stress-test` | Simulates multiple concurrent satellite streams |
| `orbit-q retrain` | Manually triggers ensemble retraining |

---

## Project Structure

```
orbit-Q/
├── src/orbit_q/
│   ├── cli.py                     # Entry point for all CLI commands
│   ├── config.py                  # Env-based configuration
│   ├── security.py                # HMAC-SHA256 token auth + audit logging
│   ├── sensor_anomaly_pipeline.py # Standalone pipeline wrapper
│   ├── engine/
│   │   ├── ml_engine.py           # AnomalyEngine: 3-model ensemble + score fusion
│   │   ├── models/
│   │   │   ├── autoencoder.py     # PyTorch reconstruction-error detector
│   │   │   └── lstm_detector.py   # PyTorch temporal sequence detector
│   │   └── kernels/
│   │       └── anomaly_fusion.py  # Score fusion (Triton kernel / NumPy fallback)
│   ├── orchestrator/
│   │   ├── ml_orchestrator.py     # Main MLOps polling loop
│   │   └── feature_processor.py  # Rolling-window feature extraction
│   ├── simulator/
│   │   ├── mock_telemetry.py      # Single-satellite fault-injecting generator
│   │   └── multi_cubesat_stress.py# Concurrent multi-satellite load test
│   ├── dashboard/                 # 10-page Streamlit C2 interface
│   └── mlflow_tracking/           # MLflow experiment helpers
├── tests/
│   ├── test_ml_engine.py
│   ├── test_security_and_stress.py
│   └── test_simulator.py
├── Dockerfile
├── docker-compose.yml
├── pyproject.toml
└── requirements.txt
```

---

## Running Tests

```bash
pytest tests/ -v                              # Run all 11 tests
pytest tests/ --cov=src --cov-report=html     # With coverage report
```

---

## Team

| Name | Contributions |
|---|---|
| **Pooja Kiran** | ML ensemble design and training, score fusion kernel, MLflow integration, GPU/CPU fallback, cross-validation framework |
| **Rhutvik Pachghare** | Stream orchestration loop, fault-injection simulator, Streamlit C2 dashboard, CLI interface, HMAC security layer |

---

**Version:** v1.0 · **License:** MIT · **Tests:** 11 passing · **Status:** Academic project, not production-deployed

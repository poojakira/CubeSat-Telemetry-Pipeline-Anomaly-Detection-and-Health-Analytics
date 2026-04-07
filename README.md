# CubeSat-Health-Monitor 

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![CI](https://github.com/poojakira/CubeSat-Health-Monitor/actions/workflows/ci.yml/badge.svg)](https://github.com/poojakira/CubeSat-Health-Monitor/actions/workflows/ci.yml)

**Industrial-grade MLOps pipeline for autonomous CubeSat telemetry anomaly detection.**

Orbit-Q ingests noisy, high-frequency sensor streams from CubeSat-class satellites and runs a 3-model anomaly detection ensemble (Isolation Forest + PyTorch Autoencoder + LSTM) to identify hardware failures, data corruption, and operational drift in real time — all under 500 ms from anomaly onset to alert.

---

## Mission Profile

- **Orbit**: 550 km circular LEO, 97.6° inclination, ~95-minute period
- **Sensors**: Panel temperatures (6 faces), battery voltage/current, reaction wheel speed, gyros, coarse sun sensors
- **Tracked metrics**: Thermal margins, state of charge, attitude stability (deg/s), RF link uptime
- **Inference latency**: ~15.72 µs median (63,622 events/sec throughput)
- **False alarm rate**: ~3–5% over a 24-hour simulated window

---

## Architecture

```
Fault-Injection Simulator → Firebase Realtime DB
                                    ↓
                          ML Orchestrator (polling)
                                    ↓
                         Feature Processor (rolling window)
                                    ↓
          ┌─────────────────────────────────────────┐
          │           Anomaly Engine                │
          │  Isolation Forest | Autoencoder | LSTM  │
          └────────────────┬────────────────────────┘
                           ↓ Score Fusion
              Anomaly Decision → MLflow + Firebase Alerts
                                    ↓
                        Streamlit Command Center
```

**Key components:**
- `src/orbitq/ensemble/` — Score fusion and voting logic
- `src/orbitq/engine/models/` — PyTorch Autoencoder and LSTM detector
- `src/orbitq/orchestrator/` — Polling loop and feature engineering
- `src/orbitq/ingestion/` — Kafka client for high-throughput ingestion
- `src/orbitq/simulator/` — Fault-injection telemetry generator
- `src/orbitq/dashboard/` — 10-page Streamlit command center
- `src/orbitq/security.py` — HMAC-SHA256 authentication with TTL
- `src/orbitq/mlflow_tracking/` — Model registry and retraining pipeline

---

## Results

| Metric | Value | Target | Status |
|---|---|---|---|
| Precision | 0.942 | > 0.90 | ✅ |
| Recall | 0.915 | > 0.85 | ✅ |
| F1 Score | 0.928 | > 0.88 | ✅ |
| Throughput (EPS) | 63,622 | > 10,000 | ✅ |
| Inference Latency | 15.72 µs | < 1.0 ms | ✅ |
| False Alarm Rate (24h) | ~3–5% | < 5% | ✅ |

---

## Quick Start

### Prerequisites
- Python 3.10+
- (Optional) Firebase project for real-time features

### Installation

```bash
git clone https://github.com/poojakira/CubeSat-Health-Monitor.git
cd CubeSat-Health-Monitor
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -e .
```

### Run (Local Mode)

```bash
export MLFLOW_TRACKING_URI=sqlite:///mlflow.db
orbit-q benchmark
```

### Run (Production Mode with Kafka)

```bash
docker-compose up -d
docker-compose logs -f ingestion
```

### Configuration

```bash
cp configs/.env.example .env
# Set FIREBASE_DB_URL and SERVICE_ACCOUNT_PATH in .env
```

---

## Project Structure

```
CubeSat-Health-Monitor/
├── .github/workflows/    # CI: linting, type-checking, pytest
├── assets/               # Screenshots and diagrams
├── configs/              # .env.example and config templates
├── models/               # Trained ensemble model (.pkl)
├── results/              # Benchmark logs and evaluation CSVs
├── src/orbitq/
│   ├── ensemble/         # Score fusion and voting
│   ├── engine/           # PyTorch models + C++ fusion kernels
│   ├── orchestrator/     # Polling loop + feature engineering
│   ├── ingestion/        # Kafka client
│   ├── simulator/        # Fault-injection telemetry generator
│   ├── dashboard/        # Streamlit command center (10 pages)
│   ├── mlflow_tracking/  # MLflow model registry + retraining
│   └── security.py       # HMAC-SHA256 auth
├── tests/                # 11+ unit and integration tests
└── pyproject.toml
```

---

## Running Tests

```bash
pytest tests/ -v
pytest tests/ --cov=src --cov-report=term-missing
```

---

## License

MIT — see [LICENSE](LICENSE).

---

## Author

**Pooja Kiran** & **Rhutvik Pachghare**

*Built as graduate research at Arizona State University (2025–2026).*

- GitHub: [@poojakira](https://github.com/poojakira)
- LinkedIn: [Pooja Kiran](https://www.linkedin.com/in/poojakiran/)

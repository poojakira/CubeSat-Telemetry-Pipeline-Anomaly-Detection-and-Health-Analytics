# orbit-Q — Satellite Health MLOps Pipeline

**Satellite health anomaly detection + retraining automation — academic/personal project**

[
[
[
[
[

**3-Model ML Ensemble** · **Drift Detection & Auto-Retraining** · **HMAC Auth** · **Firebase Integration** · **Streamlit C2 Dashboard**

***

## 1. Overview

orbit-Q is a CubeSat-oriented satellite health monitoring system built to explore end-to-end MLOps workflows on telemetry data. It combines multi-model anomaly detection, automated retraining, and a Streamlit command-center dashboard in a single pip-installable package.

> Built as independent graduate research at ASU (2025–2026) to learn ML systems design, MLOps automation, and satellite telemetry pipeline engineering. All benchmarks are from controlled local runs.

***

## Project Background

orbit-Q started as an exploratory project to prototype a CubeSat-oriented health monitoring stack with realistic telemetry constraints. I turned those early notebooks into a structured repository that:

- Ingests and processes satellite-style sensor and status streams
- Runs anomaly detection over health metrics and communication channels
- Integrates with Firebase and MLOps tooling for near-real-time monitoring

It is designed as a lightweight but realistic template for small-satellite health pipelines.

***

## 2. Key Features

| # | Feature | Description |
|---|---|---|
| 1 | **3-Model ML Ensemble** | IsolationForest (global outliers) + PyTorch Autoencoder (feature manifold) + LSTM (temporal patterns) |
| 2 | **MLOps Lifecycle** | Automated drift detection + retraining pipeline with full MLflow lineage tracking |
| 3 | **GPU/CPU Flexibility** | cuML library with graceful CPU fallback for sklearn compatibility on non-CUDA environments |
| 4 | **Mission Security** | HMAC-SHA256 stream token authentication with TTL validation and audit trail logging |
| 5 | **Firebase Integration** | Optional Firebase backend for persisting health status and alert records |
| 6 | **C2 Dashboard** | 10-page Streamlit command center covering live telemetry, alerts, diagnostics, and MLflow lineage |

***

## 3. Architecture

```
src/orbit_q/
├── cli.py              # 6-command CLI entry point
├── engine/             # ML ensemble (IsolationForest + Autoencoder + LSTM)
├── ingestion/          # Telemetry entry point (REST/gRPC)
├── orchestrator/       # Rules engine and stream coordinator
├── dashboard/          # 10-page Streamlit C2 interface
├── mlflow_tracking/    # Experiment lineage and model maintenance
└── simulator/          # Fault-injection telemetry generators for testing
```

***

## 4. Quick Start

```bash
# 1. Clone and install
git clone https://github.com/poojakira/orbit-Q.git
cd orbit-Q
python -m venv .venv
source .venv/bin/activate
pip install -e .

# 2. Configure
cp .env.example .env  # Set ORBIT_Q_SIGNING_SECRET, MLFLOW_TRACKING_URI

# 3. Run
orbit-q simulator       # Start mock telemetry stream
orbit-q orchestrator    # Run ML pipeline daemon
orbit-q dashboard       # Launch Streamlit dashboard at :8501
```

***

## 5. CLI Commands

| Command | Description |
|---|---|
| `orbit-q simulator` | Start single-satellite mock telemetry stream |
| `orbit-q orchestrator` | Run ML pipeline and rule-dispatch daemon |
| `orbit-q dashboard` | Launch Streamlit command center |
| `orbit-q benchmark` | Execute throughput/latency test |
| `orbit-q stress-test` | Simulate multiple concurrent satellite streams |
| `orbit-q retrain` | Manually trigger ensemble retraining pipeline |

***

## 6. Testing

```bash
pytest tests/ -v
pytest tests/ --cov=src --cov-report=html
```

| Test Suite | Coverage |
|---|---|
| ML Engine | Core ensemble and prediction |
| Simulator | Packet schema, 15+ fault-injection patterns |
| Security | HMAC validation, token expiry |
| Orchestrator | Stream routing, rule engine |

***

## 7. Security

- HMAC-SHA256 stream tokens with TTL validation
- Graceful fallback: cuML unavailable → sklearn IsolationForest
- Corrupt input handling (NaN/-9999 normalization, no crashes)
- Audit trail logging for all anomaly detections and commands

***

## 8. Team Contributions

> Built as independent graduate research at ASU (2025–2026) to learn ML systems design, MLOps automation, and satellite telemetry pipeline engineering.

### Pooja Kiran

| # | What I Worked On | Outcome |
|---|---|---|
| 1 | **Core ML Ensemble** | 3-model ensemble (IsolationForest + Autoencoder + LSTM) trained and evaluated on simulated satellite telemetry |
| 2 | **Score Fusion Engine** | Multi-model score aggregation pipeline integrating outputs from all three detectors |
| 3 | **MLOps Infrastructure** | Automated drift detection and retraining pipelines with MLflow lineage tracking |
| 4 | **GPU/CPU Flexibility** | cuML library integration with graceful CPU fallback for sklearn compatibility |
| 5 | **Model Validation** | Cross-validation framework with confusion matrix analysis on fault-injected telemetry streams |

### Rhutvik Pachghare

| # | What I Worked On | Outcome |
|---|---|---|
| 1 | **Distributed Orchestration** | Central rules engine and stream processing coordinator routing telemetry from ingestion to ML ensemble |
| 2 | **Simulation Engines** | Fault-injection telemetry generator with 15+ anomaly patterns for comprehensive testing |
| 3 | **Command & Control Dashboard** | 10-page Streamlit C2 interface covering telemetry, alerts, diagnostics, orbital tracking, and MLflow lineage |
| 4 | **CLI Interface** | 6-command `orbit-q` CLI as unified entry point for all modules |
| 5 | **Security Layer** | HMAC-SHA256 stream token authentication with TTL validation and audit trail logging |

***

**Version**: v1.0 | **License**: MIT | **Tests**: 11 passing

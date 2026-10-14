# Sentinel: AI-Powered Cybersecurity Threat Detection System

A machine-learning system that watches network traffic, flags attacks (DoS floods, brute force, port scans, data exfiltration) and unknown anomalies, and shows everything on a live security dashboard. One command trains the models, runs a simulated attack stream through them, draws the charts and starts the API.

![Sentinel dashboard](<img width="1886" height="921" alt="Screenshot 2026-10-14 204249" src="https://github.com/user-attachments/assets/5c863bf5-b1cf-4761-a7b8-d6ca71e33cb0" />
)

> **Note:** All traffic is simulated. The project needs no real network or corporate systems, so anyone can run and verify it.

---

## Table of contents
1. [Problem statement](#problem-statement)
2. [Industry relevance](#industry-relevance)
3. [What it does](#what-it-does)
4. [Tech stack](#tech-stack)
5. [Architecture](#architecture)
6. [Dataset and simulation](#dataset-and-simulation)
7. [Installation](#installation)
8. [How to run](#how-to-run)
9. [Results](#results)
10. [API reference](#api-reference)
11. [Screenshots](#screenshots)
12. [Project structure](#project-structure)
13. [Learning outcomes](#learning-outcomes)
14. [Future work](#future-work)

---

## Problem statement

Security teams receive millions of network events a day. Rule-based tools catch known attacks but miss new ones, and analysts drown in alerts. This project asks: **can a model learn what normal traffic looks like, recognise the main attack types, and still flag behaviour it has never seen, with few false alarms?**

## Industry relevance

Banks, IT services firms and product companies run the same idea inside their Security Operations Centres (SOC):

| Use case | How it shows up here |
|---|---|
| Intrusion detection | Random Forest classifies each connection as normal or one of 4 attack types |
| Anomaly detection | Isolation Forest learns normal traffic only, so it can flag unknown threats |
| Network security monitoring | A scored live stream and alert feed on the dashboard |
| Incident response | Every alert carries a severity and a recommended action (playbook) |
| Fraud-style detection | The same "learn normal, flag the unusual" approach used for suspicious transactions |

## What it does

1. Simulates labelled network traffic, including stealthy attacks that look almost normal.
2. Cleans the data and engineers features (log-scaled byte counts, bytes per second, ports per connection).
3. Trains three models: Random Forest, Logistic Regression (baseline) and Isolation Forest (unsupervised).
4. Evaluates with accuracy, precision, recall and F1 on unseen data.
5. Scores a fresh stream of 1,500 connections and raises alerts with severity levels.
6. Draws five charts and serves a themed dashboard plus a REST API.

## Tech stack

| Area | Tools |
|---|---|
| Language | Python 3.10+ |
| Data | Pandas, NumPy |
| Machine learning | scikit-learn (Random Forest, Logistic Regression, Isolation Forest) |
| Charts | Matplotlib, Seaborn |
| API and dashboard | Flask, HTML, CSS, JavaScript (canvas) |
| Model storage | joblib |

No GPU is needed. A full run takes about 12 seconds on a laptop.

## Architecture

```
 +------------------+     +-------------+     +-------------+     +--------------+
 | Traffic simulator| --> |  Cleaning   | --> |  Feature    | --> | Train / test |
 | (or your CSV)    |     | preprocess  |     |  engineering|     |    split     |
 +------------------+     +-------------+     +-------------+     +------+-------+
                                                                         |
                    +----------------------------------------------------+
                    v
        +------------------------+      +---------------------------+
        |  Model training        | ---> |  Evaluation               |
        |  Random Forest         |      |  accuracy, precision,     |
        |  Logistic Regression   |      |  recall, F1               |
        |  Isolation Forest      |      +---------------------------+
        +-----------+------------+
                    |  saved to models/
                    v
        +------------------------+      +---------------------------+
        |  Detection engine      | ---> |  Alerts + severity +      |
        |  scores a live stream  |      |  response playbook        |
        +------------------------+      +-------------+-------------+
                                                      v
                                        +---------------------------+
                                        |  Flask API + dashboard    |
                                        |  charts in outputs/images |
                                        +---------------------------+
```

**How a connection becomes an alert**
1. The Random Forest predicts the traffic type and a confidence.
2. The Isolation Forest scores how unusual the connection is.
3. If the forest says "normal" but the anomaly score is below a threshold (set from the first percentile of normal training traffic), the alert is raised as *unusual behaviour*.
4. Each alert type maps to a severity (low, medium, high, critical) and a recommended response.

## Dataset and simulation

Training data is generated by `src/threat_core.py` so the project runs offline and is fully reproducible (fixed random seed).

| Class | What the simulator models | Severity |
|---|---|---|
| Normal | Mixed web, DNS and mail traffic with modest volumes | none |
| DoS flood | Very high packet rate, hundreds of connections, mostly half-open | critical |
| Data exfiltration | Few long connections sending very large outbound transfers | critical |
| Brute force | Many failed logins against SSH, FTP or SMTP | high |
| Port scan | Dozens of distinct ports probed with tiny payloads | medium |

Features per connection: duration, bytes sent and received, packets per second, connection count, failed logins, SYN error rate, distinct ports, protocol and service.

About 4% of attacks are made to look nearly normal, so results are realistic rather than a perfect 100%. Attacker addresses come from reserved documentation ranges, so none belong to a real host.

**Using a real dataset:** run `python rullall.py --data your_file.csv`. The CSV needs the columns `duration, src_bytes, dst_bytes, pkts_per_sec, conn_count, failed_logins, syn_error_rate, unique_ports, protocol, service, label`. Public sets such as UNSW-NB15, CICIDS or KDD Cup need their columns mapped to these first.

## Installation

```bash
git clone https://github.com/<your-username>/ai-cybersecurity-threat-detection.git
cd ai-cybersecurity-threat-detection

# Windows
python -m venv venv
venv\Scripts\activate

# macOS / Linux
python3 -m venv venv
source venv/bin/activate

pip install -r requirements.txt
```

## How to run

```bash
python rullall.py --open
```

This runs all nine pipelines, then starts the dashboard at http://127.0.0.1:8000.

| Command | What it does |
|---|---|
| `python rullall.py` | Run every pipeline, then start the API |
| `python rullall.py --no-api` | Pipelines only |
| `python rullall.py --api-only` | Skip training, reuse saved models, start the API |
| `python rullall.py --samples 50000` | Simulate more training data |
| `python rullall.py --port 8001` | Use another port |
| `python rullall.py --data my.csv` | Train on your own CSV |

**Pipelines run by `rullall.py`:** setup, data, clean, features, train, evaluate, detect, visualize, report.

Try a prediction from the terminal while the API is running:

```bash
curl -X POST http://127.0.0.1:8000/api/predict \
  -H "Content-Type: application/json" \
  -d '{"duration":0.3,"src_bytes":250,"dst_bytes":20,"pkts_per_sec":700,
       "conn_count":230,"failed_logins":0,"syn_error_rate":0.85,
       "unique_ports":1,"protocol":"tcp","service":"http"}'
```

A successful detection returns `"verdict": "dos"`, `"severity": "critical"`, the model confidence, the reasons and a recommended action.

## Results

Measured on connections the models never saw during training (20,000 simulated connections, 75/25 split, seed 42):

| Model | Accuracy | Precision | Recall | F1 |
|---|---|---|---|---|
| Random Forest | 98.9% | 98.9% | 98.9% | 98.9% |
| Logistic Regression | 98.7% | 98.7% | 98.7% | 98.7% |
| Isolation Forest (attack vs normal, no labels) | 95.9% | 97.6% | 88.3% | 92.7% |

**Live detection test** on a fresh stream of 1,500 connections:

| Measure | Result |
|---|---|
| Alerts raised | 443 |
| Simulated attacks caught | 433 of 450 (96.2%) |
| Normal traffic falsely flagged | 0.95% |

Exact figures change slightly with the seed and data size. The simulated data is cleaner than real traffic, so expect lower scores on public datasets.

## API reference

| Method | Endpoint | Returns |
|---|---|---|
| GET | `/` | Dashboard |
| GET | `/api/health` | Service status and model info |
| GET | `/api/stats` | Headline numbers, alert mix, top source IPs |
| GET | `/api/alerts?severity=high&limit=40` | Recent alerts |
| GET | `/api/stream` | Recent scored connections |
| GET | `/api/metrics` | Model scores from the last run |
| GET | `/api/sample?kind=dos` | A sample connection of a given type |
| POST | `/api/predict` | Verdict, confidence, reasons and action for one connection |

## Screenshots

Save your own captures in `images/` using these names so the links below work.

| | |
|---|---|
| ![Dashboard](images/dashboard.png) **Dashboard** | ![Prediction](images/prediction_result.png) **Checking a connection** |
| ![Confusion matrix](images/confusion_matrix.png) **Confusion matrix** | ![Anomaly scores](images/anomaly_scores.png) **Anomaly scores** |
| ![Feature importance](images/feature_importance.png) **Feature importance** | ![Alert timeline](images/alert_timeline.png) **Alert timeline** |
| ![Training log](images/training_log.png) **Terminal run of rullall.py** | ![Traffic mix](images/attack_distribution.png) **Traffic mix** |

The five charts are created in `outputs/images/` after each run. Copy them to `images/` before committing.

## Project structure

```
ai-cybersecurity-threat-detection/
├── rullall.py            # runs every pipeline, then starts the API
├── requirements.txt
├── README.md
├── LICENSE
├── .gitignore
├── images/               # screenshots used in this README
├── src/
│   ├── threat_core.py    # simulator, features, theme, shared helpers
│   ├── data_loader.py    # setup folders, simulate or load data
│   ├── preprocess.py     # cleaning
│   ├── features.py       # feature engineering and train/test split
│   ├── train.py          # three models
│   ├── evaluate.py       # metrics
│   ├── detect.py         # live-stream scoring and alerts
│   ├── visualize.py      # charts
│   ├── report.py         # run summary
│   └── api_server.py     # Flask API and dashboard
├── data/                 # created on run (ignored by git)
├── models/               # created on run (ignored by git)
└── outputs/              # created on run: metrics, alerts, charts
```

## Learning outcomes

- Built an end-to-end ML pipeline: data, cleaning, features, training, evaluation, deployment.
- Compared supervised (Random Forest, Logistic Regression) and unsupervised (Isolation Forest) detection and saw why both are needed.
- Handled class imbalance with class weights and judged models by precision and recall, not accuracy alone.
- Turned model output into SOC-style alerts with severity levels and response actions.
- Designed a modular codebase where each pipeline is its own file.
- Served models through a REST API and a dashboard.

## Future work

- Map public datasets (UNSW-NB15, CICIDS2017) into the input format.
- Add a deep-learning autoencoder for anomaly detection.
- Stream real packet captures through the detector.
- Add authentication, Docker packaging and automated tests.

## Author

**Your Name** | [LinkedIn](https://linkedin.com/in/your-profile) | [GitHub](https://github.com/your-username)

Licensed under the MIT License. See `LICENSE`.

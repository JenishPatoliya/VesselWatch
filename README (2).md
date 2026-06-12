# ⚓ VesselWatch
### Real-Time Maritime Anomaly Detection Using Machine Learning

[![Live Demo](https://img.shields.io/badge/🚀_Live_Demo-Open_Dashboard-00b37e?style=for-the-badge)](https://vesselwatch-vmzc2tetunqfqmv7wgblw7.streamlit.app/)
[![Python](https://img.shields.io/badge/Python-3.10-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-Deployed-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)](https://streamlit.io)

---

VesselWatch processes **5.6 million real AIS vessel tracking records** to automatically detect suspicious ship behavior — dark voyages, illegal rendezvous, loitering, and more. Every flagged vessel comes with a SHAP-based explanation so analysts know *why* it was flagged, not just that it was.

**[👉 Open the live dashboard](https://vesselwatch-vmzc2tetunqfqmv7wgblw7.streamlit.app/)**

![VesselWatch Dashboard](assets/dashboard.png)
*Live map showing 16,937 vessels — red = high risk, orange = medium, blue = normal. Flagged vessels listed by risk score on the right.*

---

## The Problem

Thousands of ships cross the world's oceans every day. Coast guards cannot monitor all of them manually. Criminals exploit this — they turn off GPS mid-ocean, transfer illegal cargo between ships, then reappear looking completely normal.

Rule-based systems fail because criminals learn the rules. **VesselWatch learns what normal looks like — and flags everything that deviates, automatically.**

---

## Results

| Metric | Value |
|--------|-------|
| Vessels analyzed | **16,937** |
| Suspicious vessels flagged | **1,245** |
| Precision | **64.34%** |
| Recall | **23.89%** |
| F1 Score | **34.84%** |
| ROC-AUC | **0.8076** |

### Behavioral validation

|  | 🔴 High risk vessels | 🟢 Normal vessels |
|--|--|--|
| Avg AIS gap | **59.47 hrs** | 23.82 hrs |
| Avg position jump | **1,315 km** | 24 km |
| Avg distance from port | **230 km** | 22 km |

> High-risk vessels jump **54× further** than normal vessels after an AIS gap — confirming the model correctly identifies genuinely suspicious behavior, not random noise.

---

## What Gets Detected

| Signal | Description |
|--------|-------------|
| 🔴 **Dark events** | Ship disappears from radar mid-ocean with no AIS signal |
| 🟠 **Position jumps** | Vessel reappears far from its expected location |
| 🔵 **Rendezvous** | Two ships meet in open ocean — possible cargo transfer |
| 🟣 **Loitering** | Ship circles the same area for extended hours |
| ⚡ **Speed anomalies** | Erratic or physically impossible speed changes |
| 🧠 **Behavioral shift** | Vessel acts differently from its own 7-day baseline |

![Anomaly Breakdown](assets/analytics.png)
*Left: anomaly detections by vessel type — cargo and tankers dominate. Right: dark events account for 53% of all anomalies detected.*

---

## ML Pipeline

```
5.6M Raw AIS Pings (7 days)
        │
        ▼
┌─────────────────────┐
│    Data Cleaning     │  Remove invalid MMSIs · Fix sensor errors · Deduplicate
└─────────────────────┘
        │
        ▼
┌─────────────────────┐
│ Feature Engineering  │  15 behavioral features per vessel
│                     │  Each vessel fingerprinted against its own 7-day history
└─────────────────────┘
        │
        ▼
┌─────────────────────────────────────────┐
│            Ensemble ML Models            │
│                                         │
│  Isolation Forest  ──  outliers (70%)   │
│  DBSCAN  ──────────  rendezvous (30%)   │
│  LSTM Autoencoder  ──  trajectories     │
└─────────────────────────────────────────┘
        │
        ▼
┌─────────────────────┐
│  SHAP Explainability │  Why was this vessel flagged?
│                     │  Per-feature contribution scores for every result
└─────────────────────┘
        │
        ▼
┌─────────────────────┐
│ Interactive Dashboard│  Live map · Risk filters · SHAP charts · Vessel drill-down
└─────────────────────┘
```

---

## SHAP Explainability

Every flagged vessel comes with a full breakdown of which features drove the risk score — and in which direction.

![SHAP Explanation](assets/shap.png)
*VISION OF THE SEAS (risk score 0.77): minimum speed and position jump are the dominant anomaly drivers. Red bars push risk up, blue push it down.*

---

## Tech Stack

| Layer | Tools |
|-------|-------|
| Data processing | Python · Pandas · NumPy |
| Machine learning | Scikit-learn · TensorFlow · Keras |
| Explainability | SHAP |
| Mapping | Folium · GeoPandas |
| Dashboard | Streamlit |
| Deployment | Streamlit Cloud |

---

## Most Suspicious Vessels Found

| Vessel | Type | AIS Gap | Position Jump |
|--------|------|---------|---------------|
| EVER LUCENT | Cargo | 117 hrs | 4,143 km |
| MARJORIE C | Cargo | 118 hrs | 3,613 km |
| SILVER MONIKA | Tanker | 118 hrs | 2,274 km |
| LA MANCHA KNUTSEN | Tanker | 69 hrs | 1,776 km |

---

## Repository Structure

```
VesselWatch/
├── app.py                          ← Streamlit dashboard entry point
├── requirements.txt                ← All Python dependencies
├── final_results.csv               ← ML outputs for 16,937 vessels
├── shap_explanations.csv           ← SHAP values for top 50 flagged vessels
├── README.md
├── assets/
│   ├── dashboard.png               ← Main dashboard screenshot
│   ├── analytics.png               ← Charts and anomaly breakdown
│   └── shap.png                    ← SHAP explainability view
└── notebooks/
    └── VesselWatch_V2.ipynb        ← Full ML pipeline, step by step
```

---

## Run Locally

```bash
# Clone the repo
git clone https://github.com/JenishPatoliya/VesselWatch
cd VesselWatch

# Install dependencies
pip install -r requirements.txt

# Launch the dashboard
streamlit run app.py
```

Then open [http://localhost:8501](http://localhost:8501) in your browser.

---

## Author

**Jenish Patoliya**
📧 [jenishpatoliya1911@gmail.com](mailto:jenishpatoliya1911@gmail.com)
🐙 [github.com/JenishPatoliya](https://github.com/JenishPatoliya)

---

<div align="center">Built for maritime security · Powered by real AIS data</div>

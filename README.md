# Pest & Disease Early Warning – Flask Demo

This project implements **Feature 2: Pest & Disease Early Warning** as an end-to-end demo.

It simulates:

- Sentinel-2 NDVI/LAI and MODIS LAI
- Weather time series (Open-Meteo-like)
- FAO bulletins & social media chatter (as embeddings)
- Plant data (Perenual-like profile embedding)
- Land-use context (OpenStreetMap-style)
- Optional local extension data & farmer pest sightings (text fields in UI)

and feeds them into:

- A **multimodal transformer encoder** (`MultimodalEncoder`)
- A **UNet-like pest/disease risk generator** (`PestRiskGenerator`)
- A **Dreamer-style intervention policy head** (`InterventionPolicy`)
- A (not yet wired) **GNN** for spatial propagation (`PestPropagationGNN`)

The Flask UI exposes:

- A web form to configure field ID, lat/lon and crop type
- Optional extension/farmer notes
- A dashboard showing:
  - Pest/disease risk heatmap
  - Risk trend over time
  - Crop outbreak probabilities
  - Species-level alerts and scores
  - Suggested control measures
  - Intervention timing recommendation
  - Policy probabilities over timing actions

> ⚠️ All data is synthetic. To make this production-ready, swap out
> `backend/data/data_sources.py` with real API clients and data pipelines.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Run

```bash
python app.py
```

Then open: `http://127.0.0.1:5000/`

## API

You can also call the JSON API:

```bash
curl -X POST http://127.0.0.1:5000/api/analyze \
  -H "Content-Type: application/json" \
  -d '{"field_id":"field-001","lat":10.0,"lon":77.0,"crop_type":"Maize"}'
```

It returns a JSON structure with:

- `risk_map`
- `mean_risk`, `max_risk`
- `zones`
- `crop_outbreak_probs`
- `species_alerts`
- `control_measures`
- `risk_trend`
- `policy_probs`, `policy_value`
- `timing_recommendation`
- `alerts`

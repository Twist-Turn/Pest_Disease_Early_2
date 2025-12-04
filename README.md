---
title: Pest Disease Early Warning System
emoji: 🌾
colorFrom: green
colorTo: yellow
sdk: docker
pinned: false
license: mit
---

# Pest & Disease Early Warning System

An AI-powered early warning system for agricultural pest and disease detection using multimodal data fusion.

## Features

- **Satellite Data Analysis**: Processes LAI/NDVI data from Sentinel-2 and MODIS
- **Weather Integration**: Incorporates weather patterns for risk assessment
- **Risk Heatmaps**: Generates spatial risk maps for pest/disease outbreaks
- **Species-Specific Alerts**: Identifies specific pests and diseases
- **Intervention Timing**: AI-driven recommendations for optimal intervention timing
- **Trend Analysis**: Historical risk tracking and forecasting

## Technology Stack

- **Backend**: Flask, PyTorch
- **Models**: 
  - Multimodal Transformer for data fusion
  - UNet-style risk generator
  - RL-based intervention policy
  - GNN for spatial propagation
- **Data**: Synthetic satellite, weather, and land-use data

## Usage

1. Enter field information (ID, coordinates, crop type)
2. Add optional notes about pest sightings or extension observations
3. Click "Analyze Field" to generate risk assessment
4. Review the risk heatmap, alerts, and recommendations

## Demo Mode

This demo uses synthetic data to demonstrate the system's capabilities. In production, it would integrate:
- Real Sentinel-2/MODIS satellite imagery
- Live weather data from Open-Meteo
- FAO bulletins and social media monitoring
- User-submitted pest sightings

## API Endpoint

POST `/api/analyze` with JSON:
```json
{
  "field_id": "field-001",
  "lat": 10.0,
  "lon": 77.0,
  "crop_type": "Maize"
}
```

## License

MIT License

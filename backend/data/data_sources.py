"""
Synthetic data layer for Pest & Disease Early Warning.

In a real system you would:
- Fetch Sentinel-2 NDVI/LAI and MODIS LAI
- Query Open-Meteo for weather forecasts / histories
- Parse FAO bulletins and social media streams
- Query plant information APIs (e.g. Perenual)
- Derive land-use vectors from OpenStreetMap
- Accept user pest sightings & extension data
"""

from dataclasses import dataclass
from typing import Dict, Any, Optional, List
import numpy as np

@dataclass
class FieldContext:
    field_id: str
    lat: float
    lon: float
    crop_type: str = "Maize"
    local_extension_text: Optional[str] = None
    user_pest_notes: Optional[str] = None


CROPS = ["Maize", "Wheat", "Rice", "Soybean"]
PESTS = ["Fall armyworm", "Brown planthopper", "Stem borer", "Rust fungus"]


def _seasonal_pattern(t: int, period: int = 12) -> float:
    return 0.5 + 0.4 * np.sin(2 * np.pi * t / period)


def load_satellite_lai(field: FieldContext, timesteps: int = 12) -> np.ndarray:
    """
    Synthetic LAI/NDVI patterns with possible hotspots for stress.
    Channels: [NDVI, LAI_sentinel, LAI_modis]
    Shape: (T, H, W, C)
    """
    H, W, C = 64, 64, 3
    arr = np.zeros((timesteps, H, W, C), dtype="float32")
    for t in range(timesteps):
        season = _seasonal_pattern(t, period=timesteps)
        noise = 0.05 * np.random.randn(H, W, C).astype("float32")
        ndvi = season + noise[..., 0]
        lai_s2 = season * 1.2 + noise[..., 1]
        lai_modis = season * 1.1 + noise[..., 2]
        arr[t, ..., 0] = ndvi
        arr[t, ..., 1] = lai_s2
        arr[t, ..., 2] = lai_modis

    # Inject a pest/disease-like patch with lowered NDVI & LAI
    cx, cy = np.random.randint(H // 4, 3 * H // 4), np.random.randint(W // 4, 3 * W // 4)
    r = H // 8
    y, x = np.ogrid[:H, :W]
    mask = (x - cx) ** 2 + (y - cy) ** 2 <= r ** 2
    arr[-1, mask, :] *= 0.6
    arr = np.clip(arr, 0.0, 2.0)
    return arr


def load_weather_series(field: FieldContext, timesteps: int = 12) -> np.ndarray:
    """
    Weather features: [temp (C), precip (mm), humidity (%), wind (m/s)]
    Normalized to ~[0,1].
    """
    t = np.arange(timesteps)
    temp = 24 + 6 * np.sin(2 * np.pi * t / timesteps) + np.random.randn(timesteps)
    precip = np.maximum(0, np.random.gamma(shape=2.0, scale=4.0, size=timesteps))
    humidity = 65 + 10 * np.sin(2 * np.pi * (t + 2) / timesteps) + 5 * np.random.randn(timesteps)
    wind = 2 + np.abs(np.random.randn(timesteps))

    series = np.stack(
        [
            (temp - 10) / 25.0,
            precip / 30.0,
            humidity / 100.0,
            wind / 10.0,
        ],
        axis=1,
    ).astype("float32")
    return series


def load_landuse_features(field: FieldContext) -> np.ndarray:
    """
    Land-use context around the field (synthetic):
    - fraction of cropland, forest, urban, water, grassland
    - distance to nearest urban area, water body, forest patch (scaled)
    """
    fractions = np.random.dirichlet(np.ones(5)).astype("float32")
    distances = np.random.rand(3).astype("float32")
    return np.concatenate([fractions, distances], axis=0)


def load_text_signals(field: FieldContext, timesteps: int = 12) -> Dict[str, Any]:
    """
    Text-like signals backed by embeddings:
    - FAO bulletins
    - Social media chatter
    - Plant info
    For the demo, we only generate random embeddings, but keep shapes explicit.

    Returns:
      {
        "fao_embeddings": (T, 64),
        "social_embeddings": (T, 64),
        "plant_profile": (64,)
      }
    """
    fao_embeddings = np.random.randn(timesteps, 64).astype("float32") * 0.2
    social_embeddings = np.random.randn(timesteps, 64).astype("float32") * 0.2
    plant_profile = np.random.randn(64).astype("float32") * 0.2
    return {
        "fao_embeddings": fao_embeddings,
        "social_embeddings": social_embeddings,
        "plant_profile": plant_profile,
    }


def load_all_inputs(field: FieldContext, timesteps: int = 12) -> Dict[str, Any]:
    return {
        "lai": load_satellite_lai(field, timesteps),
        "weather": load_weather_series(field, timesteps),
        "landuse": load_landuse_features(field),
        "text": load_text_signals(field, timesteps),
    }

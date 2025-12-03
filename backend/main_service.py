"""
Pest & Disease Early Warning engine.

Combines:
- Satellite LAI/NDVI
- Weather series
- Land-use context
- Text-like embeddings (FAO bulletins, social media, plant data)

Outputs:
- Pest/disease risk heatmap per field
- Outbreak probability per crop
- Species-specific alerts
- Trend analysis of risk
- Suggested control measures
- Intervention timing recommendations (RL-style)
"""

from typing import Dict, Any, List
import numpy as np
import torch
from sklearn.cluster import KMeans

from backend.data.data_sources import FieldContext, load_all_inputs, CROPS, PESTS
from backend.models import MultimodalEncoder, PestRiskGenerator, InterventionPolicy
from backend.utils.config import DEVICE


class PestDiseaseEngine:
    def __init__(self, timesteps: int = 12, forecast_horizon: int = 6, num_zones: int = 4):
        self.timesteps = timesteps
        self.forecast_horizon = forecast_horizon
        self.num_zones = num_zones

        # token_dim: lai channels (3) + weather (4) + landuse (8) = 15
        token_dim = 3 + 4 + 8
        text_dim = 64

        self.encoder = MultimodalEncoder(token_dim=token_dim, text_dim=text_dim).to(DEVICE)
        self.risk_gen = PestRiskGenerator(in_channels=3).to(DEVICE)
        self.policy = InterventionPolicy(state_dim=256, num_actions=4).to(DEVICE)

        self.encoder.eval()
        self.risk_gen.eval()
        self.policy.eval()

    # ------------ helpers ------------

    def _prepare_tokens(self, inputs: Dict[str, Any]) -> torch.Tensor:
        lai = inputs["lai"]  # (T, H, W, C=3)
        weather = inputs["weather"]  # (T, 4)
        landuse = inputs["landuse"]  # (8,)
        T, H, W, C = lai.shape
        lai_flat = lai.reshape(T, H * W, C)

        tokens = []
        for t in range(T):
            s_t = lai_flat[t]  # (HW, 3)
            w_t = np.repeat(weather[t][None, :], H * W, axis=0)  # (HW, 4)
            lu = np.repeat(landuse[None, :], H * W, axis=0)      # (HW, 8)
            tok = np.concatenate([s_t, w_t, lu], axis=1)         # (HW, 15)
            tokens.append(tok)

        tokens = np.stack(tokens, axis=0)  # (T, HW, 15)
        tokens = tokens.reshape(1, -1, tokens.shape[-1])  # (1, T*HW, 15)
        return torch.from_numpy(tokens).float().to(DEVICE)

    def _prepare_text_inputs(self, inputs: Dict[str, Any]) -> Dict[str, torch.Tensor]:
        text = inputs["text"]
        fao = text["fao_embeddings"]            # (T, 64)
        social = text["social_embeddings"]      # (T, 64)
        plant = text["plant_profile"]           # (64,)

        # fuse FAO + social as time sequence
        fused = 0.5 * fao + 0.5 * social

        fused = torch.from_numpy(fused[None, ...]).float().to(DEVICE)  # (1, T, 64)
        plant = torch.from_numpy(plant[None, ...]).float().to(DEVICE)  # (1, 64)

        return {"text_seq": fused, "plant_profile": plant}

    def _prepare_risk_input(self, inputs: Dict[str, Any]) -> torch.Tensor:
        lai = inputs["lai"]  # (T, H, W, C)
        last = lai[-1]       # (H, W, C)
        img = np.transpose(last, (2, 0, 1))  # (C, H, W)
        img = img[None, ...]
        return torch.from_numpy(img).float().to(DEVICE)

    def _compute_zone_stats(self, risk_map: np.ndarray, num_zones: int) -> Dict[str, Any]:
        H, W = risk_map.shape
        ys, xs = np.mgrid[0:H, 0:W]
        features = np.stack(
            [
                xs.flatten() / W,
                ys.flatten() / H,
                risk_map.flatten(),
            ],
            axis=1,
        )
        kmeans = KMeans(n_clusters=num_zones, n_init=5, random_state=42)
        labels = kmeans.fit_predict(features)
        labels_img = labels.reshape(H, W)

        zone_stats = []
        for z in range(num_zones):
            mask = labels_img == z
            if mask.sum() == 0:
                mean_risk = float("nan")
            else:
                mean_risk = float(risk_map[mask].mean())
            zone_stats.append(
                {
                    "zone_id": int(z),
                    "mean_risk": mean_risk,
                    "area_fraction": float(mask.mean()),
                }
            )

        return {"labels": labels_img.tolist(), "zone_stats": zone_stats}

    def _compute_crop_outbreak_probs(self, encoded_state: np.ndarray) -> Dict[str, float]:
        """
        Simple mapping from encoded state to per-crop risk probabilities.
        """
        vec = encoded_state  # (256,)
        scores = np.maximum(0, np.tanh(vec[: len(CROPS)] * 2.0)) + 0.1
        probs = scores / scores.sum()
        return {crop: float(p) for crop, p in zip(CROPS, probs)}

    def _compute_species_alerts(self, risk_map: np.ndarray, crop_probs: Dict[str, float]) -> List[Dict[str, Any]]:
        mean_risk = float(risk_map.mean())
        max_risk = float(risk_map.max())

        base_levels = np.linspace(0.1, 0.9, num=len(PESTS))
        alerts = []
        for pest, base in zip(PESTS, base_levels):
            score = 0.5 * mean_risk + 0.5 * max_risk * base
            alerts.append(
                {
                    "species": pest,
                    "risk_score": float(score),
                    "alert_level": "High" if score > 0.7 else "Medium" if score > 0.4 else "Low",
                }
            )
        return alerts

    def _compute_control_measures(self, species_alerts: List[Dict[str, Any]]) -> List[str]:
        recs: List[str] = []
        for alert in species_alerts:
            if alert["alert_level"] == "Low":
                continue
            s = alert["species"]
            if "fungus" in s.lower():
                recs.append(f"For **{s}**, consider prophylactic fungicide or resistant varieties in high-risk zones.")
            elif "worm" in s.lower() or "borer" in s.lower():
                recs.append(f"For **{s}**, use pheromone traps and targeted biological insecticides (e.g. Bt) before peak infestation.")
            else:
                recs.append(f"For **{s}**, follow local IPM guidelines and integrate biological and chemical control carefully.")
        if not recs:
            recs.append("Current risk is low; maintain regular scouting and avoid unnecessary pesticide use.")
        return recs

    def _compute_risk_trend(self, inputs: Dict[str, Any]) -> List[float]:
        """
        Derive a synthetic risk index over past timesteps based on decline in LAI/NDVI.
        """
        lai = inputs["lai"]  # (T, H, W, C)
        T = lai.shape[0]
        risk_series = []
        for t in range(T):
            frame = lai[t]
            mean_ndvi = frame[..., 0].mean()
            # higher risk when NDVI is low
            risk = float(np.clip(1.0 - mean_ndvi / 2.0, 0.0, 1.0))
            risk_series.append(risk)
        return risk_series

    # ------------ public API ------------

    def run_field(self, field_id: str, lat: float, lon: float, crop_type: str = "Maize") -> Dict[str, Any]:
        field = FieldContext(field_id=field_id, lat=lat, lon=lon, crop_type=crop_type)
        inputs = load_all_inputs(field, timesteps=self.timesteps)

        tokens = self._prepare_tokens(inputs)
        txt = self._prepare_text_inputs(inputs)
        risk_in = self._prepare_risk_input(inputs)

        with torch.no_grad():
            enc = self.encoder(tokens, txt["text_seq"], txt["plant_profile"])  # (1,256)
            risk_map = self.risk_gen(risk_in)                                  # (1,1,H,W)
            policy_logits, policy_value = self.policy(enc)                     # (1,A),(1,1)

            policy_probs = torch.softmax(policy_logits, dim=-1).cpu().numpy()[0]
            policy_value = float(policy_value.cpu().numpy()[0, 0])

        risk_np = risk_map.cpu().numpy()[0, 0]
        encoded_state = enc.cpu().numpy()[0]

        mean_risk = float(risk_np.mean())
        max_risk = float(risk_np.max())

        zones = self._compute_zone_stats(risk_np, num_zones=self.num_zones)
        crop_probs = self._compute_crop_outbreak_probs(encoded_state)
        species_alerts = self._compute_species_alerts(risk_np, crop_probs)
        control_measures = self._compute_control_measures(species_alerts)
        risk_trend = self._compute_risk_trend(inputs)

        actions = [
            "Intervene immediately in high-risk zones.",
            "Schedule intervention within 3 days.",
            "Increase monitoring; defer intervention for now.",
            "Risk currently low; maintain routine scouting only.",
        ]
        best_action_idx = int(policy_probs.argmax())
        timing_recommendation = actions[best_action_idx]

        # Alerts for UI
        alerts: List[str] = []
        if max_risk > 0.85:
            alerts.append("Severe pest/disease risk detected in localized hotspots.")
        if mean_risk > 0.6:
            alerts.append("Field-wide pest/disease risk is elevated; plan interventions.")
        if not alerts:
            alerts.append("Current risk is moderate to low; continue targeted scouting.")

        return {
            "field_id": field_id,
            "location": {"lat": lat, "lon": lon},
            "crop_type": crop_type,
            "risk_map": risk_np.tolist(),
            "mean_risk": mean_risk,
            "max_risk": max_risk,
            "zones": zones,
            "crop_outbreak_probs": crop_probs,
            "species_alerts": species_alerts,
            "control_measures": control_measures,
            "risk_trend": risk_trend,
            "policy_probs": policy_probs.tolist(),
            "policy_value": policy_value,
            "timing_recommendation": timing_recommendation,
            "alerts": alerts,
        }


if __name__ == "__main__":
    engine = PestDiseaseEngine()
    out = engine.run_field("demo-field", 10.0, 77.0, "Maize")
    print("Timing:", out["timing_recommendation"])
    print("Alerts:", out["alerts"])

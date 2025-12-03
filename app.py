import uuid
from datetime import datetime

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from flask import Flask, render_template, request, jsonify

from backend.main_service import PestDiseaseEngine
from backend.utils.config import STATIC_GEN_DIR

app = Flask(__name__)
engine = PestDiseaseEngine()


def _save_heatmap(risk_map: np.ndarray, filename: str) -> str:
    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(risk_map, cmap="Reds")
    fig.colorbar(im, ax=ax, label="Pest/Disease risk")
    ax.set_title("Pest & Disease Risk Heatmap")
    ax.axis("off")
    path = STATIC_GEN_DIR / filename
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return f"generated/{filename}"


def _save_trend(risk_trend: np.ndarray, filename: str) -> str:
    fig, ax = plt.subplots(figsize=(5, 3))
    ax.plot(range(1, len(risk_trend) + 1), risk_trend, marker="o")
    ax.set_xlabel("Past timestep")
    ax.set_ylabel("Risk index (normalized)")
    ax.set_title("Pest/Disease Risk Trend")
    ax.grid(True)
    path = STATIC_GEN_DIR / filename
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return f"generated/{filename}"


@app.route("/", methods=["GET"])
def index():
    return render_template("index.html")


@app.route("/analyze", methods=["POST"])
def analyze():
    field_id = request.form.get("field_id", "field-001")
    lat = float(request.form.get("lat", "10.0"))
    lon = float(request.form.get("lon", "77.0"))
    crop_type = request.form.get("crop_type", "Maize")
    notes_extension = request.form.get("notes_extension", "")
    notes_pest = request.form.get("notes_pest", "")

    # For now notes are not used in the engine, but could be wired into text embeddings.
    result = engine.run_field(field_id, lat, lon, crop_type=crop_type)

    risk = np.array(result["risk_map"])
    trend = np.array(result["risk_trend"])

    run_id = uuid.uuid4().hex[:8]
    heatmap_file = f"risk_{run_id}.png"
    trend_file = f"trend_{run_id}.png"

    heatmap_path = _save_heatmap(risk, heatmap_file)
    trend_path = _save_trend(trend, trend_file)

    context = {
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "field_id": field_id,
        "lat": lat,
        "lon": lon,
        "crop_type": crop_type,
        "notes_extension": notes_extension,
        "notes_pest": notes_pest,
        "heatmap_path": heatmap_path,
        "trend_path": trend_path,
        **result,
    }
    return render_template("results.html", **context)


@app.route("/api/analyze", methods=["POST"])
def api_analyze():
    data = request.json or {}
    field_id = data.get("field_id", "field-001")
    lat = float(data.get("lat", 10.0))
    lon = float(data.get("lon", 77.0))
    crop_type = data.get("crop_type", "Maize")

    result = engine.run_field(field_id, lat, lon, crop_type=crop_type)
    return jsonify(result)


if __name__ == "__main__":
    app.run(debug=True)

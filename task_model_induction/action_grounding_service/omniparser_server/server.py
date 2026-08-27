from __future__ import annotations

import argparse
import base64
import io
import os
from pathlib import Path
from typing import Any

import numpy as np
from flask import Flask, jsonify, request
from flask.json.provider import DefaultJSONProvider
from flask_cors import CORS
from huggingface_hub import hf_hub_download
from PIL import Image
from ultralytics import YOLO


WEIGHTS_DIR = Path(os.getenv("OMNIPARSER_WEIGHTS_DIR", "/weights"))
MODEL_REPO = os.getenv("OMNIPARSER_MODEL_REPO", "microsoft/OmniParser-v2.0")
MODEL_FILE = os.getenv("OMNIPARSER_MODEL_FILE", "icon_detect/model.pt")

app = Flask(__name__)
CORS(app)
model: YOLO | None = None


class NumpyJSONProvider(DefaultJSONProvider):
    def default(self, obj: Any) -> Any:
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)


app.json = NumpyJSONProvider(app)


def load_model() -> YOLO:
    model_path = hf_hub_download(
        repo_id=MODEL_REPO,
        filename=MODEL_FILE,
        local_dir=WEIGHTS_DIR,
    )
    return YOLO(model_path)


def get_model() -> YOLO:
    global model
    if model is None:
        model = load_model()
    return model


@app.get("/health")
def health():
    return jsonify(
        {
            "status": "healthy",
            "models_loaded": model is not None,
            "model_repo": MODEL_REPO,
            "model_file": MODEL_FILE,
        }
    )


@app.post("/parse")
def parse_json():
    if not request.is_json:
        return jsonify({"error": "Content-Type must be application/json"}), 400
    payload = request.get_json()
    if not isinstance(payload, dict) or "image" not in payload:
        return jsonify({"error": "Missing required field: image"}), 400
    try:
        image_data = payload["image"]
        if isinstance(image_data, str) and image_data.startswith("data:"):
            image_data = image_data.split(",", 1)[1]
        image = Image.open(io.BytesIO(base64.b64decode(image_data))).convert("RGB")
    except Exception as exc:
        return jsonify({"error": f"Failed to decode image: {exc}"}), 400
    return _parse_image_response(image, payload)


@app.post("/parse/file")
def parse_file():
    if "image" not in request.files:
        return jsonify({"error": "No image file provided"}), 400
    try:
        image = Image.open(request.files["image"].stream).convert("RGB")
    except Exception as exc:
        return jsonify({"error": f"Failed to open image: {exc}"}), 400
    return _parse_image_response(image, request.form)


def _parse_image_response(image: Image.Image, params: Any):
    try:
        result = parse_image(
            image=image,
            box_threshold=float(params.get("box_threshold", 0.05)),
            iou_threshold=float(params.get("iou_threshold", 0.1)),
            imgsz=int(params.get("imgsz", 640)),
        )
        return jsonify(result)
    except Exception as exc:
        return jsonify({"error": f"Processing failed: {exc}"}), 500


def parse_image(
    image: Image.Image,
    box_threshold: float,
    iou_threshold: float,
    imgsz: int,
) -> dict[str, Any]:
    width, height = image.size
    predictions = get_model().predict(
        source=np.array(image),
        conf=box_threshold,
        iou=iou_threshold,
        imgsz=imgsz,
        verbose=False,
    )
    boxes = predictions[0].boxes
    parsed_content: list[dict[str, Any]] = []
    label_coordinates: dict[str, list[float]] = {}

    for index, box in enumerate(boxes):
        x1, y1, x2, y2 = [float(value) for value in box.xyxy[0].tolist()]
        confidence = float(box.conf[0]) if box.conf is not None else None
        class_id = int(box.cls[0]) if box.cls is not None else None
        bbox = [
            max(0.0, min(1.0, x1 / width)),
            max(0.0, min(1.0, y1 / height)),
            max(0.0, min(1.0, x2 / width)),
            max(0.0, min(1.0, y2 / height)),
        ]
        label_coordinates[str(index)] = bbox
        parsed_content.append(
            {
                "type": "icon",
                "bbox": bbox,
                "content": None,
                "confidence": confidence,
                "class_id": class_id,
            }
        )

    return {
        "annotated_image": None,
        "label_coordinates": label_coordinates,
        "parsed_content": parsed_content,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Lightweight OmniParser-compatible server")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8080)
    args = parser.parse_args()

    get_model()
    app.run(host=args.host, port=args.port)


if __name__ == "__main__":
    main()

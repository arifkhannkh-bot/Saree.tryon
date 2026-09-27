"""
download_models.py

Run this ONCE, with internet access, before starting the API:

    python download_models.py

MediaPipe's Tasks API (the version used by tryon_engine.py) loads its
pose-landmarker model from a local .task file rather than bundling it
inside the pip package. This script fetches the official model file
from Google's public model store and saves it under ./models/.

No customer or saree data is ever sent anywhere by this script or by
the API itself — this only downloads a generic, publicly published
ML model file (not trained on your data, doesn't touch your data).
"""

import os
import urllib.request

MODEL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")
POSE_MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/pose_landmarker/"
    "pose_landmarker_full/float16/latest/pose_landmarker_full.task"
)
POSE_MODEL_PATH = os.path.join(MODEL_DIR, "pose_landmarker_full.task")


def main():
    os.makedirs(MODEL_DIR, exist_ok=True)

    if os.path.exists(POSE_MODEL_PATH):
        print(f"Already present: {POSE_MODEL_PATH}")
        return

    print(f"Downloading pose landmarker model from:\n  {POSE_MODEL_URL}")
    urllib.request.urlretrieve(POSE_MODEL_URL, POSE_MODEL_PATH)
    size_mb = os.path.getsize(POSE_MODEL_PATH) / (1024 * 1024)
    print(f"Saved to {POSE_MODEL_PATH} ({size_mb:.1f} MB)")


if __name__ == "__main__":
    main()

import os
import requests

PERSON_MODEL_PATH = "yolov8n.pt"
DOWNLOAD_URL = "https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8n.pt"

def download_weights():
    if os.path.exists(PERSON_MODEL_PATH):
        print(f"{PERSON_MODEL_PATH} already exists. No download needed.")
        return
    print(f"Downloading {PERSON_MODEL_PATH} from {DOWNLOAD_URL} ...")
    with requests.get(DOWNLOAD_URL, stream=True) as r:
        r.raise_for_status()
        with open(PERSON_MODEL_PATH, "wb") as f:
            for chunk in r.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
    print(f"Downloaded {PERSON_MODEL_PATH} successfully.")

if __name__ == "__main__":
    download_weights()

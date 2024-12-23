import os
from app import app
import requests

def download_model():
    model_path = "./models/polarity_model/model.safetensors"
    model_url = "https://github.com/himanshuyv/ClarityAI/raw/refs/heads/main/models/polarity_model/model.safetensors?download="
    print("Model not found. Downloading...")

    response = requests.get(model_url, stream=True)
    if response.status_code == 200:
        with open(model_path, "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        print("Model downloaded successfully.")
    else:
        print(f"Failed to download the model. HTTP Status Code: {response.status_code}")
        raise Exception("Model download failed.")

if __name__ == "__main__":
    download_model()
    port = int(os.environ.get("PORT", 4000))
    app.run(host="0.0.0.0", port=port)
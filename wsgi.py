import os
from app import app
import requests

def download_model():
    # Define the model's local path and the URL to download from
    model_path = "./models/polarity_model/model.safetensors"
    model_url = "https://github.com/himanshuyv/ClarityAI/raw/refs/heads/main/models/polarity_model/model.safetensors?download="

    # Check if the model already exists
    if not os.path.exists(model_path):
        print("Model not found. Downloading...")
        
        # Create the directory if it doesn't exist
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        
        # Download the model
        response = requests.get(model_url, stream=True)
        if response.status_code == 200:
            with open(model_path, "wb") as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
            print("Model downloaded successfully.")
        else:
            print(f"Failed to download the model. HTTP Status Code: {response.status_code}")
            raise Exception("Model download failed.")
    else:
        print("Model already exists. Skipping download.")

if __name__ == "__main__":
    download_model()
    port = int(os.environ.get("PORT", 4000))
    app.run(host="0.0.0.0", port=port)
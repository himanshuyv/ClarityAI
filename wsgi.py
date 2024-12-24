import os
from app import app
import requests

def download_model():
    model_path = "./models/polarity_model/model.safetensors"
    model_url = "https://0x0.st/8rJW.safetens"

    if (os.path.exists(model_path)):
        print("Model already exists")
    else:
        print("Downloading model")
        response = requests.get(model_url)
        with open(model_path, 'wb') as file:
            file.write(response.content)
        print("Model downloaded")
        
if __name__ == "__main__":
    download_model()
    port = int(os.environ.get("PORT", 4000))
    app.run(host="0.0.0.0", port=port)
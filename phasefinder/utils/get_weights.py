
import os
import requests

model_url = 'https://github.com/bleugreen/phasefinder/raw/main/'

def get_weights(filename="phasefinder-0.1-noattn.pt", quiet=False):
    # Construct the path to save the model weights
    home_dir = os.path.expanduser("~")
    model_dir = os.path.join(home_dir, ".local", "share", "phasefinder")
    if not os.path.exists(model_dir):
        os.makedirs(model_dir, exist_ok=True)
    model_path = os.path.join(model_dir, filename)

    # Check if the model weights already exist
    if not os.path.isfile(model_path):
        print("Downloading model weights...")
        # Download the model weights
        try:
            r = requests.get(model_url+filename, allow_redirects=True)
            if r.status_code == 200:
                with open(model_path, 'wb') as f:
                    f.write(r.content)
                print("Model weights downloaded successfully.")
            else:
                print(f"Failed to download model weights. HTTP Error: {r.status_code}")
        except Exception as e:
            print(f"An error occurred during the download: {e}")
    else:
        if not quiet:
            print("Model weights already exist.")

    return model_path
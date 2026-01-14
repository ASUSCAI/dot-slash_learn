import sys
import os
import requests

# Add current directory to sys.path
sys.path.append(os.getcwd())

# Try to list models from the base URL
base_url = "https://llm.jetstream-cloud.org/llama-4-scout/v1"
# The base URL usually ends with /v1, so we can try /v1/models
models_url = base_url.replace("/chat/completions", "").rstrip("/") + "/models"

print(f"Checking models at: {models_url}")

try:
    response = requests.get(models_url)
    if response.status_code == 200:
        print("Available models:")
        models = response.json()
        for model in models.get('data', []):
            print(f"- {model['id']}")
    else:
        print(f"Failed to list models. Status code: {response.status_code}")
        print(response.text)
except Exception as e:
    print(f"Error listing models: {e}")

# Also try the root URL just in case the path structure is different for listing
root_url = "https://llm.jetstream-cloud.org/v1/models"
print(f"\nChecking models at: {root_url}")
try:
    response = requests.get(root_url)
    if response.status_code == 200:
        print("Available models:")
        models = response.json()
        for model in models.get('data', []):
            print(f"- {model['id']}")
    else:
        print(f"Failed to list models. Status code: {response.status_code}")
        print(response.text)
except Exception as e:
    print(f"Error listing models: {e}")

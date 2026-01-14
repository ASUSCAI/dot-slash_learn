import sys
import os
import time
import requests
import json

# Add current directory to sys.path
sys.path.append(os.getcwd())

API_KEY = os.environ.get("JETSTREAM_API_KEY", "EMPTY")

def make_request(url, model_name):
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {API_KEY}"
    }
    payload = {
        "model": model_name,
        "messages": [{"role": "user", "content": "Hello"}],
        "max_tokens": 10
    }
    try:
        start_time = time.time()
        response = requests.post(url, headers=headers, json=payload, timeout=30)
        duration = time.time() - start_time
        return response.status_code, response.text, duration
    except Exception as e:
        return -1, str(e), 0

def test_gpt_oss_extended():
    print("\n--- Extended Test: gpt-oss-120b ---")
    url = "https://llm.jetstream-cloud.org/gpt-oss-120b/v1/chat/completions"
    for i in range(10):
        status, text, duration = make_request(url, "gpt-oss-120b")
        print(f"Request {i+1}/10: Status {status} ({duration:.2f}s)")
        if status != 200:
            print(f"  Error: {text[:200]}...")
        time.sleep(0.5)

def test_deepseek_urls():
    print("\n--- Testing DeepSeek URL patterns ---")
    model = "deepseek-v3.1"
    urls = [
        f"https://llm.jetstream-cloud.org/{model}/v1/chat/completions",
        f"https://llm.jetstream-cloud.org/DeepSeek-V3.1/v1/chat/completions", # Case sensitive?
        "https://llm.jetstream-cloud.org/v1/chat/completions" # Generic
    ]
    
    for url in urls:
        print(f"Testing URL: {url}")
        status, text, duration = make_request(url, model)
        print(f"  Status {status} ({duration:.2f}s)")
        if status != 200:
            print(f"  Error: {text[:200]}...")
        else:
            print("  SUCCESS!")

def test_generic_endpoint():
    print("\n--- Testing Generic Endpoint for all models ---")
    url = "https://llm.jetstream-cloud.org/v1/chat/completions"
    models = ["llama-4-scout", "gpt-oss-120b", "deepseek-v3.1"]
    
    for model in models:
        print(f"Testing model {model} on generic endpoint")
        status, text, duration = make_request(url, model)
        print(f"  Status {status} ({duration:.2f}s)")
        if status != 200:
            print(f"  Error: {text[:200]}...")

if __name__ == "__main__":
    test_gpt_oss_extended()
    test_deepseek_urls()
    test_generic_endpoint()

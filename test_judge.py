import os, sys, requests
from dotenv import load_dotenv
load_dotenv()

URL = "http://107.108.92.242:4000/v1/chat/completions"
MODEL = sys.argv[1] if len(sys.argv) > 1 else "Qwen-model-name"   # python test_judge.py <model-name>

key = os.getenv("GEMMA_API_KEY", "").replace("Bearer ", "").strip()
print("model:", MODEL, "| API key found:", bool(key))
r = requests.post(URL, timeout=120,
    headers={"Content-Type": "application/json", "Authorization": f"Bearer {key}"},
    json={"model": MODEL, "temperature": 0, "max_tokens": 60,
          "messages": [{"role": "user", "content": 'Reply with only: {"faithful": "yes"} /no_think'}]})
print("HTTP status:", r.status_code)
print(r.text[:800])

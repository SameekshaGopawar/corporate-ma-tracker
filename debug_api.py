import os
import requests
from dotenv import load_dotenv

load_dotenv()

token = os.environ.get("HF_API_TOKEN", "")
headers = {"Authorization": f"Bearer {token}"}
payload = {
    "inputs": "Google acquired Wiz for $32 billion",
    "parameters": {"candidate_labels": ["Acquisition", "Partnership", "Investment", "Other"]}
}

r = requests.post(
    "https://router.huggingface.co/hf-inference/models/facebook/bart-large-mnli",
    headers=headers,
    json=payload
)
print("Status:", r.status_code)
print("Response:", r.json())

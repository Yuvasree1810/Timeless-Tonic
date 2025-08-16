import requests
from typing import List, Dict

class OllamaClient:
    def __init__(self, base_url: str = "http://localhost:11434"):
        self.base_url = base_url
        self.session = requests.Session()
    
    def list_models(self) -> List[Dict]:
        resp = self.session.get(f"{self.base_url}/api/tags")
        resp.raise_for_status()
        return resp.json().get("models", [])
    
    def generate(self, prompt: str, model: str = "llama2", timeout: int = 20) -> str:
        payload = {
            "model": model,
            "prompt": prompt,
            "stream": False
        }
        resp = self.session.post(f"{self.base_url}/api/generate", json=payload, timeout=timeout)
        resp.raise_for_status()
        return resp.json()["response"] 
"""
llm_client.py

Small wrapper for a local Ollama server.
Stores the model tag and base URL, and exposes:
- ping() to sanity-check connectivity
- generate(system, user, max_tokens=..., temperature=..., stream=False)
  to run a chat-style request; returns text when stream is False.

Typical use:
    llm = OllamaClient(model="mistral:instruct")
    llm.ping()
    answer = llm.generate(system, user, max_tokens=256, temperature=0.2)

Notes:
- Uses httpx and will raise httpx.HTTPError on network issues.
- Arguments are validated; keep timeouts modest for the UI.
"""

from src.index_store.faiss_store import FaissStore
from src.embed.embedder import Embedder
from typing import List, Dict, Optional 
import numpy as np
import json
from pathlib import Path
import httpx

# OllamaClient class definition
class OllamaClient:
    def __init__(self, model: str = "mistral:instruct",
                 host: str = "http://localhost:11434",
                 timeout: float = 60.0):
        """
        - Save base URL and model tag.
        - Create an HTTP client with the timeout.
        """
        self.model = model
        self.base_url = host
        self.client = httpx.Client(base_url=self.base_url, timeout=timeout)

    def ping(self) -> None:
        """
        GET "/" to verify the daemon/app is running.
        - Raise a clear exception if unreachable (helps catch 'port in use' vs 'not running').
        """
        r = self.client.get("/api/tags")
        r.raise_for_status()

    def generate(self, system: str, user: str, *,
                 max_tokens: int = 256, temperature: float = 0.2,
                 stream: bool = False) -> str:
        """
        Generate a reply via the local Ollama /api/chat.

        Sends system+user messages to self.model at self.base_url.
        If stream=False, returns response["message"]["content"].
        If stream=True, reads JSONL chunks, concatenates message.content until done.

        Args: system, user, max_tokens (options.num_predict), temperature, stream
        Returns: assistant text (stripped)
        Raises: ValueError, httpx.HTTPError, json.JSONDecodeError
        """
        # validates inputs
        if len(system) <= 0 or len(user) <= 0 or max_tokens <= 0 or temperature < 0 or temperature > 1:
            raise ValueError("Invalid parameter(s).")
        
        # builds the request
        payload = {
            "model": self.model,
            "messages": [
              {"role":"system","content": system},
              {"role":"user","content": user}
            ],
            "options": {
              "num_predict": max_tokens,     
              "temperature": temperature
            },
            "stream": stream
          }
        
        # sends the request
        if stream == False:
            r = self.client.post("/api/chat", json=payload)
            r.raise_for_status()
            data = r.json()
            return data["message"]["content"].strip()

        if stream == True:
            with self.client.stream("POST", "/api/chat", json=payload) as r:
                r.raise_for_status()
                collected = []
                for line in r:
                    if not line:
                        continue
                    obj = json.loads(line)
                    collected.append(obj.get("message", {}).get("content", ""))
                    piece = obj.get("message", {}).get("content", "")
                    if piece:
                        collected.append(piece)
                    if obj.get("done"):
                        break     
                return "".join(collected).strip()

if __name__ == "__main__":
    # constructs client and ping
    llm = OllamaClient(model="mistral:instruct")
    llm.ping()

    # builds a tiny system/user pair
    system = "Answer ONLY from the provided context. Cite pages like [p3] or [p3-4]."
    user = "Question: Where do I submit the proposal?\n\nContext:\n--- [p3] A-0003\nSubmit via the City Portal by February 18, 2025 at 2:00 PM ET.\n\nAnswer with citations."

    # generates and prints
    text = llm.generate(system=system, user=user, max_tokens=256, temperature=0.2, stream=False)
    print("\n=== LLM reply ===\n", text)
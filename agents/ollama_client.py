import json
import httpx
from config import settings

class OllamaError(Exception):
    pass

class OllamaJSONError(OllamaError):
    pass

class OllamaConnectionError(OllamaError):
    pass

def check_health() -> bool:
    """
    Check if the Ollama endpoint is running and the specific model is available.
    """
    try:
        response = httpx.get(f"{settings.OLLAMA_BASE_URL}/api/tags", timeout=settings.OLLAMA_TIMEOUT)
        response.raise_for_status()
        data = response.json()
        models = [model.get("name") for model in data.get("models", [])]
        if settings.OLLAMA_MODEL not in models:
            raise OllamaConnectionError(f"Ollama is running, but model '{settings.OLLAMA_MODEL}' is not available.")
        return True
    except httpx.RequestError as e:
        raise OllamaConnectionError(f"Could not connect to Ollama at {settings.OLLAMA_BASE_URL}. Ensure it is running. Error: {e}")
    except httpx.HTTPStatusError as e:
        raise OllamaConnectionError(f"Ollama health check returned an error. HTTP {e.response.status_code}")

def generate_text(prompt: str) -> str:
    """
    Generate general text using the configured Ollama model.
    """
    payload = {
        "model": settings.OLLAMA_MODEL,
        "prompt": prompt,
        "stream": False
    }
    
    try:
        response = httpx.post(
            f"{settings.OLLAMA_BASE_URL}/api/generate",
            json=payload,
            timeout=settings.OLLAMA_TIMEOUT
        )
        response.raise_for_status()
        data = response.json()
        if "response" not in data:
            raise OllamaError("Missing 'response' field in Ollama generation output.")
        return data["response"]
    except httpx.RequestError as e:
        raise OllamaConnectionError(f"Request to Ollama failed. Error: {e}")
    except httpx.HTTPStatusError as e:
        raise OllamaError(f"Ollama returned HTTP error: {e.response.status_code}")

def generate_json(prompt: str) -> dict:
    """
    Generate JSON output and strictly validate parsing.
    Does not silently invent data.
    """
    payload = {
        "model": settings.OLLAMA_MODEL,
        "prompt": prompt,
        "format": "json",
        "stream": False,
        "options": {
            "temperature": 0,
            "num_predict": 512,
        },
    }
    
    try:
        response = httpx.post(
            f"{settings.OLLAMA_BASE_URL}/api/generate",
            json=payload,
            timeout=settings.OLLAMA_TIMEOUT
        )
        response.raise_for_status()
        data = response.json()
        
        if "response" not in data:
            raise OllamaError("Missing 'response' field in Ollama output.")
        
        raw_text = data["response"]
        
        try:
            parsed = json.loads(raw_text)
            if not isinstance(parsed, dict):
                raise OllamaJSONError("Ollama returned valid JSON, but it is not a JSON object.")
            return parsed
        except json.JSONDecodeError as e:
            raise OllamaJSONError(f"Ollama returned malformed JSON. Parse error: {e}. Raw text: {raw_text}")
            
    except httpx.RequestError as e:
        raise OllamaConnectionError(f"Request to Ollama failed. Error: {e}")
    except httpx.HTTPStatusError as e:
        raise OllamaError(f"Ollama returned HTTP error: {e.response.status_code}")

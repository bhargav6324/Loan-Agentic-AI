import httpx
from config import settings

headers = {"Authorization": f"Bearer {settings.GROQ_API_KEY}", "Content-Type": "application/json"}
payload = {
    "model": "llama3-8b-8192",
    "messages": [
        {"role": "user", "content": "Hello"}
    ]
}
try:
    response = httpx.post("https://api.groq.com/openai/v1/chat/completions", headers=headers, json=payload)
    print("Status:", response.status_code)
    print("Body:", response.text)
except Exception as e:
    print(e)

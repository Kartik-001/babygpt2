import requests

url = "http://localhost:8000/generate"
payload = {
    "prompt": "Hello, world! i am",
    "max_new_tokens": 20
}

response = requests.post(url, json=payload)
data = response.json()
print("Generated text:", data["text"])

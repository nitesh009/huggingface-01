import requests

API_URL = "https://your-huggingface-api.com/predict"
text = {"text": "I love this movie!"}

response = requests.post(API_URL, json=text)
print(response.json())

import requests

API_URL = "http://localhost:8000/predict/"
text = {"text": "This movie was amazing!"}

response = requests.get(API_URL, params=text)
print(response.json())  # Output: [{'label': 'POSITIVE', 'score': 0.9998}]

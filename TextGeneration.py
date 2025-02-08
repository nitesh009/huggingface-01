# Load a Pre-trained Model
# You can load any pre-trained model from Hugging Face with a few lines of code.
# Using a Text Generation Model (GPT-2)

from transformers import pipeline

generator = pipeline("text-generation", model="gpt2")
result = generator("Once upon a time,")
print(result)

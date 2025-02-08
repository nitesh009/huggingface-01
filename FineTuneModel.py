from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer
import torch

#  Load the IMDb dataset
dataset = load_dataset("imdb")

#  Load the BERT tokenizer
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

#  Function to tokenize the dataset
def tokenize_function(examples):
    return tokenizer(examples["text"], padding="max_length", truncation=True)

#  Apply tokenization to the dataset
tokenized_datasets = dataset.map(tokenize_function, batched=True)

#  Remove raw text column
tokenized_datasets = tokenized_datasets.remove_columns(["text"])

#  Rename 'label' column to 'labels' (required by Trainer)
tokenized_datasets = tokenized_datasets.rename_column("label", "labels")

#  Set dataset format for PyTorch
tokenized_datasets.set_format("torch")

# Load pre-trained BERT model for classification (Binary: POSITIVE/NEGATIVE)
model = AutoModelForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=2)

# Define training arguments
training_args = TrainingArguments(
    output_dir="results",
    num_train_epochs=3,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    save_total_limit=2,
    logging_dir="./logs",
    logging_steps=100,
    load_best_model_at_end=True,
)

#Initialize Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["test"],
)

# Start training
trainer.train()

# Save the fine-tuned model
model.save_pretrained("./fine_tuned_bert")
tokenizer.save_pretrained("./fine_tuned_bert")

print("Model training complete and saved!")

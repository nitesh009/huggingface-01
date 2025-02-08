from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer
import torch

# 1️⃣ Load the IMDb dataset
dataset = load_dataset("imdb")

# 2️⃣ Load the BERT tokenizer
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

# 3️⃣ Function to tokenize the dataset
def tokenize_function(examples):
    return tokenizer(examples["text"], padding="max_length", truncation=True)

# 4️⃣ Apply tokenization to the dataset
tokenized_datasets = dataset.map(tokenize_function, batched=True)

# 5️⃣ Remove raw text column
tokenized_datasets = tokenized_datasets.remove_columns(["text"])

# 6️⃣ Rename 'label' column to 'labels' (required by Trainer)
tokenized_datasets = tokenized_datasets.rename_column("label", "labels")

# 7️⃣ Set dataset format for PyTorch
tokenized_datasets.set_format("torch")

# 8️⃣ Load pre-trained BERT model for classification (Binary: POSITIVE/NEGATIVE)
model = AutoModelForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=2)

# 9️⃣ Define training arguments
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

# 🔟 Initialize Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["test"],
)

# 1️⃣1️⃣ Start training
trainer.train()

# 1️⃣2️⃣ Save the fine-tuned model
model.save_pretrained("./fine_tuned_bert")
tokenizer.save_pretrained("./fine_tuned_bert")

print("✅ Model training complete and saved!")

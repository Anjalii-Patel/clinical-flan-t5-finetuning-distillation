from transformers import AutoModelForSeq2SeqLM, Seq2SeqTrainingArguments, Seq2SeqTrainer, AutoTokenizer, DataCollatorForSeq2Seq
import pandas as pd
from datasets import Dataset

# Load your CSV data
csv_file = "dataset_finetune_s.csv"  # Replace with the path to your file
df = pd.read_csv(csv_file)

# Check the first few rows to confirm the structure
# print(df.head())
# Load the pre-trained FLAN-T5 model
model = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-base")
tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-base")

# Convert DataFrame to a Hugging Face Dataset
dataset = Dataset.from_pandas(df)

# Create a function to format the data as input-output pairs
def preprocess_data(examples):
    input_texts = [f"Context: {context} Question: {question}" for context, question in zip(examples["Context"], examples["Question"])]
    target_texts = examples["Answer"]
    return {"input": input_texts, "output": target_texts}

# Apply the preprocessing function to the dataset
tokenized_dataset = dataset.map(preprocess_data, batched=True)

# Display tokenized dataset to confirm preprocessing
# print(tokenized_dataset[0])

def tokenize_function(example):
    input_enc = tokenizer(
        example['input'], truncation=True, padding="max_length", max_length=1024
    )
    output_enc = tokenizer(
        example['output'], truncation=True, padding="max_length", max_length=1024
    )
    return {
        "input_ids": input_enc["input_ids"],
        "attention_mask": input_enc["attention_mask"],
        "labels": output_enc["input_ids"]
    }


# Tokenize the dataset
tokenized_dataset = tokenized_dataset.map(tokenize_function, batched=True)
data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model, padding=True)

# Show tokenized example
# print(tokenized_dataset[0])

# Split the dataset into train and eval sets (80% train, 20% eval)
train_dataset = tokenized_dataset.shuffle(seed=42).select([i for i in range(int(0.8 * len(tokenized_dataset)))])
eval_dataset = tokenized_dataset.shuffle(seed=42).select([i for i in range(int(0.8 * len(tokenized_dataset)), len(tokenized_dataset))])

training_args = Seq2SeqTrainingArguments(
    output_dir="./flan-base-tuned",
    eval_strategy="steps",
    eval_steps=100,
    max_steps=500,
    save_strategy="steps",
    save_total_limit=3,
    learning_rate=3e-5,
    per_device_train_batch_size=2,  # Larger batch size with gradient accumulation
    gradient_accumulation_steps=2,
    # num_train_epochs=3,  # Train for at least 3 epochs
    predict_with_generate=True,
    logging_dir="./logs",
    logging_strategy="steps",
    logging_steps=10,
    load_best_model_at_end=True,
    fp16=True,  # Use FP16 for faster training
)

trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        processing_class=tokenizer,
        data_collator=data_collator
    )

# Start fine-tuning the model
trainer.train()
trainer.save_model("./new_finetuned-base")
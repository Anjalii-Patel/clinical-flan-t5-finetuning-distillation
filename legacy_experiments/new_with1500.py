import pandas as pd
from sklearn.model_selection import train_test_split
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, TrainingArguments, Trainer

# Step 1: Load Dataset
file_path = "Fine Tuning Flan-T5/dataset_finetune_s.csv"  # Replace with your file path
dataset = pd.read_csv(file_path)

train_data, eval_data = train_test_split(dataset, test_size=0.2)

# Step 2: Analyze Dataset
# Add columns for lengths of questions, context, and answers
dataset["Question Length"] = dataset["Question"].apply(len)
dataset["Context Length"] = dataset["Context"].apply(len)
dataset["Answer Length"] = dataset["Answer"].apply(len)

# Step 3: Tokenization
# Initialize the tokenizer
tokenizer = AutoTokenizer.from_pretrained("Fine Tuning Flan-T5/fine_tuned_flan_t5_large")

# Tokenize dataset
def preprocess_for_t5(dataset, max_input_length=512, max_output_length=128):
    inputs = dataset["Question"] + " Context: " + dataset["Context"]
    targets = dataset["Answer"]

    tokenized_inputs = tokenizer(
        inputs.tolist(),
        max_length=max_input_length,
        truncation=True,
        padding="max_length",
        return_tensors="pt",
    )
    tokenized_targets = tokenizer(
        targets.tolist(),
        max_length=max_output_length,
        truncation=True,
        padding="max_length",
        return_tensors="pt",
    )
    return tokenized_inputs, tokenized_targets

# tokenized_inputs, tokenized_targets = preprocess_for_t5(dataset)
# Tokenize validation data as well
tokenized_train_inputs, tokenized_train_targets = preprocess_for_t5(train_data)
tokenized_eval_inputs, tokenized_eval_targets = preprocess_for_t5(eval_data)

# Step 4: Model Loading and Fine-tuning
model = AutoModelForSeq2SeqLM.from_pretrained("Fine Tuning Flan-T5/fine_tuned_flan_t5_large")

# Convert tokenized data into training format
from torch.utils.data import Dataset

class MedicalQADataset(Dataset):
    def __init__(self, inputs, targets):
        self.inputs = inputs
        self.targets = targets

    def __len__(self):
        return len(self.inputs["input_ids"])

    def __getitem__(self, idx):
        return {
            "input_ids": self.inputs["input_ids"][idx],
            "attention_mask": self.inputs["attention_mask"][idx],
            "labels": self.targets["input_ids"][idx],
        }

# train_dataset = MedicalQADataset(tokenized_inputs, tokenized_targets)
# Create datasets for train and eval
train_dataset = MedicalQADataset(tokenized_train_inputs, tokenized_train_targets)
eval_dataset = MedicalQADataset(tokenized_eval_inputs, tokenized_eval_targets)

# Training Arguments
training_args = TrainingArguments(
    output_dir="./results",
    eval_strategy="steps",
    save_strategy="steps",
    logging_dir="./logs",
    per_device_train_batch_size=1,
    num_train_epochs=2,
    learning_rate=5e-5,
    save_steps=500,
    save_total_limit=4,
    weight_decay=0.01,
    logging_steps=50,
    warmup_steps=200,
)

# Initialize Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
)
# Train Model
trainer.train()

# Save Model
model.save_pretrained("./fine_tuned_flan_t5_large")
tokenizer.save_pretrained("./fine_tuned_flan_t5_large")

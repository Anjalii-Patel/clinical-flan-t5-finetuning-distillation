# import torch
# from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, DataCollatorWithPadding
# from datasets import load_dataset
# from torch.utils.data import DataLoader
# import torch.nn.functional as F
# from tqdm import tqdm

# # Load teacher and student models and tokenizers
# print("Loading models and tokenizers...")
# teacher_model = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-xl")
# student_model = AutoModelForSeq2SeqLM.from_pretrained("Fine Tuning Flan-T5/fine_tuned_flan_t5_large")
# teacher_tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-xl")
# student_tokenizer = AutoTokenizer.from_pretrained("Fine Tuning Flan-T5/fine_tuned_flan_t5_large")

# # Load dataset
# print("Loading dataset...")
# dataset = load_dataset('csv', data_files={'test': 'Fine Tuning Flan-T5/dataset_finetune_s.csv'}, delimiter=',')

# # Preprocess data for T5
# def preprocess_data(examples):
#     questions = examples['Question']
#     contexts = examples['Context']
#     inputs = [f"Question: {q} Context: {c}" for q, c in zip(questions, contexts)]

#     teacher_inputs = teacher_tokenizer(inputs, padding="max_length", truncation=True, max_length=1024, return_tensors="pt")
#     student_inputs = student_tokenizer(inputs, padding="max_length", truncation=True, max_length=1024, return_tensors="pt")
#     labels = teacher_tokenizer(examples['Answer'], padding="max_length", truncation=True, max_length=1024, return_tensors="pt")

#     return {
#         'input_ids': teacher_inputs['input_ids'],
#         'attention_mask': teacher_inputs['attention_mask'],
#         'labels': labels['input_ids']
#     }

# # Apply preprocessing to the dataset
# print("Preprocessing dataset...")
# test_dataset = dataset['test'].map(preprocess_data, batched=True, remove_columns=dataset['test'].column_names)

# # Define DataLoader with a DataCollator
# print("Preparing dataloader...")
# data_collator = DataCollatorWithPadding(tokenizer=teacher_tokenizer, return_tensors="pt")
# test_dataloader = DataLoader(test_dataset, batch_size=1, collate_fn=data_collator)

# # Move models to evaluation mode
# teacher_model.eval()
# student_model.eval()

# # Define distillation loss function
# def distillation_loss(student_logits, teacher_logits, temperature=2.0, alpha=0.5):
#     teacher_probs = F.softmax(teacher_logits / temperature, dim=-1)
#     student_probs = F.log_softmax(student_logits / temperature, dim=-1)
#     kl_loss = F.kl_div(student_probs, teacher_probs, reduction="batchmean") * (temperature ** 2)
#     ce_loss = F.cross_entropy(student_logits, torch.argmax(teacher_probs, dim=-1), reduction="mean")
#     return alpha * kl_loss + (1 - alpha) * ce_loss

# # Training loop with 3 epochs
# num_epochs = 3
# print(f"Starting distillation for {num_epochs} epochs...")

# with torch.no_grad():
#     for epoch in range(num_epochs):
#         print(f"\nEpoch {epoch + 1}/{num_epochs}")
#         total_loss = 0
#         step = 0

#         for batch in tqdm(test_dataloader, desc=f"Epoch {epoch + 1}/{num_epochs} - Distillation"):
#             input_ids = batch['input_ids'].to(student_model.device)
#             attention_mask = batch['attention_mask'].to(student_model.device)
#             labels = batch['labels'].to(student_model.device)

#             # Shift labels to create decoder_input_ids
#             decoder_input_ids = labels[:, :-1].contiguous()
#             labels = labels[:, 1:].contiguous()

#             # Get logits from teacher and student models
#             teacher_outputs = teacher_model(input_ids=input_ids, attention_mask=attention_mask, decoder_input_ids=decoder_input_ids)
#             student_outputs = student_model(input_ids=input_ids, attention_mask=attention_mask, decoder_input_ids=decoder_input_ids)

#             # Compute distillation loss
#             loss = distillation_loss(student_outputs.logits, teacher_outputs.logits)
#             total_loss += loss.item()
#             step += 1

#         print(f"Epoch {epoch + 1} - Average Loss: {total_loss / step:.4f}")

# # Save the distilled model
# print("Saving distilled model...")
# student_model.save_pretrained("distilled_flant5_largexl")
# student_tokenizer.save_pretrained("distilled_flant5_largexl")
# print("Model saved successfully!")


import torch
from torch.utils.data import DataLoader, random_split
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, Seq2SeqTrainer, Seq2SeqTrainingArguments
from datasets import Dataset
import pandas as pd

# Define paths and model identifiers
teacher_model_name = "google/flan-t5-xl"
student_model_path = "Fine Tuning Flan-T5/fine_tuned_flan_t5_large"  # Path to your fine-tuned FLAN-T5-Large
custom_dataset_path = "Fine Tuning Flan-T5/dataset_finetune_s.csv"  # Path to your local dataset

# Load tokenizers
tokenizer = AutoTokenizer.from_pretrained(teacher_model_name)
student_tokenizer = AutoTokenizer.from_pretrained(student_model_path)

# Load teacher and student models
teacher_model = AutoModelForSeq2SeqLM.from_pretrained(teacher_model_name)
student_model = AutoModelForSeq2SeqLM.from_pretrained(student_model_path)

# Load custom dataset and split into train/validation
data = pd.read_csv(custom_dataset_path)
dataset = Dataset.from_pandas(data)
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

# Tokenize dataset
def preprocess_function(batch):
    inputs = tokenizer(batch["Question"], batch["Context"], truncation=True, padding="max_length", max_length=512, return_tensors="pt")
    labels = tokenizer(batch["Answer"], truncation=True, padding="max_length", max_length=512, return_tensors="pt").input_ids
    inputs["labels"] = labels
    return inputs

# Convert Subsets back to Dataset
train_dataset = Dataset.from_pandas(data.iloc[train_dataset.indices])
val_dataset = Dataset.from_pandas(data.iloc[val_dataset.indices])

# Apply the preprocess function
train_dataset = train_dataset.map(preprocess_function, batched=True)
val_dataset = val_dataset.map(preprocess_function, batched=True)

# Create DataLoaders
train_dataloader = DataLoader(train_dataset, shuffle=True, batch_size=1)
val_dataloader = DataLoader(val_dataset, batch_size=1)

# Define distillation loss function
def distillation_loss(student_logits, teacher_logits, labels, temperature=2.0):
    kl_div = torch.nn.KLDivLoss(reduction="batchmean")
    teacher_probs = torch.nn.functional.log_softmax(teacher_logits / temperature, dim=-1)
    student_probs = torch.nn.functional.softmax(student_logits / temperature, dim=-1)
    loss = kl_div(student_probs, teacher_probs) * (temperature ** 2)
    return loss

# Training setup
training_args = Seq2SeqTrainingArguments(
    output_dir="./distilled_model",
    eval_strategy="epoch",
    learning_rate=5e-5,
    per_device_train_batch_size=1,
    per_device_eval_batch_size=1,
    num_train_epochs=3,
    weight_decay=0.01,
    save_total_limit=2,
    save_steps=500,
    logging_steps=100,
    predict_with_generate=True,
)

# Trainer wrapper for distillation
class DistillationTrainer(Seq2SeqTrainer):
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        labels = inputs.pop("labels")
        student_outputs = model(**inputs)  # This will pass inputs to the encoder

        with torch.no_grad():
            teacher_outputs = teacher_model(**inputs)  # Similarly for teacher model

        student_logits = student_outputs.logits
        teacher_logits = teacher_outputs.logits

        # Calculate loss here (e.g., distillation loss)
        loss = distillation_loss(student_logits, teacher_logits, labels)

        # Return loss and outputs if needed
        return (loss, student_outputs) if return_outputs else loss


# Initialize trainer
trainer = DistillationTrainer(
    model=student_model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    tokenizer=student_tokenizer,  # Use student tokenizer
)

# Train the model
trainer.train()

# Save the distilled model and tokenizer
trainer.save_model("./distilled_model")
student_tokenizer.save_pretrained("./distilled_model")  # Save the student tokenizer

import pandas as pd
from sklearn.model_selection import train_test_split
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, Seq2SeqTrainingArguments, Seq2SeqTrainer
from datasets import Dataset
import torch


# Load teacher model and tokenizer
teacher_model_name = "google/flan-t5-xl"
tokenizer = AutoTokenizer.from_pretrained(teacher_model_name)
teacher_model = AutoModelForSeq2SeqLM.from_pretrained(teacher_model_name)

# Load the dataset
print("Loading dataset...")
train_dataset = Dataset.from_pandas(pd.read_csv("Fine Tuning Flan-T5/train_finetune.csv"))
val_dataset = Dataset.from_pandas(pd.read_csv("Fine Tuning Flan-T5/val_finetune.csv"))
print("Dataset loaded successfully.")

# Tokenize the dataset
print("Tokenizing dataset...")
def tokenize_function(example):
    inputs = tokenizer(example["input"], truncation=True, max_length=512, padding="max_length")
    outputs = tokenizer(example["target"], truncation=True, max_length=512, padding="max_length")

    # Add the labels (shifted input tokens for language modeling)
    inputs["labels"] = outputs["input_ids"]

    # Ensure labels are not padded, replace padding token with -100 (ignored by loss function)
    inputs["labels"] = [label if label != tokenizer.pad_token_id else -100 for label in inputs["labels"]]

    return inputs

train_tokenized = train_dataset.map(tokenize_function, batched=True)
val_tokenized = val_dataset.map(tokenize_function, batched=True)

train_tokenized.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])
val_tokenized.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])
print("Dataset tokenization completed.")

class DistillationTrainer(Seq2SeqTrainer):
    def __init__(self, *args, teacher_model=None, temperature=2.0, **kwargs):
        super().__init__(*args, **kwargs)
        self.teacher_model = teacher_model
        self.temperature = temperature
        self.teacher_model.eval()  # Ensure the teacher model is in evaluation mode

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        # Forward pass through teacher model
        with torch.no_grad():
            teacher_outputs = self.teacher_model(**inputs)
            teacher_logits = teacher_outputs.logits

        # Forward pass through student model
        student_outputs = model(**inputs)
        student_logits = student_outputs.logits

        # Compute the distillation loss (KL Divergence)
        loss_fct = torch.nn.KLDivLoss(reduction="batchmean")
        loss = loss_fct(
            torch.nn.functional.log_softmax(student_logits / self.temperature, dim=-1),
            torch.nn.functional.softmax(teacher_logits / self.temperature, dim=-1),
        )

        return (loss, student_outputs) if return_outputs else loss


# Initialize the teacher and student models
teacher_model = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-xl")
student_model_name = "Fine Tuning Flan-T5/fine_tuned_flan_t5_large"
student_model = AutoModelForSeq2SeqLM.from_pretrained(student_model_name)
student_tokenizer = AutoTokenizer.from_pretrained(student_model_name)

# Training arguments for the student model
student_training_args = Seq2SeqTrainingArguments(
    output_dir="./student_model",
    eval_strategy="epoch",
    eval_steps=100,
    save_steps=100,
    logging_dir="./logs",
    logging_steps=100,
    learning_rate=2e-5,
    gradient_accumulation_steps=2,
    per_device_train_batch_size=1,  # Increased for better gradient updates
    per_device_eval_batch_size=1,
    num_train_epochs=3,
    save_total_limit=4,
    fp16=False,  # Use this only if GPU is available
    report_to="none",  # Suppress logging to unwanted sources
)

# Initialize the distillation trainer
print("Initializing trainer for student model (Distillation)...")
student_trainer = DistillationTrainer(
    model=student_model,
    args=student_training_args,
    train_dataset=train_tokenized,
    eval_dataset=val_tokenized,
    processing_class=student_tokenizer,
    teacher_model=teacher_model,
    temperature=2.0,  # Adjustable temperature for distillation
)

# Start the distillation process
print("Starting student model distillation...")
student_trainer.train(resume_from_checkpoint="student_model/checkpoint-100")
print("Student model distillation completed.")

# Save the fine-tuned student model
student_model.save_pretrained("./XL_LARGE")
student_tokenizer.save_pretrained("./XL_LARGE")
print("Student model saved successfully.")

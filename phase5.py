from transformers import AutoModelForQuestionAnswering, AutoTokenizer, Trainer, TrainingArguments
from peft import LoraConfig, get_peft_model
import torch
from datasets import load_dataset

# Load pre-trained model and tokenizer
model_name = "bert-base-uncased"
model = AutoModelForQuestionAnswering.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

def fine_tune_model(dataset_path="data/processed/preprocessed_jobs.csv"):
    """
    Fine-tune a model using LoRA on a task-specific dataset.
    
    Args:
        dataset_path (str): Path to your dataset (CSV with job descriptions)
    """
    # Define LoRA configuration
    lora_config = LoraConfig(
        r=16,  # Rank of low-rank matrices
        lora_alpha=32,  # Scaling factor
        lora_dropout=0.1,  # Dropout for regularization
        target_modules=["query", "key", "value"]  # Attention layers to apply LoRA
    )

    # Apply LoRA to the model
    peft_model = get_peft_model(model, lora_config)

    # Load and preprocess your dataset (example assumes a CSV with 'description' column)
    # Replace with your actual dataset preparation logic
    dataset = load_dataset('csv', data_files=dataset_path)
    
    # Tokenize dataset (example preprocessing; adjust based on your data)
    def preprocess_function(examples):
        # For demonstration, assume 'description' is context; add your own question/answer pairs
        questions = ["What skills are needed?"] * len(examples['description'])
        return tokenizer(
            questions,
            examples['description'],
            truncation=True,
            padding="max_length",
            max_length=512,
            return_tensors="pt"
        )

    tokenized_dataset = dataset.map(preprocess_function, batched=True)

    # Define training arguments
    training_args = TrainingArguments(
        output_dir="./results",
        evaluation_strategy="epoch",
        learning_rate=2e-5,  # Small learning rate for LoRA
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        num_train_epochs=3,
        weight_decay=0.01,
        logging_dir='./logs',
        logging_steps=10,
    )

    # Initialize Trainer
    trainer = Trainer(
        model=peft_model,
        args=training_args,
        train_dataset=tokenized_dataset['train'],  # Adjust if you split train/eval
        # eval_dataset=tokenized_dataset['validation'],  # Uncomment and prepare validation set if available
    )

    # Fine-tune the model
    trainer.train()

    # Save the fine-tuned model
    peft_model.save_pretrained("./fine_tuned_model")
    tokenizer.save_pretrained("./fine_tuned_model")
    print("Fine-tuned model saved to ./fine_tuned_model")

if __name__ == "__main__":
    fine_tune_model()
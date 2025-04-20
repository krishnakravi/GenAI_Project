from transformers import AutoModelForSequenceClassification, AutoTokenizer, Trainer, TrainingArguments
from peft import LoraConfig, get_peft_model
from datasets import load_dataset
import pandas as pd
import os

def clean_dataset(input_path="data/processed_exercises.csv", output_path="data/processed_exercises_clean.csv"):
    """
    Clean the dataset by handling missing values and ensuring 'Desc' is string type.
    
    Args:
        input_path (str): Path to the input CSV.
        output_path (str): Path to save the cleaned CSV.
    
    Returns:
        bool: True if cleaning was successful, False otherwise.
    """
    try:
        # Load the CSV
        df = pd.read_csv(input_path)
        
        # Log initial state
        print("Before cleaning:")
        print(f"Missing 'Desc': {df['Desc'].isnull().sum()}")
        print(f"Missing 'Type': {df['Type'].isnull().sum()}")
        print(f"Types in 'Desc':\n{df['Desc'].apply(type).value_counts()}")
        
        # Drop rows with missing 'Desc' or 'Type'
        df = df.dropna(subset=['Desc', 'Type'])
        
        # Convert 'Desc' to string
        df['Desc'] = df['Desc'].astype(str)
        
        # Ensure 'Desc' is not empty
        df = df[df['Desc'].str.strip() != '']
        
        # Log final state
        print("\nAfter cleaning:")
        print(f"Missing 'Desc': {df['Desc'].isnull().sum()}")
        print(f"Missing 'Type': {df['Type'].isnull().sum()}")
        print(f"Types in 'Desc':\n{df['Desc'].apply(type).value_counts()}")
        
        # Save cleaned dataset
        df.to_csv(output_path, index=False)
        print(f"Cleaned dataset saved to {output_path}")
        return True
    except Exception as e:
        print(f"Error cleaning dataset: {str(e)}")
        return False

def fine_tune_model(dataset_path="data/processed_exercises_clean.csv"):
    """
    Fine-tune a BERT model for sequence classification using LoRA.
    
    Args:
        dataset_path (str): Path to the cleaned dataset CSV.
    """
    # Ensure dataset exists
    if not os.path.exists(dataset_path):
        print(f"Dataset not found at {dataset_path}. Attempting to clean original dataset...")
        success = clean_dataset("data/processed_exercises.csv", dataset_path)
        if not success:
            raise FileNotFoundError("Could not prepare dataset.")

    # Load dataset
    dataset = load_dataset('csv', data_files=dataset_path)
    
    # Get unique labels from 'Type' column
    unique_labels = dataset['train'].unique('Type')
    if not unique_labels:
        raise ValueError("No valid labels found in 'Type' column.")
    
    num_labels = len(unique_labels)
    label2id = {label: idx for idx, label in enumerate(unique_labels)}
    id2label = {idx: label for label, idx in label2id.items()}
    
    # Load model and tokenizer
    model_name = "bert-base-uncased"
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=num_labels,
        id2label=id2label,
        label2id=label2id
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    # Preprocess function with debugging
    def preprocess_function(examples):
        descs = examples['Desc']
        # Ensure all descriptions are strings
        descs = [str(desc) if desc is not None else "No description" for desc in descs]
        # Tokenize
        encodings = tokenizer(
            descs,
            truncation=True,
            padding="max_length",
            max_length=512,
            return_tensors="pt"
        )
        # Convert labels to indices
        encodings['labels'] = [label2id[label] for label in examples['Type']]
        return encodings
    
    # Tokenize dataset
    try:
        tokenized_dataset = dataset.map(preprocess_function, batched=True)
    except Exception as e:
        print(f"Error during tokenization: {str(e)}")
        raise
    
    # LoRA configuration
    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.1,
        target_modules=["query", "key", "value"]
    )
    
    # Apply LoRA to model
    peft_model = get_peft_model(model, lora_config)
    
    # Training arguments
    training_args = TrainingArguments(
        output_dir="./results",
        learning_rate=2e-5,
        per_device_train_batch_size=8,
        num_train_epochs=3,
        weight_decay=0.01,
        logging_dir='./logs',
        logging_steps=10,
        save_strategy="no",
        report_to="none"
    )
    
    # Initialize Trainer
    trainer = Trainer(
        model=peft_model,
        args=training_args,
        train_dataset=tokenized_dataset['train'],
    )
    
    # Fine-tune the model
    try:
        trainer.train()
    except Exception as e:
        print(f"Error during training: {str(e)}")
        raise
    
    # Save the model
    peft_model.save_pretrained("./fine_tuned_model")
    tokenizer.save_pretrained("./fine_tuned_model")
    print("Model saved to ./fine_tuned_model")

if __name__ == "__main__":
    fine_tune_model()
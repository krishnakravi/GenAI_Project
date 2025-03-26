import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from gensim.models import Word2Vec
from nltk.tokenize import word_tokenize
from sklearn.manifold import TSNE
import tensorflow as tf
from transformers import BertTokenizer, TFBertModel

def train_word2vec(data_file, output_dir):
    """Train Word2Vec model on exercise descriptions"""
    print(f"Loading data from {data_file}...")
    df = pd.read_csv(data_file)
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Process text and train Word2Vec model
    print("Tokenizing sentences...")
    
    # Change 'processed_text' to 'processed_description'
    tokenized_sentences = [word_tokenize(text.lower()) for text in df['processed_description'] if isinstance(text, str)]
    
    print(f"Training Word2Vec model on {len(tokenized_sentences)} sentences...")
    model = Word2Vec(sentences=tokenized_sentences, vector_size=100, window=5, min_count=1, workers=4)
    
    # Save model
    model_path = os.path.join(output_dir, "word2vec_model")
    model.save(model_path)
    print(f"Word2Vec model saved to {model_path}")
    
    return model

def generate_bert_embeddings(data_file, output_dir):
    """Generate BERT embeddings for exercise descriptions"""
    print(f"Loading data from {data_file}...")
    df = pd.read_csv(data_file)
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Load pre-trained BERT model and tokenizer
    print("Loading BERT model and tokenizer...")
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    model = TFBertModel.from_pretrained('bert-base-uncased')
    
    # Process descriptions and generate embeddings
    print("Generating BERT embeddings...")
    embeddings = []
    
    # Process a subset of descriptions for demonstration
    # Change 'processed_text' to 'processed_description'
    for i, text in enumerate(df['processed_description'].head(100)):
        if not isinstance(text, str) or not text:
            embeddings.append(np.zeros(768))
            continue
            
        inputs = tokenizer(text, return_tensors="tf", padding=True, truncation=True, max_length=128)
        outputs = model(inputs)
        
        # Use the [CLS] token embedding as the sentence embedding
        embedding = outputs.last_hidden_state[:, 0, :].numpy()[0]
        embeddings.append(embedding)
        
        if (i+1) % 10 == 0:
            print(f"Processed {i+1} descriptions...")
    
    # Save embeddings
    embeddings_array = np.array(embeddings)
    embeddings_path = os.path.join(output_dir, "bert_embeddings.npy")
    np.save(embeddings_path, embeddings_array)
    print(f"BERT embeddings saved to {embeddings_path}")
    
    return embeddings_array
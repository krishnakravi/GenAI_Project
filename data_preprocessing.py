import pandas as pd
import numpy as np
import re
import string
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import nltk

# Download required NLTK resources
nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)

def load_exercise_data(file_path='megaGymDataset.csv'):
    """Load and prepare the exercise dataset"""
    # Read CSV file
    try:
        df = pd.read_csv(file_path)
        print(f"Successfully loaded exercise data with {df.shape[0]} exercises.")
        return df
    except Exception as e:
        print(f"Error loading exercise data: {e}")
        return None

def clean_text(text):
    """Clean and preprocess text data"""
    if isinstance(text, float) and np.isnan(text):
        return ""
    
    # Convert to string if not already
    text = str(text)
    
    # Remove special characters and numbers
    text = re.sub(r'[^\w\s]', ' ', text)
    
    # Convert to lowercase
    text = text.lower()
    
    # Tokenize
    tokens = word_tokenize(text)
    
    # Remove stopwords
    stop_words = set(stopwords.words('english'))
    tokens = [word for word in tokens if word not in stop_words]
    
    # Join tokens back to text
    cleaned_text = ' '.join(tokens)
    
    return cleaned_text

def preprocess_exercise_data(df):
    """Preprocess the exercise dataset"""
    if df is None:
        return None
    
    # Create a copy to avoid modifying the original
    processed_df = df.copy()
    
    # Clean description text if it exists
    if 'description' in processed_df.columns:
        processed_df['processed_description'] = processed_df['description'].apply(clean_text)
    elif df.shape[1] >= 3:  # Assume column 2 might contain descriptions
        processed_df['processed_description'] = processed_df.iloc[:, 2].apply(clean_text)
    
    # Extract features
    processed_df['exercise_type'] = processed_df.iloc[:, 3] if df.shape[1] > 3 else ""  # Type column
    processed_df['muscle_group'] = processed_df.iloc[:, 4] if df.shape[1] > 4 else ""   # Muscle group column
    processed_df['equipment'] = processed_df.iloc[:, 5] if df.shape[1] > 5 else ""      # Equipment column
    processed_df['difficulty'] = processed_df.iloc[:, 6] if df.shape[1] > 6 else ""     # Difficulty column
    
    print("Exercise data preprocessing completed.")
    return processed_df

def save_processed_data(df, output_path='data/processed_exercises.csv'):
    """Save the processed data to CSV"""
    try:
        df.to_csv(output_path, index=False)
        print(f"Processed data saved to {output_path}")
    except Exception as e:
        print(f"Error saving processed data: {e}")

def main():
    # Load the exercise data
    raw_df = load_exercise_data()
    
    # Preprocess the data
    processed_df = preprocess_exercise_data(raw_df)
    
    # Save the processed data
    if processed_df is not None:
        save_processed_data(processed_df)

if __name__ == '__main__':
    main()
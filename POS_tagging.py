import pandas as pd
import nltk
from nltk import pos_tag
from nltk.tokenize import word_tokenize
from collections import Counter

# Download required NLTK resources
nltk.download('punkt', quiet=True)
nltk.download('averaged_perceptron_tagger', quiet=True)

def extract_fitness_terms(text):
    """Extract fitness-related terms from text based on POS tagging"""
    if not isinstance(text, str) or not text:
        return []
    
    # Tokenize text
    tokens = word_tokenize(text.lower())
    
    # Perform POS tagging
    tagged_tokens = pos_tag(tokens)
    
    # Extract nouns (NN, NNS), verbs (VB, VBD, VBG), and adjectives (JJ)
    fitness_terms = [word for word, tag in tagged_tokens if 
                     (tag.startswith('NN') or tag.startswith('VB') or tag.startswith('JJ')) 
                     and len(word) > 2]
    
    return fitness_terms

def analyze_exercise_descriptions(df, description_col='processed_description'):
    """Analyze exercise descriptions to extract key fitness terms"""
    if df is None or description_col not in df.columns:
        print(f"Column '{description_col}' not found in the dataframe")
        return None
    
    # Extract fitness terms from all descriptions
    all_fitness_terms = []
    
    for description in df[description_col]:
        terms = extract_fitness_terms(description)
        all_fitness_terms.extend(terms)
    
    # Count frequencies
    term_frequencies = Counter(all_fitness_terms)
    
    # Convert to DataFrame
    term_df = pd.DataFrame(term_frequencies.items(), columns=['Term', 'Frequency'])
    term_df = term_df.sort_values('Frequency', ascending=False).reset_index(drop=True)
    
    print(f"Extracted {len(term_df)} unique fitness terms from exercise descriptions")
    
    return term_df

def tag_exercises_with_terms(df, term_df, description_col='processed_description'):
    """Tag each exercise with its key fitness terms"""
    if df is None or description_col not in df.columns:
        print(f"Column '{description_col}' not found in the dataframe")
        return None
    
    # Get top fitness terms (those with higher frequency)
    top_terms = set(term_df.head(100)['Term'])
    
    # Function to find top terms in a description
    def find_top_terms(description):
        if not isinstance(description, str) or not description:
            return []
        
        tokens = word_tokenize(description.lower())
        return [term for term in tokens if term in top_terms]
    
    # Add key terms to each exercise
    df['key_fitness_terms'] = df[description_col].apply(find_top_terms)
    
    print("Exercises tagged with key fitness terms")
    return df

def main():
    # Load processed exercise data
    try:
        processed_df = pd.read_csv('data/processed_exercises.csv')
        
        # Extract and analyze fitness terms
        term_df = analyze_exercise_descriptions(processed_df)
        
        # Save fitness terms
        if term_df is not None:
            term_df.to_csv('data/fitness_terms.csv', index=False)
            print("Fitness terms saved to data/fitness_terms.csv")
        
        # Tag exercises with terms
        tagged_df = tag_exercises_with_terms(processed_df, term_df)
        
        # Save tagged exercises
        if tagged_df is not None:
            tagged_df.to_csv('data/tagged_exercises.csv', index=False)
            print("Tagged exercises saved to data/tagged_exercises.csv")
            
    except Exception as e:
        print(f"Error in POS tagging analysis: {e}")

if __name__ == '__main__':
    main()
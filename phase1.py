import pandas as pd
import numpy as np
import os
from data_preprocessing import load_exercise_data, preprocess_exercise_data, save_processed_data
from POS_tagging import analyze_exercise_descriptions, tag_exercises_with_terms
from word_embeddings import train_word2vec, generate_bert_embeddings

def run_phase1():
    """Execute all components of Phase 1: Basic NLP Analysis of Exercise Data"""
    print("Starting Phase 1: Basic NLP Analysis of Exercise Data")
    
    # Create data directory if it doesn't exist
    os.makedirs('data', exist_ok=True)
    os.makedirs('reports', exist_ok=True)
    
    # Step 1: Load and preprocess exercise data
    print("\nStep 1: Loading and preprocessing exercise data...")
    raw_df = load_exercise_data()
    processed_df = preprocess_exercise_data(raw_df)
    
    if processed_df is not None:
        save_processed_data(processed_df)
        
        # Step 2: POS tagging and term extraction
        print("\nStep 2: Performing POS tagging and term extraction...")
        term_df = analyze_exercise_descriptions(processed_df)
        
        if term_df is not None:
            term_df.to_csv('data/fitness_terms.csv', index=False)
            print("Fitness terms saved to data/fitness_terms.csv")
            
            # Tag exercises with terms
            tagged_df = tag_exercises_with_terms(processed_df, term_df)
            
            if tagged_df is not None:
                tagged_df.to_csv('data/tagged_exercises.csv', index=False)
                print("Tagged exercises saved to data/tagged_exercises.csv")
                
               # Step 3: Creating word embeddings for exercises...
                print("\nStep 3: Creating word embeddings for exercises...")
                model = train_word2vec("data/tagged_exercises.csv", "data/embeddings")  # Capture the model
                generate_bert_embeddings("data/tagged_exercises.csv", "data/embeddings")

                if model is not None:
                    print("Word2Vec model created successfully!")
                                
                # Generate summary report
                generate_phase1_report(raw_df, processed_df, term_df)
    
    print("\nPhase 1 completed!")

def generate_phase1_report(raw_df, processed_df, term_df, output_path='reports/phase1_summary.txt'):
    """Generate a summary report for Phase 1"""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    with open(output_path, 'w') as f:
        f.write("Phase 1: Basic NLP Analysis of Exercise Data - Summary Report\n")
        f.write("="*70 + "\n\n")
        
        # Dataset summary
        f.write("Dataset Summary:\n")
        f.write("-"*70 + "\n")
        f.write(f"Total number of exercises: {raw_df.shape[0]}\n")
        
        # Exercise types summary
        if 'exercise_type' in processed_df.columns:
            exercise_types = processed_df['exercise_type'].value_counts()
            f.write("\nExercise Types Distribution:\n")
            for ex_type, count in exercise_types.items():
                f.write(f"- {ex_type}: {count} exercises\n")
        
        # Muscle groups summary
        if 'muscle_group' in processed_df.columns:
            muscle_groups = processed_df['muscle_group'].value_counts()
            f.write("\nMuscle Groups Distribution:\n")
            for muscle, count in muscle_groups.items():
                f.write(f"- {muscle}: {count} exercises\n")
        
        # Equipment summary
        if 'equipment' in processed_df.columns:
            equipment = processed_df['equipment'].value_counts()
            f.write("\nEquipment Distribution:\n")
            for equip, count in equipment.items():
                f.write(f"- {equip}: {count} exercises\n")
        
        # Difficulty levels
        if 'difficulty' in processed_df.columns:
            difficulty = processed_df['difficulty'].value_counts()
            f.write("\nDifficulty Levels Distribution:\n")
            for diff, count in difficulty.items():
                f.write(f"- {diff}: {count} exercises\n")
        
        # Top fitness terms
        if term_df is not None:
            f.write("\nTop 20 Fitness Terms:\n")
            for i, (term, freq) in enumerate(term_df.head(20).values):
                f.write(f"{i+1}. {term} (frequency: {freq})\n")
    
    print(f"Phase 1 summary report saved to {output_path}")

if __name__ == '__main__':
    run_phase1()
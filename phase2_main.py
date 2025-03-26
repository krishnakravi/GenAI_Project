import os
from phase2 import implement_chain_of_thought
from tree_of_thought import implement_tree_of_thought
from graph_of_thought import implement_graph_of_thought
from embedding_visualizer import visualize_embeddings_comparison  # Add this import

def run_phase2():
    """
    Run all steps for Phase 2
    """
    print("Phase 2: Implementing Exercise Recommendation Frameworks")
    
    # Create directories
    os.makedirs("data/cot", exist_ok=True)
    os.makedirs("data/tot", exist_ok=True)
    os.makedirs("data/got", exist_ok=True)
    os.makedirs("data/embeddings/visualizations", exist_ok=True)  # Add this line
    os.makedirs("reports", exist_ok=True)
    
    # Step 0: Visualize embeddings from Phase 1
    print("\nStep 0: Visualizing embeddings from Phase 1...")
    embedding_viz_dir = visualize_embeddings_comparison(
        "data/embeddings/word2vec_model",
        "data/embeddings/bert_embeddings.npy",
        "data/tagged_exercises.csv",
        "data/embeddings/visualizations"
    )
    
    # Step 1: Implement Chain of Thought reasoning
    print("\nStep 1: Implementing Chain of Thought reasoning...")
    cot_results = implement_chain_of_thought("megaGymDataset.csv", "data/cot")
    
    # Step 2: Implement Tree of Thought reasoning
    print("\nStep 2: Implementing Tree of Thought reasoning...")
    tot_results = implement_tree_of_thought("megaGymDataset.csv", "data/tot")
    
    # Step 3: Implement Graph of Thought reasoning
    print("\nStep 3: Implementing Graph of Thought reasoning...")
    got_results = implement_graph_of_thought("megaGymDataset.csv", "data/got")
    
    # Generate report
    generate_report(cot_results, tot_results, got_results, embedding_viz_dir)  # Update this line
    
    print("\nPhase 2 completed successfully!")
    print("Check the 'data' directory for outputs and 'reports' for the summary report.")

def generate_report(cot_results, tot_results, got_results, embedding_viz_dir=None):  # Update the function signature
    """
    Generate a summary report for Phase 2
    """
    # Fix: Add UTF-8 encoding to support all Unicode characters
    with open("reports/phase2_report.md", "w", encoding="utf-8") as f:
        f.write("# Phase 2: Exercise Recommendation Frameworks Report\n\n")
        
        # Add embedding visualizations section
        f.write("## 0. Embedding Visualizations\n\n")
        f.write("Visualizing embeddings helps us understand how exercises and fitness concepts are related in the vector space.\n\n")
        
        if embedding_viz_dir:
            f.write("### Word2Vec Embeddings\n\n")
            f.write("Word2Vec captures semantic relationships between fitness terms:\n\n")
            f.write("![Word2Vec t-SNE](../data/embeddings/visualizations/word2vec_visualization_tsne.png)\n\n")
            f.write("*Word2Vec embeddings visualized using t-SNE*\n\n")
            
            f.write("### BERT Embeddings\n\n")
            f.write("BERT provides contextual embeddings of exercise descriptions, colored by exercise type:\n\n")
            f.write("![BERT by Exercise Type](../data/embeddings/visualizations/bert_visualization_by_category_tsne.png)\n\n")
            f.write("*BERT embeddings visualized using t-SNE, colored by exercise type*\n\n")
            
            f.write("Different dimensionality reduction techniques (t-SNE, PCA, UMAP) offer various perspectives on how exercises relate to each other in the embedding space.\n\n")
        
        f.write("## 1. Chain of Thought (CoT) Implementation\n\n")
        f.write("Chain of Thought allows us to break down complex reasoning into step-by-step processes.\n\n")
        f.write("### Application to Exercise Recommendations\n\n")
        f.write("- Implemented a 6-step reasoning process for exercise matching\n")
        f.write("- Analyzed exercise characteristics, difficulty, benefits, and techniques\n")
        f.write("- Generated personalized recommendations based on user profiles\n\n")
        
        # Rest of the function remains the same
        # Sample user result
        if cot_results and len(cot_results) > 0:
            sample = cot_results[0]
            user = sample['user']
            
            f.write(f"#### Example: User {user['name']}\n\n")
            f.write(f"Fitness Level: {user['fitness_level']}\n")
            f.write(f"Goals: {', '.join(user['goals'])}\n")
            f.write(f"Equipment: {', '.join(user['equipment_available'])}\n\n")
            
            if 'matched_exercises' in sample and len(sample['matched_exercises']) > 0:
                top_exercise = sample['matched_exercises'][0]
                f.write(f"**Top Match Score**: {top_exercise['final_match_score']:.2f}\n\n")
                f.write(f"**Recommendation**: {top_exercise['final_recommendation']}\n\n")
        
        f.write("## 2. Tree of Thought (ToT) Implementation\n\n")
        f.write("Tree of Thought explores multiple decision-making pathways for exercise recommendations.\n\n")
        f.write("### Workout Plan Decision Tree\n\n")
        f.write("- Primary Goal Path: Explores workout routines based on primary fitness goals\n")
        f.write("- Experience Level Path: Tailors recommendations to user's fitness experience\n")
        f.write("- Equipment-Based Path: Creates workout options based on available equipment\n\n")
        
        f.write("### Example Workout Branches\n\n")
        f.write("1. **Weight Loss Focus**\n")
        f.write("   - Cardio emphasis: HIIT -> Circuit Training -> Endurance Training\n")
        f.write("   - Strength component: Full Body -> Split Routines -> Specialized Training\n\n")
        f.write("2. **Muscle Building Focus**\n")
        f.write("   - Beginner -> Intermediate -> Advanced progression\n")
        f.write("   - Body weight -> Free weights -> Complex movements progression\n\n")
        
        f.write("## 3. Graph of Thought (GoT) Implementation\n\n")
        f.write("Graph of Thought maps relationships between exercises, equipment, and muscle groups.\n\n")
        
        f.write("### Exercise Relationship Network\n\n")
        f.write("- Created a graph of relationships between exercises and their attributes\n")
        f.write("- Identified complementary exercises that target related muscle groups\n")
        f.write("- Discovered exercise substitutions based on available equipment\n\n")
        
        f.write("### Key Insights\n\n")
        if got_results and 'graph_stats' in got_results:
            stats = got_results['graph_stats']
            f.write(f"- Analyzed {stats['num_exercises']} distinct exercises\n")
            f.write(f"- Mapped {stats['num_body_parts']} body parts and {stats['num_equipment_types']} equipment types\n")
            f.write(f"- Identified {stats.get('num_edges', 'many')} relationships between exercises\n\n")
            
            if 'most_connected_body_parts' in got_results:
                f.write("#### Most Versatile Body Parts (with most exercise options):\n")
                for body_part, count in got_results['most_connected_body_parts']:
                    f.write(f"- {body_part}: {count} exercises\n")
        
        f.write("\n## 4. Integration of Reasoning Frameworks\n\n")
        f.write("These three reasoning frameworks combine to create a comprehensive exercise recommendation system:\n\n")
        f.write("1. **CoT** provides detailed analysis of how well exercises match user profiles\n")
        f.write("2. **ToT** explores branching workout pathways for progressive training\n")
        f.write("3. **GoT** maps the relationships between exercises for versatile workout planning\n\n")
        
        f.write("See visualizations in the `data/cot`, `data/tot`, and `data/got` directories.\n")
    
    print(f"Report generated at reports/phase2_report.md")

if __name__ == "__main__":
    run_phase2()
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import json
import os
import random
from collections import Counter

def implement_graph_of_thought(data_file, output_dir):
    """
    Implement Graph of Thought reasoning for exercise recommendation
    
    Args:
        data_file (str): Path to preprocessed exercise data
        output_dir (str): Directory to save output files
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Load exercise data
    df = pd.read_csv('megaGymDataset.csv')
    
    # Create a graph
    G = nx.Graph()
    
    # Add nodes for each exercise
    for idx, row in df.iterrows():
        exercise_name = row['Title'] if pd.notna(row['Title']) else f"Exercise_{idx}"
        exercise_type = row['Type'] if pd.notna(row['Type']) else "Unknown"
        body_part = row['BodyPart'] if pd.notna(row['BodyPart']) else "Unknown"
        equipment = row['Equipment'] if pd.notna(row['Equipment']) else "Unknown"
        difficulty = row['Level'] if pd.notna(row['Level']) else "Unknown"
        
        # Add exercise node with attributes
        G.add_node(exercise_name, 
                  type='exercise',
                  exercise_type=exercise_type,
                  body_part=body_part,
                  equipment=equipment,
                  difficulty=difficulty)
        
        # Add nodes for body parts, equipment, etc. if they don't exist
        if not G.has_node(body_part):
            G.add_node(body_part, type='body_part')
        
        if not G.has_node(equipment):
            G.add_node(equipment, type='equipment')
        
        if not G.has_node(exercise_type):
            G.add_node(exercise_type, type='exercise_type')
            
        if not G.has_node(difficulty):
            G.add_node(difficulty, type='difficulty')
        
        # Add edges between exercise and its attributes
        G.add_edge(exercise_name, body_part, type='targets')
        G.add_edge(exercise_name, equipment, type='requires')
        G.add_edge(exercise_name, exercise_type, type='classified_as')
        G.add_edge(exercise_name, difficulty, type='difficulty_level')
    
    # Create additional connections between related exercises
    for idx1, row1 in df.iterrows():
        name1 = row1['Title'] if pd.notna(row1['Title']) else f"Exercise_{idx1}"
        body_part1 = row1['BodyPart'] if pd.notna(row1['BodyPart']) else "Unknown"
        equipment1 = row1['Equipment'] if pd.notna(row1['Equipment']) else "Unknown"
        
        # Link exercises that target the same body part and use similar equipment
        for idx2, row2 in df.iloc[idx1+1:].iterrows():
            name2 = row2['Title'] if pd.notna(row2['Title']) else f"Exercise_{idx2}"
            body_part2 = row2['BodyPart'] if pd.notna(row2['BodyPart']) else "Unknown"
            equipment2 = row2['Equipment'] if pd.notna(row2['Equipment']) else "Unknown"
            
            edge_weight = 0
            relationship = []
            
            # Check for shared body part
            if body_part1 == body_part2:
                edge_weight += 0.5
                relationship.append("same_body_part")
            
            # Check for shared equipment
            if equipment1 == equipment2:
                edge_weight += 0.3
                relationship.append("same_equipment")
            
            # Only add edge if there's a meaningful relationship
            if edge_weight > 0:
                G.add_edge(name1, name2, weight=edge_weight, relationship="|".join(relationship))
    
    # Analyze the graph to identify exercise clusters
    print("Analyzing exercise relationships...")
    
    # Get clusters of related exercises
    # Using connected components for simplicity, but could use community detection for more sophistication
    body_part_clusters = {}
    equipment_clusters = {}
    
    # Group exercises by body part
    for body_part in set(nx.get_node_attributes(G, 'type').keys()):
        if G.nodes[body_part].get('type') == 'body_part':
            exercises = [n for n in G.neighbors(body_part) if G.nodes[n].get('type') == 'exercise']
            if len(exercises) > 0:
                body_part_clusters[body_part] = exercises
    
    # Group exercises by equipment
    for equipment in set(nx.get_node_attributes(G, 'type').keys()):
        if G.nodes[equipment].get('type') == 'equipment':
            exercises = [n for n in G.neighbors(equipment) if G.nodes[n].get('type') == 'exercise']
            if len(exercises) > 0:
                equipment_clusters[equipment] = exercises
    
    # Find substitute exercises (exercises targeting the same muscle group with different equipment)
    substitution_graph = {}
    for body_part, exercises in body_part_clusters.items():
        if len(exercises) > 5:  # Only consider body parts with sufficient exercises
            equipment_options = {}
            for exercise in exercises:
                ex_equipment = G.nodes[exercise].get('equipment', 'Unknown')
                if ex_equipment not in equipment_options:
                    equipment_options[ex_equipment] = []
                equipment_options[ex_equipment].append(exercise)
            
            # If we have multiple equipment options for this body part
            if len(equipment_options) > 1:
                for eq1, ex_list1 in equipment_options.items():
                    substitution_graph[eq1] = {}
                    for eq2, ex_list2 in equipment_options.items():
                        if eq1 != eq2:
                            substitution_graph[eq1][eq2] = {
                                "from_exercises": ex_list1[:3],  # Limit to 3 examples
                                "to_exercises": ex_list2[:3]     # Limit to 3 examples
                            }
    
    # Generate progression paths based on difficulty
    progression_paths = {}
    for body_part, exercises in body_part_clusters.items():
        difficulty_levels = {}
        for exercise in exercises:
            difficulty = G.nodes[exercise].get('difficulty', 'Unknown')
            if difficulty not in difficulty_levels:
                difficulty_levels[difficulty] = []
            difficulty_levels[difficulty].append(exercise)
        
        # If we have multiple difficulty levels
        if len(difficulty_levels) > 1:
            progression_paths[body_part] = {
                "beginner": difficulty_levels.get("Beginner", [])[:5],
                "intermediate": difficulty_levels.get("Intermediate", [])[:5],
                "advanced": difficulty_levels.get("Advanced", [])[:5]
            }
    
    # Create a smaller visualization graph for clarity
    viz_graph = nx.Graph()
    
    # Sample a few body parts for visualization
    sampled_body_parts = random.sample(list(body_part_clusters.keys()), min(5, len(body_part_clusters)))
    
    for body_part in sampled_body_parts:
        viz_graph.add_node(body_part, type='body_part')
        for exercise in body_part_clusters[body_part][:3]:  # Only take first 3 exercises
            viz_graph.add_node(exercise, type='exercise')
            viz_graph.add_edge(body_part, exercise)
    
    # Create position layout
    pos = nx.spring_layout(viz_graph, k=0.5, iterations=50)
    
    # Draw the graph
    plt.figure(figsize=(12, 10))
    
    # Draw nodes with different colors based on type
    body_part_nodes = [n for n, attr in viz_graph.nodes(data=True) if attr.get('type') == 'body_part']
    exercise_nodes = [n for n, attr in viz_graph.nodes(data=True) if attr.get('type') == 'exercise']
    
    nx.draw_networkx_nodes(viz_graph, pos, nodelist=body_part_nodes, node_color='red', node_size=500, alpha=0.8)
    nx.draw_networkx_nodes(viz_graph, pos, nodelist=exercise_nodes, node_color='blue', node_size=300, alpha=0.6)
    
    # Draw edges
    nx.draw_networkx_edges(viz_graph, pos, width=1.0, alpha=0.5)
    
    # Draw labels with smaller font size
    nx.draw_networkx_labels(viz_graph, pos, font_size=8, font_family='sans-serif')
    
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(f"{output_dir}/exercise_relationships.png", dpi=300)
    plt.close()
    
    # Save results to JSON
    results = {
        "body_part_clusters": {k: v[:10] for k, v in body_part_clusters.items()},  # Limit to 10 exercises per body part
        "equipment_clusters": {k: v[:10] for k, v in equipment_clusters.items()},   # Limit to 10 exercises per equipment
        "substitution_options": substitution_graph,
        "progression_paths": progression_paths,
        "graph_stats": {
            "num_nodes": G.number_of_nodes(),
            "num_edges": G.number_of_edges(),
            "num_body_parts": len([n for n, attr in G.nodes(data=True) if attr.get('type') == 'body_part']),
            "num_equipment_types": len([n for n, attr in G.nodes(data=True) if attr.get('type') == 'equipment']),
            "num_exercises": len([n for n, attr in G.nodes(data=True) if attr.get('type') == 'exercise'])
        },
        "most_connected_body_parts": sorted(
            [(body_part, len(exercises)) for body_part, exercises in body_part_clusters.items()],
            key=lambda x: x[1], reverse=True
        )[:5],
        "most_versatile_equipment": sorted(
            [(equipment, len(exercises)) for equipment, exercises in equipment_clusters.items()],
            key=lambda x: x[1], reverse=True
        )[:5]
    }
    
    with open(f"{output_dir}/got_results.json", "w") as f:
        json.dump(results, f, indent=2)
    
    print(f"Graph of Thought reasoning implemented and saved to {output_dir}/got_results.json")
    print(f"Graph visualization saved to {output_dir}/exercise_relationships.png")
    
    return results

if __name__ == "__main__":
    implement_graph_of_thought("megaGymDataset.csv", "data/got")
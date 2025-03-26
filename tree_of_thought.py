import pandas as pd
import json
import os
from collections import defaultdict
import random
import matplotlib.pyplot as plt
import networkx as nx

def implement_tree_of_thought(data_file, output_dir):
    """
    Implement Tree of Thought reasoning for exercise recommendation
    
    Args:
        data_file (str): Path to preprocessed exercise data
        output_dir (str): Directory to save output files
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Load exercise data
    df = pd.read_csv('megaGymDataset.csv')
    
    # Define fitness goals and their corresponding pathways
    fitness_goals = {
        "Weight Loss": {
            "description": "Exercises focused on burning calories and fat loss",
            "pathways": [
                {
                    "name": "Cardio Focus",
                    "description": "Emphasizes cardiovascular exercises for calorie burning",
                    "exercise_types": ["Cardio", "Plyometrics"],
                    "intensity": "High",
                    "frequency": "4-6 days per week",
                    "reasoning": "Cardio exercises elevate heart rate and maximize calorie burn, supporting weight loss goals."
                },
                {
                    "name": "HIIT Training",
                    "description": "High-intensity interval training for maximum calorie burn",
                    "exercise_types": ["Cardio", "Plyometrics", "Strength"],
                    "intensity": "Very High",
                    "frequency": "3-4 days per week",
                    "reasoning": "HIIT combines intense bursts with recovery periods, creating an afterburn effect that increases calorie expenditure for hours post-workout."
                },
                {
                    "name": "Circuit Training",
                    "description": "Full-body workout with minimal rest between exercises",
                    "exercise_types": ["Strength", "Cardio"],
                    "intensity": "Moderate to High",
                    "frequency": "3-5 days per week",
                    "reasoning": "Circuit training keeps heart rate elevated while building muscle, creating an efficient workout for calorie burning and metabolic boost."
                }
            ]
        },
        "Muscle Gain": {
            "description": "Exercises focused on building muscle mass and strength",
            "pathways": [
                {
                    "name": "Hypertrophy Focus",
                    "description": "Targets muscle growth through moderate weights and higher reps",
                    "exercise_types": ["Strength"],
                    "intensity": "Moderate to High",
                    "frequency": "4-5 days per week",
                    "reasoning": "Moderate weight with higher rep ranges (8-12) creates optimal muscle tension and metabolic stress for hypertrophy."
                },
                {
                    "name": "Strength Focus",
                    "description": "Emphasizes heavy weights and compound movements",
                    "exercise_types": ["Strength"],
                    "intensity": "High",
                    "frequency": "3-4 days per week",
                    "reasoning": "Heavy compound exercises recruit more muscle fibers and stimulate greater strength gains through progressive overload."
                },
                {
                    "name": "Body Part Split",
                    "description": "Targets specific muscle groups on different days",
                    "exercise_types": ["Strength"],
                    "intensity": "Moderate to High",
                    "frequency": "5-6 days per week",
                    "reasoning": "Focusing on individual muscle groups allows for greater volume per body part and adequate recovery between sessions."
                }
            ]
        },
        "General Fitness": {
            "description": "Balanced approach for overall health and fitness",
            "pathways": [
                {
                    "name": "Balanced Routine",
                    "description": "Mix of cardio, strength, and flexibility training",
                    "exercise_types": ["Cardio", "Strength", "Stretching"],
                    "intensity": "Moderate",
                    "frequency": "3-5 days per week",
                    "reasoning": "A balanced approach ensures all components of fitness are addressed, promoting overall health and well-rounded physical development."
                },
                {
                    "name": "Functional Fitness",
                    "description": "Exercises that mimic everyday movements",
                    "exercise_types": ["Strength", "Plyometrics"],
                    "intensity": "Moderate",
                    "frequency": "3-4 days per week",
                    "reasoning": "Functional training improves performance in daily activities by focusing on movement patterns rather than isolated muscles."
                },
                {
                    "name": "Active Recovery",
                    "description": "Low-intensity exercises for recovery and mobility",
                    "exercise_types": ["Stretching", "Cardio"],
                    "intensity": "Low",
                    "frequency": "2-3 days per week",
                    "reasoning": "Active recovery promotes blood flow to muscles, enhances mobility, and reduces soreness while allowing the body to recuperate."
                }
            ]
        },
        "Endurance": {
            "description": "Exercises to improve stamina and cardiovascular fitness",
            "pathways": [
                {
                    "name": "Long Duration Cardio",
                    "description": "Extended moderate-intensity cardiovascular training",
                    "exercise_types": ["Cardio"],
                    "intensity": "Moderate",
                    "frequency": "3-5 days per week",
                    "reasoning": "Sustained cardio activity improves heart efficiency, increases capillary density, and enhances the body's ability to utilize oxygen."
                },
                {
                    "name": "Endurance Intervals",
                    "description": "Alternating between moderate and higher intensities",
                    "exercise_types": ["Cardio", "Plyometrics"],
                    "intensity": "Moderate to High",
                    "frequency": "3-4 days per week",
                    "reasoning": "Interval training improves both aerobic and anaerobic systems, increasing overall endurance capacity."
                },
                {
                    "name": "Cross-Training",
                    "description": "Variety of endurance activities to prevent overuse",
                    "exercise_types": ["Cardio", "Strength"],
                    "intensity": "Moderate",
                    "frequency": "4-6 days per week",
                    "reasoning": "Varying exercise modalities reduces repetitive stress while still building endurance, decreasing injury risk while improving overall fitness."
                }
            ]
        }
    }
    
    # Define experience levels and their constraints
    experience_levels = {
        "Beginner": {
            "description": "New to exercise or returning after long break",
            "suitable_difficulties": ["Beginner"],
            "recommended_frequency": "2-3 days per week",
            "workout_duration": "20-30 minutes",
            "rest_periods": "Longer rest periods (1-2 minutes)",
            "progression_rate": "Focus on form and consistency before increasing intensity"
        },
        "Intermediate": {
            "description": "Regular exerciser with good technique",
            "suitable_difficulties": ["Beginner", "Intermediate"],
            "recommended_frequency": "3-5 days per week",
            "workout_duration": "30-60 minutes",
            "rest_periods": "Moderate rest periods (45-90 seconds)",
            "progression_rate": "Gradual increases in intensity, weight, or duration"
        },
        "Advanced": {
            "description": "Experienced exerciser seeking new challenges",
            "suitable_difficulties": ["Beginner", "Intermediate", "Advanced"],
            "recommended_frequency": "4-6 days per week",
            "workout_duration": "45-90 minutes",
            "rest_periods": "Variable rest periods based on training goal",
            "progression_rate": "Can handle more specialized and intense training methods"
        }
    }
    
    # Define sample users
    users = [
        {
            "name": "Alex",
            "age": 32,
            "fitness_level": "Beginner",
            "primary_goal": "Weight Loss",
            "secondary_goal": "General Fitness",
            "available_equipment": ["Body Only", "Dumbbells"],
            "time_available": "30 minutes",
            "limitations": ["Knee issues"],
            "preferences": {
                "workout_location": "Home",
                "exercise_types": ["Cardio", "Strength"]
            }
        },
        {
            "name": "Jordan",
            "age": 28,
            "fitness_level": "Intermediate",
            "primary_goal": "Muscle Gain",
            "secondary_goal": "Endurance",
            "available_equipment": ["Barbell", "Dumbbells", "Cable", "Bands"],
            "time_available": "60 minutes",
            "limitations": [],
            "preferences": {
                "workout_location": "Gym",
                "exercise_types": ["Strength"]
            }
        },
        {
            "name": "Taylor",
            "age": 35,
            "fitness_level": "Advanced",
            "primary_goal": "Endurance",
            "secondary_goal": "General Fitness",
            "available_equipment": ["Body Only", "Kettlebells", "Medicine Ball", "Bands"],
            "time_available": "45 minutes",
            "limitations": ["Lower back sensitivity"],
            "preferences": {
                "workout_location": "Home",
                "exercise_types": ["Cardio", "Plyometrics"]
            }
        }
    ]
    
    # Process the dataset to extract useful information
    body_parts = sorted(df['BodyPart'].dropna().unique())
    equipment_types = sorted(df['Equipment'].dropna().unique())
    exercise_types = sorted(df['Type'].dropna().unique())
    
    # Build exercise recommendations using Tree of Thought
    tot_results = []
    
    for user in users:
        # Initialize a dictionary to store the decision tree paths for this user
        user_decision_tree = {
            "user": user,
            "primary_goal_paths": [],
            "secondary_goal_paths": [],
            "equipment_based_paths": [],
            "time_constrained_paths": [],
            "body_part_focus_paths": []
        }
        
        # 1. Primary Goal Paths
        primary_goal = user["primary_goal"]
        if primary_goal in fitness_goals:
            goal_info = fitness_goals[primary_goal]
            
            for pathway in goal_info["pathways"]:
                # Check if pathway is suitable for user's experience level
                is_suitable = True
                pathway_reasoning = []
                
                # Check intensity suitability
                if user["fitness_level"] == "Beginner" and pathway["intensity"] == "Very High":
                    is_suitable = False
                    pathway_reasoning.append("Intensity too high for beginner level")
                
                # Check equipment availability
                required_equipment = []
                for ex_type in pathway["exercise_types"]:
                    type_df = df[df["Type"] == ex_type]
                    eq_needed = type_df["Equipment"].dropna().unique().tolist()
                    required_equipment.extend(eq_needed)
                
                required_equipment = list(set(required_equipment))
                missing_equipment = [eq for eq in required_equipment if eq not in user["available_equipment"] and eq != "Body Only" and eq != "None"]
                
                if missing_equipment and len(missing_equipment) > len(required_equipment) // 2:
                    is_suitable = False
                    pathway_reasoning.append(f"Missing essential equipment: {', '.join(missing_equipment[:3])}")
                
                # Filter exercises based on pathway criteria
                pathway_exercises = []
                for ex_type in pathway["exercise_types"]:
                    type_df = df[df["Type"] == ex_type]
                    
                    # Filter by equipment
                    eq_filter = type_df["Equipment"].isin(user["available_equipment"] + ["Body Only", "None"])
                    filtered_df = type_df[eq_filter]
                    
                    # Filter by difficulty
                    if user["fitness_level"] == "Beginner":
                        diff_filter = filtered_df["Level"].isin(["Beginner"])
                    elif user["fitness_level"] == "Intermediate":
                        diff_filter = filtered_df["Level"].isin(["Beginner", "Intermediate"])
                    else:  # Advanced
                        diff_filter = filtered_df["Level"].isin(["Beginner", "Intermediate", "Advanced"])
                    
                    filtered_df = filtered_df[diff_filter]
                    
                    # Check user limitations and exclude problematic exercises
                    if "Knee issues" in user["limitations"]:
                        filtered_df = filtered_df[~filtered_df["BodyPart"].isin(["Quadriceps", "Calves"])]
                    if "Lower back sensitivity" in user["limitations"]:
                        filtered_df = filtered_df[~filtered_df["BodyPart"].isin(["Lower Back"])]
                    
                    # Get top exercises by rating (if available) or random selection
                    if "Rating" in filtered_df.columns and filtered_df["Rating"].notna().any():
                        top_exercises = filtered_df.nlargest(5, "Rating")
                    else:
                        top_exercises = filtered_df.sample(min(5, len(filtered_df)))
                    
                    pathway_exercises.extend(top_exercises.to_dict("records"))
                
                # Create a comprehensive path with reasoning
                path = {
                    "pathway_name": pathway["name"],
                    "pathway_description": pathway["description"],
                    "is_suitable": is_suitable,
                    "reasoning": pathway_reasoning if not is_suitable else [pathway["reasoning"]],
                    "recommended_exercises": pathway_exercises[:10],  # Limit to 10 exercises
                    "frequency": pathway["frequency"],
                    "intensity": pathway["intensity"]
                }
                
                user_decision_tree["primary_goal_paths"].append(path)
        
        # 2. Secondary Goal Paths (similar approach but for secondary goal)
        secondary_goal = user["secondary_goal"]
        if secondary_goal in fitness_goals:
            goal_info = fitness_goals[secondary_goal]
            
            for pathway in goal_info["pathways"]:
                # Similar logic as above, but we'll keep it shorter for the secondary goal
                is_suitable = True
                pathway_reasoning = []
                
                if user["fitness_level"] == "Beginner" and pathway["intensity"] == "Very High":
                    is_suitable = False
                    pathway_reasoning.append("Intensity too high for beginner level")
                
                # Filter exercises (simplified for secondary goal)
                pathway_exercises = []
                for ex_type in pathway["exercise_types"]:
                    type_df = df[df["Type"] == ex_type]
                    eq_filter = type_df["Equipment"].isin(user["available_equipment"] + ["Body Only", "None"])
                    filtered_df = type_df[eq_filter]
                    
                    # Simple random selection for secondary goal
                    selected_exercises = filtered_df.sample(min(3, len(filtered_df)))
                    pathway_exercises.extend(selected_exercises.to_dict("records"))
                
                path = {
                    "pathway_name": pathway["name"],
                    "pathway_description": pathway["description"],
                    "is_suitable": is_suitable,
                    "reasoning": pathway_reasoning if not is_suitable else [pathway["reasoning"]],
                    "recommended_exercises": pathway_exercises[:5],  # Limit to 5 exercises for secondary goal
                    "frequency": pathway["frequency"],
                    "intensity": pathway["intensity"]
                }
                
                user_decision_tree["secondary_goal_paths"].append(path)
        
        # 3. Equipment-Based Paths
        user_equipment = user["available_equipment"]
        for equipment in user_equipment:
            equipment_df = df[df["Equipment"] == equipment]
            
            # Skip if not enough exercises available for this equipment
            if len(equipment_df) < 5:
                continue
            
            # Filter by user fitness level
            if user["fitness_level"] == "Beginner":
                equipment_df = equipment_df[equipment_df["Level"].isin(["Beginner"])]
            elif user["fitness_level"] == "Intermediate":
                equipment_df = equipment_df[equipment_df["Level"].isin(["Beginner", "Intermediate"])]
            else:  # Advanced
                equipment_df = equipment_df[equipment_df["Level"].isin(["Beginner", "Intermediate", "Advanced"])]
            
            # Group by body part to create a full-body routine
            body_part_groups = defaultdict(list)
            for _, row in equipment_df.iterrows():
                body_part = row["BodyPart"] if pd.notna(row["BodyPart"]) else "Other"
                body_part_groups[body_part].append(row.to_dict())
            
            # Create a balanced routine from various body parts
            balanced_routine = []
            for body_part, exercises in body_part_groups.items():
                if len(exercises) > 0:
                    balanced_routine.append(exercises[0])  # Add one exercise per body part
                if len(balanced_routine) >= 8:  # Limit to 8 exercises for a balanced routine
                    break
            
            path = {
                "equipment": equipment,
                "description": f"Full-body workout using {equipment}",
                "recommended_exercises": balanced_routine,
                "reasoning": [f"{equipment} allows for a diverse range of exercises targeting different muscle groups.",
                             "Using a single piece of equipment simplifies workout setup and transitions."]
            }
            
            user_decision_tree["equipment_based_paths"].append(path)
        
        # 4. Time-Constrained Paths
        time_available = user["time_available"]
        time_mins = int(time_available.split()[0])  # Extract minutes from "30 minutes"
        
        time_paths = []
        if time_mins <= 20:
            # For very short workouts, suggest high-intensity options
            high_intensity_df = df[df["Type"].isin(["Plyometrics", "Cardio"])]
            high_intensity_df = high_intensity_df[high_intensity_df["Equipment"].isin(user["available_equipment"] + ["Body Only", "None"])]
            
            if len(high_intensity_df) >= 5:
                exercises = high_intensity_df.sample(min(6, len(high_intensity_df))).to_dict("records")
                
                path = {
                    "duration": f"{time_mins} minutes",
                    "description": "Quick High-Intensity Circuit",
                    "recommended_exercises": exercises,
                    "reasoning": ["Short, intense circuits maximize efficiency for time-constrained workouts.",
                                 "Full-body movements elevate heart rate quickly and provide both strength and cardio benefits."]
                }
                time_paths.append(path)
        
        elif time_mins <= 45:
            # For medium-length workouts, suggest focused routines
            if "Strength" in user["preferences"]["exercise_types"]:
                # Create a focused strength routine
                strength_df = df[df["Type"] == "Strength"]
                strength_df = strength_df[strength_df["Equipment"].isin(user["available_equipment"] + ["Body Only", "None"])]
                
                # Group by body part to create a split routine
                upper_body_df = strength_df[strength_df["BodyPart"].isin(["Chest", "Back", "Shoulders", "Biceps", "Triceps"])]
                lower_body_df = strength_df[strength_df["BodyPart"].isin(["Quadriceps", "Hamstrings", "Glutes", "Calves"])]
                
                # Create upper/lower routines
                upper_exercises = upper_body_df.sample(min(5, len(upper_body_df))).to_dict("records")
                lower_exercises = lower_body_df.sample(min(5, len(lower_body_df))).to_dict("records")
                
                path1 = {
                    "duration": f"{time_mins} minutes",
                    "description": "Upper Body Strength Focus",
                    "recommended_exercises": upper_exercises,
                    "reasoning": ["Focusing on upper body allows sufficient volume for growth in a time-constrained session.",
                                 "Alternating with lower body days ensures balanced development and adequate recovery."]
                }
                
                path2 = {
                    "duration": f"{time_mins} minutes",
                    "description": "Lower Body Strength Focus",
                    "recommended_exercises": lower_exercises,
                    "reasoning": ["Lower body workouts engage larger muscle groups, maximizing hormonal response.",
                                 "Focusing on legs/core provides efficient training stimulus in a time-limited session."]
                }
                
                time_paths.extend([path1, path2])
            
            else:
                # Create a mixed cardio/bodyweight routine
                mixed_df = df[df["Type"].isin(["Cardio", "Plyometrics", "Strength"])]
                mixed_df = mixed_df[mixed_df["Equipment"].isin(["Body Only", "None"])]
                
                exercises = mixed_df.sample(min(8, len(mixed_df))).to_dict("records")
                
                path = {
                    "duration": f"{time_mins} minutes",
                    "description": "Efficient Full-Body Circuit",
                    "recommended_exercises": exercises,
                    "reasoning": ["Combining multi-joint movements provides a balanced workout in limited time.",
                                 "Circuit format keeps heart rate elevated for cardiovascular benefits while building strength."]
                }
                time_paths.append(path)
        
        else:  # More than 45 minutes
            # For longer workouts, suggest comprehensive routines
            # Create a full split routine (if user prefers strength)
            if "Strength" in user["preferences"]["exercise_types"]:
                strength_df = df[df["Type"] == "Strength"]
                strength_df = strength_df[strength_df["Equipment"].isin(user["available_equipment"] + ["Body Only", "None"])]
                
                # Group by body part
                push_df = strength_df[strength_df["BodyPart"].isin(["Chest", "Shoulders", "Triceps"])]
                pull_df = strength_df[strength_df["BodyPart"].isin(["Back", "Biceps"])]
                legs_df = strength_df[strength_df["BodyPart"].isin(["Quadriceps", "Hamstrings", "Glutes", "Calves"])]
                
                # Create push/pull/legs routines
                push_exercises = push_df.sample(min(7, len(push_df))).to_dict("records")
                pull_exercises = pull_df.sample(min(7, len(pull_df))).to_dict("records")
                legs_exercises = legs_df.sample(min(7, len(legs_df))).to_dict("records")
                
                path1 = {
                    "duration": f"{time_mins} minutes",
                    "description": "Push Day (Chest, Shoulders, Triceps)",
                    "recommended_exercises": push_exercises,
                    "reasoning": ["Push-focused workout allows targeted volume for these complementary muscle groups.",
                                 "This split optimizes recovery by spacing out training for each muscle group."]
                }
                
                path2 = {
                    "duration": f"{time_mins} minutes",
                    "description": "Pull Day (Back, Biceps)",
                    "recommended_exercises": pull_exercises,
                    "reasoning": ["Pull-focused workout targets the posterior chain with complementary movements.",
                                 "This approach balances push movements and maintains structural balance."]
                }
                
                path3 = {
                    "duration": f"{time_mins} minutes",
                    "description": "Legs Day (Lower Body)",
                    "recommended_exercises": legs_exercises,
                    "reasoning": ["Dedicated leg day allows sufficient volume and intensity for lower body development.",
                                 "Lower body training stimulates the most significant hormonal response, benefiting overall growth."]
                }
                
                time_paths.extend([path1, path2, path3])
            
            else:
                # Create a comprehensive cardio/functional routine
                cardio_df = df[df["Type"].isin(["Cardio", "Plyometrics"])]
                cardio_df = cardio_df[cardio_df["Equipment"].isin(user["available_equipment"] + ["Body Only", "None"])]
                
                functional_df = df[df["Type"] == "Strength"]
                functional_df = functional_df[functional_df["Equipment"].isin(user["available_equipment"] + ["Body Only", "None"])]
                
                # Build a mixed routine
                cardio_exercises = cardio_df.sample(min(5, len(cardio_df))).to_dict("records")
                functional_exercises = functional_df.sample(min(5, len(functional_df))).to_dict("records")
                
                path = {
                    "duration": f"{time_mins} minutes",
                    "description": "Comprehensive Cardio & Functional Training",
                    "recommended_exercises": cardio_exercises + functional_exercises,
                    "reasoning": ["Extended workout time allows for both cardiovascular conditioning and strength work.",
                                 "This comprehensive approach provides balanced fitness development addressing multiple components."]
                }
                time_paths.append(path)
        
        user_decision_tree["time_constrained_paths"] = time_paths
        
        # 5. Body Part Focus Paths
        # Create targeted routines for major body parts
        major_body_parts = ["Abdominals", "Chest", "Back", "Shoulders", "Arms", "Legs"]
        
        body_part_paths = []
        for body_part in major_body_parts:
            # For "Arms", we'll include both biceps and triceps
            if body_part == "Arms":
                body_part_df = df[df["BodyPart"].isin(["Biceps", "Triceps"])]
            # For "Legs", we'll include all lower body parts
            elif body_part == "Legs":
                body_part_df = df[df["BodyPart"].isin(["Quadriceps", "Hamstrings", "Glutes", "Calves"])]
            else:
                body_part_df = df[df["BodyPart"] == body_part]
            
            body_part_df = body_part_df[body_part_df["Equipment"].isin(user["available_equipment"] + ["Body Only", "None"])]
            
            # Skip if not enough exercises available
            if len(body_part_df) < 3:
                continue
            
            # Get a mix of exercises
            selected_exercises = body_part_df.sample(min(6, len(body_part_df))).to_dict("records")
            
            path = {
                "body_part": body_part,
                "description": f"{body_part} Focused Workout",
                "recommended_exercises": selected_exercises,
                "reasoning": [f"Dedicated {body_part} training allows targeted development of this muscle group.",
                             "Focusing on one area increases training volume and can overcome plateaus."]
            }
            
            body_part_paths.append(path)
        
        user_decision_tree["body_part_focus_paths"] = body_part_paths
        
        # Add the complete decision tree to results
        tot_results.append(user_decision_tree)
    
    # Create a visualization of a decision tree path for the first user
    if tot_results:
        G = nx.DiGraph()
        
        # Add user node
        user = tot_results[0]["user"]
        user_node = f"User: {user['name']}"
        G.add_node(user_node, color='blue')
        
        # Add primary goal node
        primary_goal = user["primary_goal"]
        primary_node = f"Goal: {primary_goal}"
        G.add_node(primary_node, color='green')
        G.add_edge(user_node, primary_node)
        
        # Add pathways
        for i, path in enumerate(tot_results[0]["primary_goal_paths"]):
            if path["is_suitable"]:
                pathway_node = f"Pathway {i+1}: {path['pathway_name']}"
                G.add_node(pathway_node, color='orange')
                G.add_edge(primary_node, pathway_node)
                
                # Add exercise nodes (limit to 3 for clarity)
                for j, exercise in enumerate(path["recommended_exercises"][:3]):
                    exercise_name = exercise["Title"] if "Title" in exercise else f"Exercise {j+1}"
                    G.add_node(exercise_name, color='red')
                    G.add_edge(pathway_node, exercise_name)
        
        # Create the visualization
        plt.figure(figsize=(12, 8))
        pos = nx.spring_layout(G)
        
        # Node colors based on type
        node_colors = [G.nodes[n].get('color', 'gray') for n in G.nodes()]
        
        nx.draw(G, pos, with_labels=True, node_color=node_colors, 
                node_size=2000, font_size=8, font_weight='bold',
                arrows=True, arrowsize=15, edge_color='gray')
        
        plt.title("Exercise Recommendation Decision Tree")
        plt.savefig(f"{output_dir}/exercise_decision_tree.png", dpi=300, bbox_inches='tight')
        plt.close()
    
    # Save the complete TOT results
    with open(f"{output_dir}/tot_results.json", "w") as f:
        json.dump(tot_results, f, indent=2)
    
    print(f"Tree of Thought reasoning implemented and saved to {output_dir}/tot_results.json")
    print(f"Decision tree visualization saved to {output_dir}/exercise_decision_tree.png")
    
    return tot_results

if __name__ == "__main__":
    implement_tree_of_thought("megaGymDataset.csv", "data/tot")
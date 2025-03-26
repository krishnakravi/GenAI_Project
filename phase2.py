import pandas as pd
import json
import os
import spacy

# Load spaCy model for text analysis
nlp = spacy.load("en_core_web_sm")

def implement_chain_of_thought(data_file, output_dir):
    """
    Implement Chain of Thought reasoning for exercise recommendations
    
    Args:
        data_file (str): Path to preprocessed exercise data
        output_dir (str): Directory to save output files
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Load preprocessed data
    df = pd.read_csv(data_file)
    
    # Define a base CoT framework for exercise recommendations
    cot_framework = {
        "steps": [
            {
                "step": 1,
                "description": "Identify exercise characteristics",
                "reasoning": "Extract exercise type, muscle groups, and equipment needed"
            },
            {
                "step": 2,
                "description": "Determine difficulty level and intensity",
                "reasoning": "Analyze exercise difficulty (beginner, intermediate, advanced)"
            },
            {
                "step": 3,
                "description": "Identify exercise benefits and target areas",
                "reasoning": "List specific muscle groups and fitness benefits"
            },
            {
                "step": 4,
                "description": "Analyze exercise technique and form",
                "reasoning": "Understand proper form and potential modifications"
            },
            {
                "step": 5, 
                "description": "Match user profile to exercise requirements",
                "reasoning": "Compare the user's fitness level and goals to exercise characteristics"
            },
            {
                "step": 6,
                "description": "Generate personalized recommendation",
                "reasoning": "Provide a final recommendation including exercise suitability and progression path"
            }
        ]
    }
    
    # Sample user profiles for demonstration
    user_profiles = [
        {
            "id": 1,
            "name": "Alex Johnson",
            "fitness_level": "Beginner",
            "goals": ["Weight loss", "General fitness"],
            "equipment_available": ["Body Only", "Dumbbells"],
            "preferred_exercises": ["Cardio", "Full body workouts"],
            "time_available": "30 minutes",
            "preferences": {
                "home_workout": True,
                "workout_intensity": "Moderate",
                "focus_areas": ["Abdominals", "Legs"]
            }
        },
        {
            "id": 2,
            "name": "Jordan Smith",
            "fitness_level": "Intermediate",
            "goals": ["Muscle gain", "Strength"],
            "equipment_available": ["Barbell", "Dumbbells", "Cable"],
            "preferred_exercises": ["Strength training", "Weight lifting"],
            "time_available": "60 minutes",
            "preferences": {
                "home_workout": False,
                "workout_intensity": "High",
                "focus_areas": ["Chest", "Back", "Arms"]
            }
        },
        {
            "id": 3,
            "name": "Taylor Williams",
            "fitness_level": "Advanced",
            "goals": ["Endurance", "Functional fitness"],
            "equipment_available": ["Body Only", "Kettlebells", "Medicine Ball"],
            "preferred_exercises": ["HIIT", "Functional training"],
            "time_available": "45 minutes",
            "preferences": {
                "home_workout": True,
                "workout_intensity": "Very High",
                "focus_areas": ["Full body", "Core"]
            }
        }
    ]
    
    # Apply CoT to match users with exercises
    cot_results = []
    
    for user in user_profiles:
        matched_exercises = []
        
        # Filter exercises based on user preferences
        filtered_df = df.copy()
        
        # Filter by equipment if specified
        if user["equipment_available"]:
            filtered_df = filtered_df[filtered_df["Equipment"].isin(user["equipment_available"])]
        
        # Filter by difficulty level 
        difficulty_mapping = {
            "Beginner": ["Beginner"],
            "Intermediate": ["Beginner", "Intermediate"],
            "Advanced": ["Beginner", "Intermediate", "Advanced"]
        }
        
        if user["fitness_level"] in difficulty_mapping:
            filtered_df = filtered_df[filtered_df["Level"].isin(difficulty_mapping[user["fitness_level"]])]
        
        # Get top 10 exercises for demonstration
        for idx, exercise in filtered_df.head(10).iterrows():
            exercise_description = exercise["Desc"] if pd.notna(exercise["Desc"]) else ""
            exercise_name = exercise["Title"] if "Title" in exercise else f"Exercise {idx}"
            
            # Process the exercise description
            doc = nlp(exercise_description)
            
            # Initialize CoT reasoning chain for this exercise-user pair
            reasoning_chain = {
                "user_id": user["id"],
                "user_name": user["name"],
                "exercise_id": idx,
                "exercise_name": exercise_name,
                "reasoning_steps": []
            }
            
            # Step 1: Identify exercise characteristics
            exercise_type = exercise["Type"] if pd.notna(exercise["Type"]) else "Not specified"
            muscle_group = exercise["BodyPart"] if pd.notna(exercise["BodyPart"]) else "Not specified"
            equipment = exercise["Equipment"] if pd.notna(exercise["Equipment"]) else "Not specified"
            
            reasoning_chain["reasoning_steps"].append({
                "step": 1,
                "output": f"Exercise type: {exercise_type}, Target muscles: {muscle_group}, Equipment: {equipment}"
            })
            
            # Step 2: Determine difficulty level
            difficulty = exercise["Level"] if pd.notna(exercise["Level"]) else "Not specified"
            
            reasoning_chain["reasoning_steps"].append({
                "step": 2,
                "output": f"Difficulty level: {difficulty}"
            })
            
            # Step 3: Identify benefits and target areas
            benefits = []
            if len(doc.ents) > 0:
                for ent in doc.ents:
                    if ent.label_ in ["ORG", "PRODUCT"]:
                        benefits.append(ent.text)
            
            # If no entities found, use muscle group as benefit
            if not benefits and muscle_group != "Not specified":
                benefits = [f"Targets {muscle_group}"]
            
            reasoning_chain["reasoning_steps"].append({
                "step": 3,
                "output": f"Benefits: {', '.join(benefits) if benefits else 'General fitness improvement'}"
            })
            
            # Step 4: Analyze technique
            technique_info = "Proper form is essential" 
            if len(exercise_description) > 50:
                technique_info = exercise_description[:100] + "..."
            
            reasoning_chain["reasoning_steps"].append({
                "step": 4,
                "output": f"Technique note: {technique_info}"
            })
            
            # Step 5: Match user profile
            # Calculate match score based on multiple factors
            match_score = 0.0
            
            # Match on equipment
            if equipment in user["equipment_available"]:
                match_score += 0.3
            
            # Match on exercise type and user preference
            if any(pref.lower() in exercise_type.lower() for pref in user["preferred_exercises"]):
                match_score += 0.3
            
            # Match on muscle group and user focus areas
            if any(focus.lower() in muscle_group.lower() for focus in user["preferences"]["focus_areas"]):
                match_score += 0.4
            
            # Adjust for difficulty
            if user["fitness_level"] == "Beginner" and difficulty == "Advanced":
                match_score -= 0.2
            elif user["fitness_level"] == "Advanced" and difficulty == "Beginner":
                match_score -= 0.1
            
            # Ensure score is between 0 and 1
            match_score = max(0, min(1, match_score))
            
            reasoning_chain["reasoning_steps"].append({
                "step": 5,
                "output": f"User-exercise match score: {match_score:.2f}",
                "match_details": {
                    "equipment_match": equipment in user["equipment_available"],
                    "type_match": any(pref.lower() in exercise_type.lower() for pref in user["preferred_exercises"]),
                    "focus_match": any(focus.lower() in muscle_group.lower() for focus in user["preferences"]["focus_areas"])
                }
            })
            
            # Step 6: Generate recommendation
            if match_score > 0.7:
                recommendation = "Excellent match! This exercise aligns well with your goals and preferences."
            elif match_score > 0.4:
                recommendation = "Good match with some considerations for your fitness level and goals."
            else:
                recommendation = "Not an ideal match. Consider other exercises that better align with your preferences."
            
            progression_path = ""
            if difficulty == "Beginner":
                progression_path = "As you progress, increase reps or add weights."
            elif difficulty == "Intermediate":
                progression_path = "Focus on perfect form and consider advanced variations as you improve."
            else:
                progression_path = "This is an advanced exercise. Ensure your form is perfect to prevent injury."
            
            reasoning_chain["reasoning_steps"].append({
                "step": 6,
                "output": recommendation,
                "progression_path": progression_path
            })
            
            reasoning_chain["final_match_score"] = match_score
            reasoning_chain["final_recommendation"] = recommendation
            
            matched_exercises.append(reasoning_chain)
        
        # Sort exercises by match score
        matched_exercises.sort(key=lambda x: x["final_match_score"], reverse=True)
        cot_results.append({
            "user": user,
            "matched_exercises": matched_exercises
        })
    
    # Save CoT results
    with open(f"{output_dir}/cot_results.json", "w") as f:
        json.dump(cot_results, f, indent=2)
    
    print(f"Chain of Thought reasoning implemented and saved to {output_dir}/cot_results.json")
    
    return cot_results

if __name__ == "__main__":
    implement_chain_of_thought("data/processed_exercises.csv", "data/cot")
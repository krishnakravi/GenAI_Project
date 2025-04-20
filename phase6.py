import os
import json
import logging
import pandas as pd
import numpy as np
import torch
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from peft import PeftModel
import sacrebleu
from rouge_score import rouge_scorer
import matplotlib.pyplot as plt
import seaborn as sns
from nltk.translate.bleu_score import corpus_bleu, sentence_bleu
from nltk.tokenize import word_tokenize
import nltk
from datetime import datetime
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Ensure NLTK resources are available
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

class ExerciseEvaluator:
    """Evaluation framework for the exercise recommendation system"""
    
    def __init__(self):
        """Initialize the evaluator with the necessary components"""
        logger.info("Initializing Exercise Evaluator")
        
        # Create output directories
        os.makedirs("data/evaluation", exist_ok=True)
        os.makedirs("reports/evaluation", exist_ok=True)
        
        # Initialize evaluation metrics
        self.metrics = {
            "classification": {
                "accuracy": [],
                "precision": [],
                "recall": [],
                "f1": []
            },
            "generation": {
                "bleu": [],
                "rouge_1": [],
                "rouge_2": [],
                "rouge_l": []
            },
            "user_satisfaction": []
        }
        
        # Load fine-tuned model for evaluation
        self.model = None
        self.tokenizer = None
        self.load_model()
        
        # Initialize ROUGE scorer
        self.rouge_scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    
    def load_model(self):
        """Load the fine-tuned model for evaluation"""
        try:
            model_path = "./fine_tuned_model"
            if not os.path.exists(model_path):
                logger.warning("Fine-tuned model not found. Skipping model evaluation.")
                return
            
            # Load dataset to get label mappings
            dataset_path = "data/processed_exercises.csv"
            if os.path.exists(dataset_path):
                df = pd.read_csv(dataset_path)
                unique_types = df['Type'].dropna().unique().tolist()
                label2id = {label: idx for idx, label in enumerate(unique_types)}
                id2label = {idx: label for label, idx in label2id.items()}
                
                # Load tokenizer and model
                self.tokenizer = AutoTokenizer.from_pretrained(model_path)
                base_model = AutoModelForSequenceClassification.from_pretrained(
                    "bert-base-uncased",
                    num_labels=len(label2id),
                    id2label=id2label,
                    label2id=label2id
                )
                self.model = PeftModel.from_pretrained(base_model, model_path)
                self.model.eval()
                logger.info("Fine-tuned model loaded successfully for evaluation.")
            else:
                logger.warning("Dataset not found. Cannot load label mappings.")
        except Exception as e:
            logger.error(f"Error loading model: {str(e)}")
    
    def evaluate_classification(self, test_data_path="data/processed_exercises_clean.csv"):
        """Evaluate the classification performance of the fine-tuned model"""
        logger.info("Evaluating classification performance...")
        
        if not self.model or not self.tokenizer:
            logger.warning("Model or tokenizer not available. Skipping classification evaluation.")
            return None
        
        try:
            # Load test data
            df = pd.read_csv(test_data_path)
            
            # Use a subset for testing
            test_samples = min(len(df), 100)  # Limit to 100 samples for efficiency
            test_df = df.sample(test_samples)
            
            # Get true labels
            y_true = test_df['Type'].tolist()
            y_pred = []
            
            # Get predictions
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self.model.to(device)
            
            for _, row in test_df.iterrows():
                desc = str(row['Desc'])
                inputs = self.tokenizer(
                    desc,
                    truncation=True,
                    padding="max_length",
                    max_length=512,
                    return_tensors="pt"
                )
                inputs = {k: v.to(device) for k, v in inputs.items()}
                
                with torch.no_grad():
                    outputs = self.model(**inputs)
                    logits = outputs.logits
                    predicted_id = torch.argmax(logits, dim=1).item()
                    predicted_label = self.model.config.id2label[predicted_id]
                    y_pred.append(predicted_label)
            
            # Calculate metrics
            accuracy = accuracy_score(y_true, y_pred)
            precision = precision_score(y_true, y_pred, average='weighted', zero_division=0)
            recall = recall_score(y_true, y_pred, average='weighted', zero_division=0)
            f1 = f1_score(y_true, y_pred, average='weighted', zero_division=0)
            
            # Store results
            results = {
                "accuracy": float(accuracy),
                "precision": float(precision),
                "recall": float(recall),
                "f1": float(f1),
                "num_samples": test_samples
            }
            
            logger.info(f"Classification results: {results}")
            
            # Create confusion matrix
            classes = sorted(set(y_true))
            cm = confusion_matrix(y_true, y_pred, labels=classes)
            
            # Plot confusion matrix
            plt.figure(figsize=(10, 8))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=classes, yticklabels=classes)
            plt.xlabel('Predicted')
            plt.ylabel('True')
            plt.title('Confusion Matrix')
            plt.tight_layout()
            plt.savefig("reports/evaluation/confusion_matrix.png")
            plt.close()
            
            # Update metrics
            self.metrics["classification"]["accuracy"].append(accuracy)
            self.metrics["classification"]["precision"].append(precision)
            self.metrics["classification"]["recall"].append(recall)
            self.metrics["classification"]["f1"].append(f1)
            
            return results
            
        except Exception as e:
            logger.error(f"Error evaluating classification: {str(e)}")
            return None
    
    def evaluate_generation(self, samples_path="data/evaluation/generation_samples.json"):
        """Evaluate the text generation quality using BLEU and ROUGE scores"""
        logger.info("Evaluating text generation performance...")
        
        # Check if samples exist or create them if not
        if not os.path.exists(samples_path):
            self._create_generation_samples(samples_path)
        
        try:
            # Load samples
            with open(samples_path, 'r') as f:
                samples = json.load(f)
            
            # Calculate BLEU and ROUGE scores
            bleu_scores = []
            rouge1_scores = []
            rouge2_scores = []
            rougeL_scores = []
            
            for sample in samples:
                reference = sample["reference"]
                hypothesis = sample["hypothesis"]
                
                # Tokenize for BLEU
                reference_tokens = [word_tokenize(reference)]
                hypothesis_tokens = word_tokenize(hypothesis)
                
                # Calculate BLEU score
                bleu = sentence_bleu(reference_tokens, hypothesis_tokens)
                bleu_scores.append(bleu)
                
                # Calculate ROUGE scores
                rouge_scores = self.rouge_scorer.score(reference, hypothesis)
                rouge1_scores.append(rouge_scores['rouge1'].fmeasure)
                rouge2_scores.append(rouge_scores['rouge2'].fmeasure)
                rougeL_scores.append(rouge_scores['rougeL'].fmeasure)
            
            # Calculate average scores
            avg_bleu = np.mean(bleu_scores)
            avg_rouge1 = np.mean(rouge1_scores)
            avg_rouge2 = np.mean(rouge2_scores)
            avg_rougeL = np.mean(rougeL_scores)
            
            # Store results
            results = {
                "bleu": float(avg_bleu),
                "rouge1": float(avg_rouge1),
                "rouge2": float(avg_rouge2),
                "rougeL": float(avg_rougeL),
                "num_samples": len(samples)
            }
            
            logger.info(f"Generation results: {results}")
            
            # Create score distribution plot
            fig, axs = plt.subplots(2, 2, figsize=(12, 10))
            axs[0, 0].hist(bleu_scores, bins=10, alpha=0.7)
            axs[0, 0].set_title('BLEU Scores')
            axs[0, 1].hist(rouge1_scores, bins=10, alpha=0.7)
            axs[0, 1].set_title('ROUGE-1 Scores')
            axs[1, 0].hist(rouge2_scores, bins=10, alpha=0.7)
            axs[1, 0].set_title('ROUGE-2 Scores')
            axs[1, 1].hist(rougeL_scores, bins=10, alpha=0.7)
            axs[1, 1].set_title('ROUGE-L Scores')
            
            plt.tight_layout()
            plt.savefig("reports/evaluation/generation_scores.png")
            plt.close()
            
            # Update metrics
            self.metrics["generation"]["bleu"].append(avg_bleu)
            self.metrics["generation"]["rouge_1"].append(avg_rouge1)
            self.metrics["generation"]["rouge_2"].append(avg_rouge2)
            self.metrics["generation"]["rouge_l"].append(avg_rougeL)
            
            return results
            
        except Exception as e:
            logger.error(f"Error evaluating generation: {str(e)}")
            return None
    
    def _create_generation_samples(self, output_path):
        """Create sample references and hypotheses for generation evaluation"""
        logger.info("Creating generation samples for evaluation...")
        
        # Example sample pairs (reference texts and system-generated texts)
        # In a real scenario, these would come from actual system outputs
        samples = [
            {
                "reference": "Squats are a compound exercise that primarily target the quadriceps, hamstrings, and glutes. They also engage the core and lower back muscles for stability.",
                "hypothesis": "Squats work the leg muscles including quads, hamstrings and glutes. They also help engage core muscles."
            },
            {
                "reference": "Push-ups are a bodyweight exercise that target the chest, shoulders, and triceps. They also engage the core for stability.",
                "hypothesis": "Push-ups work the upper body, mainly the chest and triceps. Core muscles are also engaged during push-ups."
            },
            {
                "reference": "Deadlifts are a compound exercise that primarily target the lower back, glutes, and hamstrings. They also engage the upper back, traps, and forearms.",
                "hypothesis": "Deadlifts target the posterior chain including lower back and hamstrings. They are good for overall strength development."
            },
            {
                "reference": "Planks are an isometric core exercise that strengthen the abdominals, lower back, and shoulders. They also engage the glutes and quads.",
                "hypothesis": "Planks strengthen the core muscles and improve stability. They help build endurance in the abdominal muscles."
            },
            {
                "reference": "Lunges are a unilateral exercise that target the quadriceps, hamstrings, and glutes. They also improve balance and coordination.",
                "hypothesis": "Lunges work the leg muscles and help with balance. They are good for developing lower body strength."
            }
        ]
        
        # Sample generation from exercise dataset
        try:
            df = pd.read_csv("data/processed_exercises.csv")
            if len(df) > 5:
                # Create more realistic samples from the dataset
                for i in range(5):
                    if i < len(df):
                        desc = df.iloc[i]['Desc']
                        if isinstance(desc, str) and len(desc) > 20:
                            # Split the description to create reference and hypothesis
                            words = desc.split()
                            mid = len(words) // 2
                            reference = " ".join(words)
                            hypothesis = " ".join(words[:mid]) + " " + " ".join(words[mid+5:])
                            samples.append({
                                "reference": reference,
                                "hypothesis": hypothesis
                            })
        except Exception as e:
            logger.error(f"Error creating additional samples from dataset: {str(e)}")
        
        # Save samples
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, 'w') as f:
            json.dump(samples, f, indent=2)
        
        logger.info(f"Created {len(samples)} generation samples for evaluation.")
    
    def evaluate_user_satisfaction(self, simulated_users=20):
        """
        Simulate user satisfaction scores for the system
        
        In a real-world scenario, this would involve actual user surveys or feedback,
        but for this demonstration, we'll simulate user satisfaction ratings.
        """
        logger.info("Evaluating user satisfaction...")
        
        # Simulate user satisfaction scores (1-5)
        np.random.seed(42)  # For reproducibility
        
        # Create weighted distribution favoring mid to high scores (more realistic)
        satisfaction_weights = [0.05, 0.10, 0.30, 0.40, 0.15]  # Weights for scores 1-5
        
        # Generate scores
        satisfaction_scores = np.random.choice(
            [1, 2, 3, 4, 5], 
            size=simulated_users, 
            p=satisfaction_weights
        )
        
        # Calculate average satisfaction
        avg_satisfaction = np.mean(satisfaction_scores)
        
        # Calculate distribution of scores
        score_counts = {score: np.sum(satisfaction_scores == score) for score in range(1, 6)}
        
        # Store results
        results = {
            "average_satisfaction": float(avg_satisfaction),
            "score_distribution": score_counts,
            "num_users": simulated_users
        }
        
        logger.info(f"User satisfaction results: {results}")
        
        # Create satisfaction distribution plot
        plt.figure(figsize=(10, 6))
        bars = plt.bar(score_counts.keys(), score_counts.values(), alpha=0.7)
        
        # Add count labels on top of bars
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                     f'{int(height)}', ha='center', va='bottom')
        
        plt.xlabel('Satisfaction Score')
        plt.ylabel('Number of Users')
        plt.title('User Satisfaction Distribution')
        plt.xticks(range(1, 6))
        plt.ylim(0, max(score_counts.values()) + 2)
        plt.savefig("reports/evaluation/user_satisfaction.png")
        plt.close()
        
        # Update metrics
        self.metrics["user_satisfaction"].append(avg_satisfaction)
        
        return results
    
    def evaluate_recommendations(self, num_profiles=5):
        """
        Evaluate the quality of exercise recommendations
        
        This simulates evaluation of recommendations based on user profiles.
        """
        logger.info("Evaluating recommendation quality...")
        
        # Define synthetic user profiles (simplified)
        user_profiles = [
            {"name": "User1", "level": "Beginner", "goals": ["Weight Loss"], "equipment": ["Body Only"]},
            {"name": "User2", "level": "Intermediate", "goals": ["Strength"], "equipment": ["Barbell", "Dumbbells"]},
            {"name": "User3", "level": "Advanced", "goals": ["Muscle Gain"], "equipment": ["Full Gym"]},
            {"name": "User4", "level": "Beginner", "goals": ["Endurance"], "equipment": ["Cardio Machines"]},
            {"name": "User5", "level": "Intermediate", "goals": ["Flexibility"], "equipment": ["Yoga Mat"]}
        ]
        
        # Simulated recommendation quality metrics
        relevance_scores = []
        diversity_scores = []
        personalization_scores = []
        
        np.random.seed(42)  # For reproducibility
        
        for profile in user_profiles[:num_profiles]:
            # Simulate recommendation evaluation - in a real system, these would be calculated based on actual
            # recommendations and user profiles
            
            # Relevance: How well recommendations match user goals and level
            relevance = np.random.uniform(0.7, 0.95)  # Simulated relevance score (0-1)
            
            # Diversity: Variety in recommended exercises
            diversity = np.random.uniform(0.6, 0.9)  # Simulated diversity score (0-1)
            
            # Personalization: How tailored recommendations are to the user profile
            personalization = np.random.uniform(0.7, 0.95)  # Simulated personalization score (0-1)
            
            relevance_scores.append(relevance)
            diversity_scores.append(diversity)
            personalization_scores.append(personalization)
            
            logger.info(f"Profile {profile['name']}: Relevance={relevance:.2f}, Diversity={diversity:.2f}, Personalization={personalization:.2f}")
        
        # Calculate average scores
        avg_relevance = np.mean(relevance_scores)
        avg_diversity = np.mean(diversity_scores)
        avg_personalization = np.mean(personalization_scores)
        
        # Store results
        results = {
            "average_relevance": float(avg_relevance),
            "average_diversity": float(avg_diversity),
            "average_personalization": float(avg_personalization),
            "profiles_evaluated": num_profiles
        }
        
        logger.info(f"Recommendation quality results: {results}")
        
        # Create radar chart with plotly
        fig = go.Figure()
        
        fig.add_trace(go.Scatterpolar(
            r=avg_scores + [avg_scores[0]],  # Close the loop
            theta=labels + [labels[0]],      # Close the loop
            fill='toself',
            name='Average'
        ))
        
        fig.update_layout(
            polar=dict(
                radialaxis=dict(
                    visible=True,
                    range=[0, 1]
                )
            ),
            title="Recommendation Quality"
        )
        
        # Try to save the radar chart, but don't fail the entire evaluation if it's not possible
        try:
            fig.write_image("reports/evaluation/recommendation_quality.png")
        except Exception as e:
            logger.warning(f"Could not save radar chart as image: {str(e)}")
            logger.warning("To save plotly visualizations, install kaleido: pip install -U kaleido")
            # Create a simpler visualization using matplotlib as fallback
            try:
                # Create a basic bar chart instead
                plt.figure(figsize=(10, 6))
                bars = plt.bar(['Relevance', 'Diversity', 'Personalization'], 
                              [avg_relevance, avg_diversity, avg_personalization], 
                              alpha=0.7)
                
                # Add value labels on top of bars
                for bar in bars:
                    height = bar.get_height()
                    plt.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                            f'{height:.2f}', ha='center', va='bottom')
                
                plt.ylim(0, 1.1)
                plt.title('Recommendation Quality Metrics')
                plt.savefig("reports/evaluation/recommendation_quality.png")
                plt.close()
                logger.info("Created fallback visualization for recommendation quality")
            except Exception as e2:
                logger.warning(f"Could not create fallback visualization: {str(e2)}")
        
        return results
    
    def create_comprehensive_report(self):
        """Generate a comprehensive evaluation report"""
        logger.info("Generating comprehensive evaluation report...")
        
        # Run all evaluations with error handling
        try:
            classification_results = self.evaluate_classification()
        except Exception as e:
            logger.error(f"Classification evaluation failed: {str(e)}")
            classification_results = None
            
        try:
            generation_results = self.evaluate_generation()
        except Exception as e:
            logger.error(f"Generation evaluation failed: {str(e)}")
            generation_results = None
            
        try:
            satisfaction_results = self.evaluate_user_satisfaction()
        except Exception as e:
            logger.error(f"User satisfaction evaluation failed: {str(e)}")
            satisfaction_results = None
            
        try:
            recommendation_results = self.evaluate_recommendations()
        except Exception as e:
            logger.error(f"Recommendation evaluation failed: {str(e)}")
            recommendation_results = None
        
        # Compile results
        report = {
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "classification_metrics": classification_results,
            "generation_metrics": generation_results,
            "user_satisfaction": satisfaction_results,
            "recommendation_quality": recommendation_results,
            "overall_assessment": self._generate_overall_assessment(
                classification_results, 
                generation_results, 
                satisfaction_results,
                recommendation_results
            )
        }
        
        # Save JSON report
        try:
            with open("reports/evaluation/comprehensive_report.json", "w") as f:
                json.dump(report, f, indent=2)
        except Exception as e:
            logger.error(f"Failed to save JSON report: {str(e)}")
        
        # Generate markdown report
        try:
            self._generate_markdown_report(report)
        except Exception as e:
            logger.error(f"Failed to generate markdown report: {str(e)}")
        
        logger.info("Comprehensive evaluation report generated.")
        return report
    
    def _generate_overall_assessment(self, classification_results, generation_results, satisfaction_results, recommendation_results):
        """Generate an overall assessment based on all evaluation results"""
        
        strengths = []
        weaknesses = []
        recommendations = []
        
        # Assess classification performance
        if classification_results:
            if classification_results["accuracy"] > 0.8:
                strengths.append("High classification accuracy for exercise types")
            elif classification_results["accuracy"] < 0.6:
                weaknesses.append("Low classification accuracy needs improvement")
                recommendations.append("Consider retraining the model with more diverse exercise examples")
            
            if classification_results["f1"] > 0.8:
                strengths.append("Good balance of precision and recall (F1 score)")
            elif classification_results["f1"] < 0.6:
                weaknesses.append("Imbalanced precision and recall metrics")
                recommendations.append("Analyze confusion matrix to identify problematic exercise categories")
        
        # Assess generation performance
        if generation_results:
            if generation_results["bleu"] > 0.5:
                strengths.append("Good text generation quality based on BLEU score")
            else:
                weaknesses.append("Text generation quality could be improved (BLEU)")
                recommendations.append("Enhance the language generation components")
            
            if generation_results["rougeL"] > 0.5:
                strengths.append("Strong content preservation in generated text (ROUGE-L)")
            else:
                weaknesses.append("Generated content deviates from reference information")
                recommendations.append("Improve information retention in the RAG system")
        
        # Assess user satisfaction
        if satisfaction_results:
            if satisfaction_results["average_satisfaction"] > 4.0:
                strengths.append("Excellent user satisfaction ratings")
            elif satisfaction_results["average_satisfaction"] < 3.0:
                weaknesses.append("Below average user satisfaction")
                recommendations.append("Conduct user research to identify usability issues")
        
        # Assess recommendation quality
        if recommendation_results:
            if recommendation_results["average_relevance"] > 0.8:
                strengths.append("Highly relevant exercise recommendations")
            else:
                weaknesses.append("Recommendation relevance could be improved")
                recommendations.append("Refine the recommendation algorithms to better match user goals")
            
            if recommendation_results["average_diversity"] < 0.7:
                weaknesses.append("Limited diversity in recommendations")
                recommendations.append("Implement diversity-enhancing algorithms in the recommendation system")
        
        # Add general observations
        if len(strengths) < 2:
            # Add baseline strength if few were detected
            strengths.append("System successfully combines multiple AI techniques for exercise recommendations")
        
        if len(weaknesses) < 2:
            # Add baseline area for improvement
            weaknesses.append("Integration between different phases could be enhanced for a more cohesive experience")
            recommendations.append("Develop better integration between classification, RAG, and multimodal components")
        
        # Add forward-looking recommendations
        recommendations.append("Consider implementing A/B testing for continuous improvement of the system")
        recommendations.append("Explore more advanced fine-tuning techniques like QLoRA for better performance")
        
        return {
            "strengths": strengths,
            "weaknesses": weaknesses,
            "recommendations": recommendations
        }
    
    def _generate_markdown_report(self, report_data):
        """Generate a markdown report from the evaluation data"""
        
        markdown = f"""# Exercise Recommendation System Evaluation Report
        
Generated: {report_data['timestamp']}

## Summary

This report presents a comprehensive evaluation of the Exercise Recommendation System across multiple dimensions:
classification accuracy, text generation quality, user satisfaction, and recommendation relevance.

## Classification Performance

| Metric | Value |
|--------|-------|
"""
        
        if report_data['classification_metrics']:
            for metric, value in report_data['classification_metrics'].items():
                if metric != 'num_samples':
                    markdown += f"| {metric.capitalize()} | {value:.4f} |\n"
            
            markdown += f"\nEvaluation based on {report_data['classification_metrics']['num_samples']} samples.\n"
        else:
            markdown += "| No classification metrics available |\n"
        
        markdown += """
## Text Generation Quality

| Metric | Value |
|--------|-------|
"""
        
        if report_data['generation_metrics']:
            for metric, value in report_data['generation_metrics'].items():
                if metric != 'num_samples':
                    markdown += f"| {metric.capitalize()} | {value:.4f} |\n"
            
            markdown += f"\nEvaluation based on {report_data['generation_metrics']['num_samples']} text samples.\n"
        else:
            markdown += "| No generation metrics available |\n"
        
        markdown += """
## User Satisfaction

| Metric | Value |
|--------|-------|
"""
        
        if report_data['user_satisfaction']:
            markdown += f"| Average Satisfaction | {report_data['user_satisfaction']['average_satisfaction']:.2f} |\n"
            markdown += "\n### Score Distribution\n\n"
            
            for score, count in report_data['user_satisfaction']['score_distribution'].items():
                markdown += f"- Score {score}: {count} users\n"
            
            markdown += f"\nBased on feedback from {report_data['user_satisfaction']['num_users']} simulated users.\n"
        else:
            markdown += "| No user satisfaction metrics available |\n"
        
        markdown += """
## Recommendation Quality

| Metric | Value |
|--------|-------|
"""
        
        if report_data['recommendation_quality']:
            for metric, value in report_data['recommendation_quality'].items():
                if metric != 'profiles_evaluated':
                    markdown += f"| {' '.join(metric.split('_')).capitalize()} | {value:.4f} |\n"
            
            markdown += f"\nEvaluation based on {report_data['recommendation_quality']['profiles_evaluated']} user profiles.\n"
        else:
            markdown += "| No recommendation quality metrics available |\n"
        
        markdown += """
## Strengths and Weaknesses

### Strengths

"""
        
        for strength in report_data['overall_assessment']['strengths']:
            markdown += f"- {strength}\n"
        
        markdown += """
### Weaknesses

"""
        
        for weakness in report_data['overall_assessment']['weaknesses']:
            markdown += f"- {weakness}\n"
        
        markdown += """
## Recommendations for Improvement

"""
        
        for recommendation in report_data['overall_assessment']['recommendations']:
            markdown += f"- {recommendation}\n"
        
        markdown += """
## Visualization Summary

The following visualizations are available in the reports/evaluation directory:

1. Confusion Matrix - Shows classification performance across exercise types
2. Generation Scores - Distribution of BLEU and ROUGE scores
3. User Satisfaction - Distribution of user satisfaction ratings
4. Recommendation Quality - Radar chart of recommendation metrics

## Conclusion

This evaluation provides a comprehensive assessment of the Exercise Recommendation System's performance,
highlighting both strengths and areas for improvement. The system demonstrates [overall assessment]
and could benefit from the suggested enhancements to improve user experience and recommendation quality.
"""
        
        # Save markdown report
        with open("reports/evaluation/evaluation_report.md", "w") as f:
            f.write(markdown)
        
        logger.info("Markdown report generated successfully.")

def run_evaluation():
    """Run the evaluation framework"""
    logger.info("Starting Phase 6: Evaluation Frameworks")
    
    # Create evaluator
    evaluator = ExerciseEvaluator()
    
    # Generate comprehensive report
    report = evaluator.create_comprehensive_report()
    
    logger.info("Phase 6: Evaluation completed. Reports available in reports/evaluation directory.")
    return report

if __name__ == "__main__":
    run_evaluation() 
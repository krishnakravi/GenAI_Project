import tkinter as tk
from tkinter import ttk, scrolledtext, messagebox
import threading
import os
import pandas as pd
import json

# Import the existing project modules
from phase1 import run_phase1
from phase2 import implement_chain_of_thought
from phase3 import build_rag_system
from phase1setup import check_and_install_dependencies

class ExerciseRecommendationApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Personalized Exercise Recommendation System")
        self.root.geometry("900x700")
        self.root.minsize(900, 700)
        
        # Initialize variables
        self.rag_system = None
        self.user_profiles = self.load_user_profiles()
        self.df = None
        
        # Load the exercise data
        self.load_exercise_data()
        
        # Create tabs
        self.tab_control = ttk.Notebook(root)
        
        # Add tabs for each main functionality
        self.setup_tab = ttk.Frame(self.tab_control)
        self.recommend_tab = ttk.Frame(self.tab_control)
        self.qa_tab = ttk.Frame(self.tab_control)
        self.reports_tab = ttk.Frame(self.tab_control)
        
        self.tab_control.add(self.setup_tab, text="Setup")
        self.tab_control.add(self.recommend_tab, text="Recommend Exercises")
        self.tab_control.add(self.qa_tab, text="Exercise Q&A")
        self.tab_control.add(self.reports_tab, text="Reports")
        
        self.tab_control.pack(expand=1, fill="both")
        
        # Initialize tabs
        self.init_setup_tab()
        self.init_recommend_tab()
        self.init_qa_tab()
        self.init_reports_tab()
    
    def load_exercise_data(self):
        try:
            self.df = pd.read_csv('megaGymDataset.csv')
        except Exception as e:
            messagebox.showerror("Error", f"Could not load exercise data: {str(e)}")
    
    def load_user_profiles(self):
        # Start with the predefined profiles from phase2.py
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
        return user_profiles
    
    def init_setup_tab(self):
        frame = ttk.Frame(self.setup_tab, padding=20)
        frame.pack(fill="both", expand=True)
        
        # Welcome message
        welcome_label = ttk.Label(
            frame, 
            text="Welcome to the Personalized Exercise Recommendation System",
            font=("Arial", 14, "bold")
        )
        welcome_label.pack(pady=10)
        
        description = (
            "This system provides personalized exercise recommendations using advanced NLP "
            "and prompt engineering techniques. The system analyzes exercise data, matches "
            "exercises to user profiles, and creates custom workout plans."
        )
        desc_label = ttk.Label(frame, text=description, wraplength=800)
        desc_label.pack(pady=10)
        
        # Setup button
        setup_frame = ttk.Frame(frame)
        setup_frame.pack(pady=20)
        
        setup_button = ttk.Button(
            setup_frame, 
            text="Setup Environment",
            command=self.setup_environment
        )
        setup_button.grid(row=0, column=0, padx=10)
        
        self.setup_status = ttk.Label(setup_frame, text="Status: Not started")
        self.setup_status.grid(row=0, column=1, padx=10)
        
        # Status log
        log_label = ttk.Label(frame, text="Setup Log:")
        log_label.pack(pady=(20, 5), anchor="w")
        
        self.log_text = scrolledtext.ScrolledText(frame, width=80, height=15)
        self.log_text.pack(fill="both", expand=True)
        self.log_text.config(state="disabled")
    
    def init_recommend_tab(self):
        frame = ttk.Frame(self.recommend_tab, padding=20)
        frame.pack(fill="both", expand=True)
        
        # User selection
        user_frame = ttk.LabelFrame(frame, text="Select User Profile", padding=10)
        user_frame.pack(fill="x", pady=10)
        
        ttk.Label(user_frame, text="User:").grid(row=0, column=0, padx=5, pady=5, sticky="w")
        
        self.user_var = tk.StringVar()
        user_combo = ttk.Combobox(user_frame, textvariable=self.user_var, width=30)
        user_combo['values'] = [user["name"] for user in self.user_profiles]
        user_combo.current(0)
        user_combo.grid(row=0, column=1, padx=5, pady=5, sticky="w")
        user_combo.bind("<<ComboboxSelected>>", self.on_user_selected)
        
        # User details
        self.user_details_frame = ttk.LabelFrame(frame, text="User Details", padding=10)
        self.user_details_frame.pack(fill="x", pady=10)
        
        # Recommendation button
        self.recommend_button = ttk.Button(
            frame, 
            text="Get Exercise Recommendations",
            command=self.get_recommendations
        )
        self.recommend_button.pack(pady=10)
        
        # Results frame
        results_frame = ttk.LabelFrame(frame, text="Recommended Exercises", padding=10)
        results_frame.pack(fill="both", expand=True, pady=10)
        
        self.results_text = scrolledtext.ScrolledText(results_frame, width=80, height=15)
        self.results_text.pack(fill="both", expand=True)
        self.results_text.config(state="disabled")
        
        # Initialize with first user
        self.on_user_selected(None)
    
    def init_qa_tab(self):
        frame = ttk.Frame(self.qa_tab, padding=20)
        frame.pack(fill="both", expand=True)
        
        ttk.Label(frame, text="Ask a question about exercises, fitness, or workouts:").pack(anchor="w", pady=(0, 5))
        
        # Question entry
        self.question_var = tk.StringVar()
        question_entry = ttk.Entry(frame, textvariable=self.question_var, width=80)
        question_entry.pack(fill="x", pady=5)
        
        # Example questions
        examples_frame = ttk.LabelFrame(frame, text="Example Questions", padding=10)
        examples_frame.pack(fill="x", pady=10)
        
        examples = [
            "What exercises are best for weight loss?",
            "How should I start exercising as a beginner?",
            "What's the best way to build muscle in my arms?",
            "What should I eat before and after a workout?",
            "Can you recommend a workout routine for building strength?"
        ]
        
        for i, example in enumerate(examples):
            example_btn = ttk.Button(
                examples_frame, 
                text=example,
                command=lambda e=example: self.set_example_question(e)
            )
            example_btn.pack(anchor="w", pady=2)
        
        # Ask button
        ask_button = ttk.Button(
            frame, 
            text="Ask Question",
            command=self.ask_question
        )
        ask_button.pack(pady=10)
        
        # Answer frame
        answer_frame = ttk.LabelFrame(frame, text="Answer", padding=10)
        answer_frame.pack(fill="both", expand=True, pady=10)
        
        self.answer_text = scrolledtext.ScrolledText(answer_frame, width=80, height=15)
        self.answer_text.pack(fill="both", expand=True)
        self.answer_text.config(state="disabled")
    
    def init_reports_tab(self):
        frame = ttk.Frame(self.reports_tab, padding=20)
        frame.pack(fill="both", expand=True)
        
        ttk.Label(frame, text="Generate and View Reports", font=("Arial", 12, "bold")).pack(pady=10)
        
        # Report generation buttons
        buttons_frame = ttk.Frame(frame)
        buttons_frame.pack(pady=20)
        
        phase1_btn = ttk.Button(
            buttons_frame, 
            text="Generate Phase 1 Report",
            command=lambda: self.run_in_thread(run_phase1)
        )
        phase1_btn.grid(row=0, column=0, padx=10, pady=5)
        
        phase2_btn = ttk.Button(
            buttons_frame, 
            text="Generate Phase 2 Report",
            command=self.generate_phase2_report
        )
        phase2_btn.grid(row=0, column=1, padx=10, pady=5)
        
        phase3_btn = ttk.Button(
            buttons_frame, 
            text="Generate Phase 3 Report",
            command=self.generate_phase3_report
        )
        phase3_btn.grid(row=0, column=2, padx=10, pady=5)
        
        # View report buttons
        view_frame = ttk.Frame(frame)
        view_frame.pack(pady=20)
        
        view_phase1_btn = ttk.Button(
            view_frame, 
            text="View Phase 1 Report",
            command=lambda: self.view_report("reports/phase1_summary.txt")
        )
        view_phase1_btn.grid(row=0, column=0, padx=10, pady=5)
        
        view_phase2_btn = ttk.Button(
            view_frame, 
            text="View Phase 2 Report",
            command=lambda: self.view_report("reports/phase2_report.md")
        )
        view_phase2_btn.grid(row=0, column=1, padx=10, pady=5)
        
        view_phase3_btn = ttk.Button(
            view_frame, 
            text="View Phase 3 Report",
            command=lambda: self.view_report("reports/phase3_report.md")
        )
        view_phase3_btn.grid(row=0, column=2, padx=10, pady=5)
        
        # Report viewer
        viewer_frame = ttk.LabelFrame(frame, text="Report Viewer", padding=10)
        viewer_frame.pack(fill="both", expand=True, pady=10)
        
        self.report_text = scrolledtext.ScrolledText(viewer_frame, width=80, height=20)
        self.report_text.pack(fill="both", expand=True)
        self.report_text.config(state="disabled")

        # In the init_reports_tab method, add this after the existing buttons:

        # Add embedding visualization buttons if available
        if os.path.exists("data/embeddings/visualizations"):
            viz_label = ttk.Label(frame, text="View Embedding Visualizations:", font=("Arial", 12, "bold"))
            viz_label.pack(pady=(20,5))
            
            viz_frame = ttk.Frame(frame)
            viz_frame.pack(pady=10)
            
            viz_types = [
                ("Word2Vec (t-SNE)", "data/embeddings/visualizations/word2vec_visualization_tsne.png"),
                ("BERT by Category (t-SNE)", "data/embeddings/visualizations/bert_visualization_by_category_tsne.png"),
                ("Word2Vec (PCA)", "data/embeddings/visualizations/word2vec_visualization_pca.png"),
                ("BERT by Category (PCA)", "data/embeddings/visualizations/bert_visualization_by_category_pca.png"),
                ("Word2Vec (UMAP)", "data/embeddings/visualizations/word2vec_visualization_umap.png"),
                ("BERT by Category (UMAP)", "data/embeddings/visualizations/bert_visualization_by_category_umap.png"),
            ]
            
            viz_row = 0
            viz_col = 0
            
            for label, path in viz_types:
                if os.path.exists(path):
                    viz_btn = ttk.Button(
                        viz_frame,
                        text=label,
                        command=lambda p=path: self.view_visualization(p)
                    )
                    viz_btn.grid(row=viz_row, column=viz_col, padx=10, pady=5)
                    
                    viz_col += 1
                    if viz_col > 2:  # 3 buttons per row
                        viz_col = 0
                        viz_row += 1

        # Add a method to view visualizations
        def view_visualization(self, image_path):
            """Display an image visualization"""
            try:
                # Use a new toplevel window to display the image
                viz_window = tk.Toplevel(self.root)
                viz_window.title("Embedding Visualization")
                
                # Create a frame with scrollbars
                frame = ttk.Frame(viz_window)
                frame.pack(fill="both", expand=True)
                
                # Create scrollbars
                h_scrollbar = ttk.Scrollbar(frame, orient=tk.HORIZONTAL)
                v_scrollbar = ttk.Scrollbar(frame, orient=tk.VERTICAL)
                
                # Create canvas
                canvas = tk.Canvas(frame, width=800, height=600,
                                xscrollcommand=h_scrollbar.set,
                                yscrollcommand=v_scrollbar.set)
                
                # Configure scrollbars
                h_scrollbar.config(command=canvas.xview)
                h_scrollbar.pack(side=tk.BOTTOM, fill=tk.X)
                
                v_scrollbar.config(command=canvas.yview)
                v_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
                
                canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
                
                # Load the image
                from PIL import Image, ImageTk
                img = Image.open(image_path)
                photo = ImageTk.PhotoImage(img)
                
                # Add image to canvas
                canvas.create_image(0, 0, image=photo, anchor=tk.NW)
                canvas.config(scrollregion=canvas.bbox(tk.ALL))
                
                # Keep a reference to avoid garbage collection
                canvas.image = photo
                
            except Exception as e:
                messagebox.showerror("Error", f"Error viewing visualization: {str(e)}")
    
    def run_in_thread(self, func, *args, **kwargs):
        thread = threading.Thread(target=func, args=args, kwargs=kwargs)
        thread.daemon = True
        thread.start()
    
    def setup_environment(self):
        self.setup_status.config(text="Status: Setting up...")
        self.log_text.config(state="normal")
        self.log_text.delete(1.0, tk.END)
        self.log_text.insert(tk.END, "Setting up environment...\n")
        self.log_text.config(state="disabled")
        
        def run_setup():
            try:
                # Create necessary directories
                os.makedirs("data", exist_ok=True)
                os.makedirs("data/processed", exist_ok=True)
                os.makedirs("data/embeddings", exist_ok=True)
                os.makedirs("reports", exist_ok=True)
                os.makedirs("results", exist_ok=True)
                
                # Check dependencies
                check_and_install_dependencies()
                
                # Setup RAG system if needed
                if self.rag_system is None:
                    self.log_text.config(state="normal")
                    self.log_text.insert(tk.END, "Setting up RAG system (this may take a minute)...\n")
                    self.log_text.config(state="disabled")
                    self.rag_system = build_rag_system()
                
                self.root.after(0, lambda: self.setup_status.config(text="Status: Setup completed"))
                self.log_text.config(state="normal")
                self.log_text.insert(tk.END, "Environment setup completed successfully.\n")
                self.log_text.config(state="disabled")
            except Exception as e:
                self.root.after(0, lambda: self.setup_status.config(text="Status: Setup failed"))
                self.log_text.config(state="normal")
                self.log_text.insert(tk.END, f"Error during setup: {str(e)}\n")
                self.log_text.config(state="disabled")
        
        self.run_in_thread(run_setup)
    
    def on_user_selected(self, event):
        # Clear previous details
        for widget in self.user_details_frame.winfo_children():
            widget.destroy()
        
        # Get selected user
        user_name = self.user_var.get()
        user = next((u for u in self.user_profiles if u["name"] == user_name), None)
        
        if user:
            # Display user details
            row = 0
            ttk.Label(self.user_details_frame, text=f"Fitness Level: {user['fitness_level']}").grid(
                row=row, column=0, padx=5, pady=2, sticky="w")
            row += 1
            
            ttk.Label(self.user_details_frame, text=f"Goals: {', '.join(user['goals'])}").grid(
                row=row, column=0, padx=5, pady=2, sticky="w")
            row += 1
            
            ttk.Label(self.user_details_frame, text=f"Equipment: {', '.join(user['equipment_available'])}").grid(
                row=row, column=0, padx=5, pady=2, sticky="w")
            row += 1
            
            ttk.Label(self.user_details_frame, text=f"Time Available: {user['time_available']}").grid(
                row=row, column=0, padx=5, pady=2, sticky="w")
            row += 1
            
            ttk.Label(self.user_details_frame, text=f"Focus Areas: {', '.join(user['preferences']['focus_areas'])}").grid(
                row=row, column=0, padx=5, pady=2, sticky="w")
    
    def get_recommendations(self):
        user_name = self.user_var.get()
        user = next((u for u in self.user_profiles if u["name"] == user_name), None)
        
        if not user:
            messagebox.showerror("Error", "Please select a valid user profile")
            return
        
        self.recommend_button.config(state="disabled", text="Generating Recommendations...")
        self.results_text.config(state="normal")
        self.results_text.delete(1.0, tk.END)
        self.results_text.insert(tk.END, "Generating exercise recommendations...\n")
        self.results_text.config(state="disabled")
        
        def generate_recommendations():
            try:
                # Ensure directories exist
                os.makedirs("data/cot", exist_ok=True)
                
                # Get recommendations with chain of thought
                cot_results = implement_chain_of_thought("megaGymDataset.csv", "data/cot")
                
                # Find results for this user
                user_results = next((r for r in cot_results if r["user"]["id"] == user["id"]), None)
                
                if user_results and "matched_exercises" in user_results:
                    recommendations = user_results["matched_exercises"]
                    
                    self.root.after(0, lambda: self.display_recommendations(recommendations))
                else:
                    self.root.after(0, lambda: self.show_recommendation_error("No recommendations found for this user"))
            except Exception as e:
                self.root.after(0, lambda: self.show_recommendation_error(str(e)))
        
        self.run_in_thread(generate_recommendations)
    
    def display_recommendations(self, recommendations):
        self.results_text.config(state="normal")
        self.results_text.delete(1.0, tk.END)
        
        if not recommendations:
            self.results_text.insert(tk.END, "No recommendations found.")
        else:
            self.results_text.insert(tk.END, f"Top {min(5, len(recommendations))} Recommended Exercises:\n\n")
            
            for i, rec in enumerate(recommendations[:5]):
                if i > 0:
                    self.results_text.insert(tk.END, "\n" + "-"*80 + "\n\n")
                
                self.results_text.insert(tk.END, f"Exercise: {rec['exercise_name']}\n")
                self.results_text.insert(tk.END, f"Match Score: {rec['final_match_score']:.2f}\n")
                self.results_text.insert(tk.END, f"Recommendation: {rec['final_recommendation']}\n\n")
                
                self.results_text.insert(tk.END, "Reasoning Process:\n")
                for step in rec['reasoning_steps']:
                    self.results_text.insert(tk.END, f"Step {step['step']}: {step['output']}\n")
        
        self.results_text.config(state="disabled")
        self.recommend_button.config(state="normal", text="Get Exercise Recommendations")
    
    def show_recommendation_error(self, error_message):
        self.results_text.config(state="normal")
        self.results_text.delete(1.0, tk.END)
        self.results_text.insert(tk.END, f"Error generating recommendations: {error_message}")
        self.results_text.config(state="disabled")
        self.recommend_button.config(state="normal", text="Get Exercise Recommendations")
    
    def set_example_question(self, question):
        self.question_var.set(question)
    
    def ask_question(self):
        question = self.question_var.get().strip()
        
        if not question:
            messagebox.showerror("Error", "Please enter a question")
            return
        
        if self.rag_system is None:
            messagebox.showinfo("Setup Required", "Please set up the environment first (in the Setup tab)")
            return
        
        self.answer_text.config(state="normal")
        self.answer_text.delete(1.0, tk.END)
        self.answer_text.insert(tk.END, "Thinking...\n")
        self.answer_text.config(state="disabled")
        
        def process_question():
            try:
                result = self.rag_system.answer_query(question)
                
                if result and "answer" in result:
                    answer = result["answer"]
                    self.root.after(0, lambda: self.display_answer(answer))
                else:
                    self.root.after(0, lambda: self.display_answer("Sorry, I couldn't find an answer to that question."))
            except Exception as e:
                self.root.after(0, lambda: self.display_answer(f"Error processing question: {str(e)}"))
        
        self.run_in_thread(process_question)
    
    def display_answer(self, answer):
        self.answer_text.config(state="normal")
        self.answer_text.delete(1.0, tk.END)
        self.answer_text.insert(tk.END, answer)
        self.answer_text.config(state="disabled")
    
    def generate_phase2_report(self):
        messagebox.showinfo("Report Generation", "Generating Phase 2 report. This may take a few minutes.")
        
        def run_phase2():
            try:
                from phase2_main import run_phase2
                run_phase2()
                messagebox.showinfo("Success", "Phase 2 report generated successfully!")
            except Exception as e:
                messagebox.showerror("Error", f"Failed to generate Phase 2 report: {str(e)}")
        
        self.run_in_thread(run_phase2)
    
    def generate_phase3_report(self):
        messagebox.showinfo("Report Generation", "Generating Phase 3 report. This may take a few minutes.")
        
        def run_phase3():
            try:
                from phase3 import run_phase3
                run_phase3()
                messagebox.showinfo("Success", "Phase 3 report generated successfully!")
            except Exception as e:
                messagebox.showerror("Error", f"Failed to generate Phase 3 report: {str(e)}")
        
        self.run_in_thread(run_phase3)
    
    def view_report(self, report_path):
        try:
            if not os.path.exists(report_path):
                messagebox.showinfo("Report Not Found", f"Report file not found: {report_path}")
                return
            
            with open(report_path, "r", encoding="utf-8") as f:
                content = f.read()
            
            self.report_text.config(state="normal")
            self.report_text.delete(1.0, tk.END)
            self.report_text.insert(tk.END, content)
            self.report_text.config(state="disabled")
        except Exception as e:
            messagebox.showerror("Error", f"Error viewing report: {str(e)}")

def main():
    root = tk.Tk()
    app = ExerciseRecommendationApp(root)
    root.mainloop()

if __name__ == "__main__":
    main()
import os
import time
import logging
import argparse
from tqdm import tqdm

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def setup_environment():
    """Setup the project environment"""
    # Create necessary directories
    os.makedirs("data", exist_ok=True)
    os.makedirs("data/processed", exist_ok=True)
    os.makedirs("data/embeddings", exist_ok=True)
    os.makedirs("reports", exist_ok=True)
    os.makedirs("results", exist_ok=True)
    
    # Check dependencies (will be handled by phase1setup)
    from phase1setup import check_and_install_dependencies
    check_and_install_dependencies()
    
    print("Environment setup completed successfully.")

def check_megagym_dataset():
    """Check if the megaGymDataset.csv file exists"""
    if not os.path.exists("megaGymDataset.csv"):
        print("\nWARNING: megaGymDataset.csv not found in the current directory!")
        print("Please ensure the dataset file is in the project root directory before continuing.")
        return False
    return True

def display_welcome():
    """Display welcome message"""
    print("\n" + "="*70)
    print(" PERSONALIZED EXERCISE RECOMMENDATION SYSTEM ".center(70, "="))
    print("="*70)
    print("""
This project provides personalized exercise recommendations using advanced NLP and 
prompt engineering techniques. The system analyzes exercise data, matches exercises 
to user profiles, and creates custom workout plans based on fitness goals, 
equipment availability, and experience level.

The project is organized in three phases:
1. Basic NLP Analysis of Exercise Data
2. Advanced Reasoning with Prompt Engineering Techniques
3. Retrieval-Augmented Generation for Exercise Questions
    """)

def run_all_phases():
    """Run all three phases of the project"""
    start_time = time.time()
    
    # Create directories
    os.makedirs("data", exist_ok=True)
    os.makedirs("reports", exist_ok=True)
    os.makedirs("results", exist_ok=True)
    
    # Run Phase 1
    print("\n" + "="*50)
    print("PHASE 1: BASIC NLP ANALYSIS OF EXERCISE DATA")
    print("="*50)
    from phase1_main import run_phase1
    run_phase1()
    
    # Run Phase 2
    print("\n" + "="*50)
    print("PHASE 2: EXERCISE RECOMMENDATION FRAMEWORKS")
    print("="*50)
    from phase2_main import run_phase2
    run_phase2()
    
    # Run Phase 3
    print("\n" + "="*50)
    print("PHASE 3: EXERCISE KNOWLEDGE ASSISTANT")
    print("="*50)
    from phase3 import build_rag_system, demo_rag_queries
    rag_system = build_rag_system()
    demo_results = demo_rag_queries(rag_system)
    
    # Print completion message
    elapsed_time = time.time() - start_time
    print("\n" + "="*50)
    print(f"PROJECT COMPLETED in {elapsed_time:.2f} seconds!")
    print("="*50)
    print("\nResults and reports have been saved in the following directories:")
    print("- Phase 1: data/processed, data/pos_analysis, data/embeddings")
    print("- Phase 2: data/cot, data/tot, data/got")
    print("- Phase 3: data/knowledge_base, data/embeddings, data/faiss, results")
    print("\nReports:")
    print("- Phase 1: reports/phase1_report.md")
    print("- Phase 2: reports/phase2_report.md")
    print("- Phase 3: results/rag_demo_results.json")

def main():
    """Main function"""
    # Setup command line arguments
    parser = argparse.ArgumentParser(description='Personalized Exercise Recommendation System')
    parser.add_argument('--gui', action='store_true', help='Launch the graphical user interface')
    args = parser.parse_args()
    
    # Check if we should run the GUI
    if args.gui:
        # Import and launch the UI
        try:
            from app import main as run_gui
            run_gui()
        except ImportError:
            print("ERROR: Could not import the GUI. Make sure app.py is in the project directory.")
        return
    
    # Otherwise run the command-line version
    display_welcome()
    
    # Setup environment
    print("\nSetting up environment...")
    setup_environment()
    
    # Check for dataset
    if not check_megagym_dataset():
        return
    
    # Run all phases
    print("\nStarting all phases...")
    run_all_phases()

if __name__ == "__main__":
    main()
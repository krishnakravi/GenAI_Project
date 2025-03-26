import os
import nltk
import sys
import subprocess
import pkg_resources

def check_and_install_dependencies():
    """Check and install required dependencies for Phase 1"""
    print("Checking and installing dependencies for Phase 1...")
    
    # Create directories
    os.makedirs('data', exist_ok=True)
    os.makedirs('data/processed', exist_ok=True)
    os.makedirs('data/embeddings', exist_ok=True)
    os.makedirs('reports', exist_ok=True)
    
    # Required packages
    required_packages = [
        'pandas',
        'numpy',
        'nltk',
        'matplotlib',
        'seaborn',
        'scikit-learn',
        'gensim',
        'sentence-transformers',
        'umap-learn'
    ]
    
    # Check and install missing packages
    installed_packages = {pkg.key for pkg in pkg_resources.working_set}
    missing_packages = [pkg for pkg in required_packages if pkg.lower() not in installed_packages]
    
    if missing_packages:
        print(f"Installing missing packages: {', '.join(missing_packages)}")
        subprocess.check_call([sys.executable, '-m', 'pip', 'install', *missing_packages])
    else:
        print("All required packages are already installed.")
    
    # Download required NLTK resources
    nltk_resources = ['punkt', 'stopwords', 'averaged_perceptron_tagger', 'wordnet']
    for resource in nltk_resources:
        try:
            nltk.data.find(f'tokenizers/{resource}')
            print(f"NLTK resource '{resource}' is already downloaded.")
        except LookupError:
            print(f"Downloading NLTK resource: {resource}")
            nltk.download(resource, quiet=True)
    
    print("Setup for Phase 1 completed successfully.")

if __name__ == "__main__":
    check_and_install_dependencies()
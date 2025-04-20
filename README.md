# GenAI Project: Exercise Recommendation System

A comprehensive AI-powered exercise recommendation system that combines NLP, computer vision, and advanced reasoning frameworks to provide personalized fitness guidance.

## Project Overview

This project implements a sophisticated exercise recommendation system using various AI techniques across five phases:

1. **Phase 1**: Basic NLP Analysis of Exercise Data
2. **Phase 2**: Exercise Recommendation Frameworks
3. **Phase 3**: Exercise Knowledge Assistant (RAG)
4. **Phase 4**: Multimodal Agents
5. **Phase 5**: Model Fine-Tuning with LoRA
6. **Phase 6**: Evaluation Frameworks

## Features

### Phase 1: Basic NLP Analysis
- Preprocesses exercise data to extract key features
- Analyzes exercise descriptions using POS tagging
- Generates word embeddings using Word2Vec and BERT
- Produces visualizations of exercise relationships

### Phase 2: Advanced Reasoning Frameworks
- **Chain of Thought (CoT)**:
  - 6-step reasoning process for exercise matching
  - Analyzes exercise characteristics, difficulty, benefits, and techniques
  - Generates personalized recommendations based on user profiles

- **Tree of Thought (ToT)**:
  - Explores multiple workout plan pathways
  - Branches based on fitness goals, experience level, and equipment
  - Creates progressive training plans

- **Graph of Thought (GoT)**:
  - Maps relationships between exercises, equipment, and muscle groups
  - Identifies complementary exercises for related muscle groups
  - Discovers exercise substitutions based on available equipment

### Phase 3: Exercise Knowledge Assistant (RAG)
- Implements a Retrieval-Augmented Generation system
- Uses `all-MiniLM-L6-v2` for semantic search
- FAISS for efficient similarity search
- Custom knowledge base of exercise information
- Generates detailed exercise instructions and recommendations

### Phase 4: Multimodal Agents
- Image recognition using BLIP model (`Salesforce/blip-image-captioning-base`)
- Voice input processing with speech recognition
- Text input handling with advanced NLP
- Unified interface for all input modalities

### Phase 5: Model Fine-Tuning with LoRA
- Fine-tunes `bert-base-uncased` using LoRA
- Low-rank adaptation for efficient training
- Custom configuration for exercise-specific tasks
- Saves fine-tuned model for improved performance

### Phase 6: Evaluation Frameworks
- Comprehensive evaluation of system performance using standard metrics
- Assesses classification accuracy with precision, recall, and F1-Score
- Evaluates text generation quality using BLEU and ROUGE metrics
- Simulates user satisfaction and recommendation quality analysis
- Generates detailed reports with visualizations and recommendations for improvement

## Installation

1. Clone the repository:
   ```bash
   git clone <repository-url>
   cd GenAI_Project
   ```

2. Install dependencies:
   ```bash
   python phase1setup.py
   ```

   This will install required packages:
   - pandas, numpy, nltk
   - matplotlib, seaborn, scikit-learn
   - gensim, sentence-transformers
   - transformers, torch
   - faiss-cpu
   - customtkinter
   - pillow, opencv-python
   - speech_recognition, pytesseract

3. Download required NLTK resources (automatically handled by setup):
   - punkt
   - stopwords
   - averaged_perceptron_tagger
   - wordnet

## Usage

### Running the Complete Project

To run all five phases sequentially:
```bash
python main.py
```

### Running Individual Phases

```bash
# Phase 1: Basic NLP Analysis
python phase1_main.py

# Phase 2: Advanced Reasoning
python phase2_main.py

# Phase 3: RAG System
python phase3.py

# Phase 4: Multimodal Demo
python phase4.py

# Phase 5: Model Fine-Tuning
python phase5.py

# Phase 6: Evaluation
python phase6.py
```

### GUI Interface

For an interactive experience with the recommendation system:
```bash
python app.py
```

## Project Structure

```
GenAI_Project/
├── app.py                  # Main GUI application
├── main.py                 # Project entry point
├── phase1.py              # Basic NLP analysis
├── phase1_main.py         # Phase 1 entry point
├── phase1setup.py         # Setup and dependencies
├── phase2.py              # Chain of Thought implementation
├── phase2_main.py         # Phase 2 entry point
├── phase3.py              # RAG system implementation
├── phase4.py              # Multimodal agents
├── phase5.py              # LoRA fine-tuning
├── phase6.py              # Evaluation frameworks
├── data/                  # Data directory
│   ├── processed/         # Processed data
│   ├── embeddings/        # Word embeddings
│   ├── evaluation/        # Evaluation data
│   ├── cot/              # Chain of Thought results
│   ├── tot/              # Tree of Thought results
│   └── got/              # Graph of Thought results
├── reports/              # Generated reports
│   └── evaluation/       # Evaluation reports and visualizations
└── results/              # Output results
```

## Models Used

1. **BLIP Model**: `Salesforce/blip-image-captioning-base`
   - Used for image captioning and exercise recognition

2. **Sentence Transformer**: `all-MiniLM-L6-v2`
   - Used for semantic search in RAG system

3. **Base Model**: `bert-base-uncased`
   - Fine-tuned using LoRA for exercise-specific tasks

## Future Enhancements

- Integration with more powerful language models
- User profile-based personalization
- Expansion of the knowledge base
- Interactive features for recommendation refinement
- Implementation of QLoRA for better efficiency
- Enhanced evaluation metrics for model performance

## Contributors

- Krishna K R
- Sunav K N
- Anosh P Shroff
- Akshay Kannan


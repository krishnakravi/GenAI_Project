# Personalized Exercise Recommendation System

## Overview

This project provides personalized exercise recommendations using advanced Natural Language Processing (NLP) and prompt engineering techniques. The system analyzes exercise data, matches exercises to user profiles, and creates custom workout plans based on fitness goals, equipment availability, and experience level.

## Project Structure

The project is organized into three phases:

1. **Phase 1: Basic NLP Analysis of Exercise Data**
   - Data preprocessing and cleaning
   - POS (Part-of-Speech) tagging for exercise descriptions
   - Word embeddings using Word2Vec and BERT

2. **Phase 2: Advanced Reasoning with Prompt Engineering Techniques**
   - Chain of Thought (CoT) reasoning for exercise matching
   - Tree of Thought (ToT) for exploring workout decision pathways
   - Graph of Thought (GoT) for mapping exercise relationships

3. **Phase 3: Retrieval-Augmented Generation for Exercise Questions**
   - Knowledge base creation with exercise information
   - FAISS indexing for efficient similarity search
   - RAG system for answering exercise-related questions

## Technologies Used

- **Python**: Core programming language
- **NLP Libraries**:
  - NLTK: For text processing, tokenization, and POS tagging
  - Gensim: For Word2Vec embeddings
  - Sentence Transformers: For BERT embeddings
- **Machine Learning**:
  - scikit-learn: For dimensionality reduction (PCA, t-SNE)
  - UMAP: For non-linear dimensionality reduction
  - FAISS: For efficient similarity search
- **Data Processing**:
  - Pandas: For data manipulation
  - NumPy: For numerical operations
- **Visualization**:
  - Matplotlib & Seaborn: For data visualization
- **GUI**:
  - Tkinter: For the application interface

## Dataset

The project uses the MegaGym dataset (`megaGymDataset.csv`), which contains exercises with descriptions, target muscle groups, equipment requirements, and difficulty levels.

## Installation & Setup

1. Clone the repository:
   ```
   git clone <repository-url>
   cd GenAI_Project
   ```

2. Install dependencies:
   ```
   python phase1setup.py
   ```

   This will install required packages:
   - pandas
   - numpy
   - nltk
   - matplotlib
   - seaborn
   - scikit-learn
   - gensim
   - sentence-transformers
   - umap-learn

3. Download required NLTK resources (automatically handled by setup):
   - punkt
   - stopwords
   - averaged_perceptron_tagger
   - wordnet

## Usage

### Running the Complete Project

To run all three phases sequentially:

```
python main.py
```

### Running Individual Phases

```
# Phase 1: Basic NLP Analysis
python phase1_main.py

# Phase 2: Advanced Reasoning
python phase2_main.py

# Phase 3: Retrieval-Augmented Generation
python phase3.py
```

### GUI Interface

For an interactive experience with the recommendation system:

```
python run_gui.py
```

or

```
python app.py
```

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

### Phase 3: Retrieval-Augmented Generation

- Creates a knowledge base of exercise information
- Uses vector embeddings for semantic search
- Answers questions about:
  - Exercises for specific goals (weight loss, muscle building)
  - Beginner workout recommendations
  - Targeted muscle group training
  - Nutrition for workouts
  - Customized workout routines

## Project Outputs

### Reports

- `reports/phase1_summary.txt`: Summary of NLP analysis results
- `reports/phase2_report.md`: Detailed report on reasoning frameworks
- `reports/phase3_report.md`: RAG system results and sample queries

### Data & Visualizations

- `data/embeddings/visualizations/`: Word2Vec and BERT embedding visualizations
- `data/tot/exercise_decision_tree.png`: Tree of Thought exercise decision paths
- `data/got/exercise_relationships.png`: Graph of exercise relationships

## Sample Results

### Chain of Thought Exercise Matching

The system analyzes user profiles and recommends exercises with detailed reasoning:

```
User: Alex Johnson
Fitness Level: Beginner
Goals: Weight loss, General fitness
Equipment: Body Only, Dumbbells

Top Match Score: 0.70
Recommendation: Good match with some considerations for your fitness level and goals.
```

### RAG System Sample Queries

The system can answer questions like:

1. What exercises are best for weight loss?
2. How should I start exercising as a beginner?
3. What's the best way to build muscle in my arms?
4. What should I eat before and after a workout?
5. Can you recommend a workout routine for building strength?

## Future Enhancements

- Integration with more powerful language models for response generation
- User profile-based personalization
- Expansion of the knowledge base with specialized exercise information
- Interactive features to refine recommendations based on user feedback

## Contributors

- Krishna K R
- Sunav K N
- Anosh P Shroff
- Akshay Kannan


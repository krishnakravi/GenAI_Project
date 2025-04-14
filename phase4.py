import os
import json
import logging
import pandas as pd
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

# Configure logging for debugging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ExerciseKnowledgeBase:
    """Knowledge base for exercise and fitness information"""
    
    def __init__(self, data_path="/Users/Sunav/cprog/sem3_workspace/college/gen ai project/gen ai sunav/GenAI_Project/megaGymDataset.csv"):
        self.data_path = data_path
        self.articles = []
        self.load_exercise_data()
    
    def load_exercise_data(self):
        """Load and process exercise data into knowledge articles"""
        logger.info(f"Loading exercise data from {self.data_path}")
        
        try:
            df = pd.read_csv(self.data_path)
            self._create_exercise_articles(df)
            logger.info(f"Loaded {len(self.articles)} knowledge articles")
        except Exception as e:
            logger.error(f"Error loading data: {str(e)}")
            raise
    
    def _create_exercise_articles(self, df):
        """Convert exercise data into articles"""
        # Group exercises by type and create articles
        exercise_types = df['Type'].unique()
        for ex_type in exercise_types:
            type_exercises = df[df['Type'] == ex_type]
            article = {
                "id": f"type_{ex_type.lower().replace(' ', '_')}",
                "title": f"{ex_type} Exercises",
                "content": f"Information about {ex_type} exercises:\n\n"
            }
            for _, exercise in type_exercises.head(10).iterrows():
                title = exercise['Title'] if pd.notna(exercise['Title']) else "Unnamed Exercise"
                desc = exercise['Desc'] if pd.notna(exercise['Desc']) else "No description."
                article["content"] += f"- {title}: {desc}\n"
            self.articles.append(article)

class RAG:
    """Retrieval-Augmented Generation system"""
    
    def __init__(self, knowledge_base, model_name="all-MiniLM-L6-v2"):
        self.knowledge_base = knowledge_base
        self.model_name = model_name
        self.index = None
        self.articles = knowledge_base.articles
        self.embeddings = None
        self.model = None
        self._initialize_embeddings()
    
    def _initialize_embeddings(self):
        """Set up embeddings and FAISS index"""
        logger.info(f"Initializing embeddings with {self.model_name}")
        try:
            self.model = SentenceTransformer(self.model_name)
            texts = [article["content"] for article in self.articles]
            self.embeddings = self.model.encode(texts)
            dimension = self.embeddings.shape[1]
            self.index = faiss.IndexFlatL2(dimension)
            self.index.add(np.array(self.embeddings).astype('float32'))
            logger.info(f"Created embeddings and index for {len(texts)} articles")
        except Exception as e:
            logger.error(f"Error initializing embeddings: {str(e)}")
            raise
    
    def retrieve_relevant_documents(self, query, k=3):
        """Retrieve top-k relevant documents for a query"""
        query_embedding = self.model.encode([query])
        distances, indices = self.index.search(np.array(query_embedding).astype('float32'), k=k)
        return [self.articles[idx] for idx in indices[0]]
    
    def generate_response(self, query, relevant_docs):
        """Generate a response using retrieved documents"""
        context = "\n\n".join([doc["content"] for doc in relevant_docs])
        # Basic response (can be enhanced with a language model)
        response = f"For your query '{query}', here's what I found:\n\n{context[:500]}..."
        return response
    
    def answer_query(self, query):
        """Process a query and return an answer"""
        logger.info(f"Processing query: {query}")
        try:
            relevant_docs = self.retrieve_relevant_documents(query)
            answer = self.generate_response(query, relevant_docs)
            return {"query": query, "answer": answer}
        except Exception as e:
            logger.error(f"Error processing query: {str(e)}")
            return {"query": query, "answer": "Sorry, I couldn’t process your query."}

def build_rag_system():
    """Initialize and return the RAG system"""
    knowledge_base = ExerciseKnowledgeBase()
    rag = RAG(knowledge_base)
    return rag

if __name__ == "__main__":
    rag_system = build_rag_system()
    query = "What are some strength exercises for legs?"
    response = rag_system.answer_query(query)
    print(response['answer'])
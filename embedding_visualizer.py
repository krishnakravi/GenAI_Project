import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from gensim.models import Word2Vec
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import umap
import seaborn as sns

def load_embeddings(word2vec_path, bert_path):
    """
    Load both Word2Vec and BERT embeddings
    """
    # Load Word2Vec model
    try:
        word2vec_model = Word2Vec.load(word2vec_path)
        print(f"Successfully loaded Word2Vec model from {word2vec_path}")
    except Exception as e:
        print(f"Error loading Word2Vec model: {str(e)}")
        word2vec_model = None
        
    # Load BERT embeddings
    try:
        bert_embeddings = np.load(bert_path)
        print(f"Successfully loaded BERT embeddings from {bert_path}")
    except Exception as e:
        print(f"Error loading BERT embeddings: {str(e)}")
        bert_embeddings = None
        
    return word2vec_model, bert_embeddings

def visualize_word2vec(model, output_dir, n_words=100, method="tsne"):
    """
    Visualize Word2Vec embeddings using dimensionality reduction
    
    Args:
        model: Word2Vec model
        output_dir: Directory to save the visualization
        n_words: Number of words to visualize
        method: Dimensionality reduction method ('tsne', 'pca', or 'umap')
    """
    if model is None:
        print("No Word2Vec model provided")
        return
    
    # Get embedding vectors for visualization
    words = list(model.wv.index_to_key)[:n_words]
    word_vectors = np.array([model.wv[word] for word in words])
    
    # Apply dimensionality reduction
    print(f"Applying {method.upper()} dimensionality reduction on Word2Vec embeddings...")
    
    if method == "tsne":
        reduced_embeddings = TSNE(n_components=2, random_state=42, perplexity=min(30, max(5, n_words // 5))).fit_transform(word_vectors)
    elif method == "pca":
        reduced_embeddings = PCA(n_components=2, random_state=42).fit_transform(word_vectors)
    elif method == "umap":
        reduced_embeddings = umap.UMAP(n_components=2, random_state=42).fit_transform(word_vectors)
    else:
        print(f"Unknown method {method}, using t-SNE")
        reduced_embeddings = TSNE(n_components=2, random_state=42).fit_transform(word_vectors)
    
    # Create visualization
    plt.figure(figsize=(12, 10))
    plt.scatter(reduced_embeddings[:, 0], reduced_embeddings[:, 1], edgecolors='k', c='lightblue', s=100)
    
    # Add labels for words
    for i, word in enumerate(words):
        plt.annotate(word, xy=(reduced_embeddings[i, 0], reduced_embeddings[i, 1]), fontsize=8)
    
    plt.title(f"Word2Vec Embeddings Visualization using {method.upper()}")
    plt.tight_layout()
    
    # Save visualization
    os.makedirs(output_dir, exist_ok=True)
    save_path = f"{output_dir}/word2vec_visualization_{method}.png"
    plt.savefig(save_path, dpi=300)
    plt.close()
    
    print(f"Word2Vec visualization saved to {save_path}")
    return save_path

def visualize_bert_by_category(embeddings, output_dir, categories=None, df_path=None, method="tsne"):
    """
    Visualize BERT embeddings using dimensionality reduction and color code by category
    
    Args:
        embeddings: BERT embeddings array
        output_dir: Directory to save the visualization
        categories: List of categories for each embedding
        df_path: Path to DataFrame with categories (alternative to categories parameter)
        method: Dimensionality reduction method ('tsne', 'pca', or 'umap')
    """
    if embeddings is None:
        print("No BERT embeddings provided")
        return
    
    # Load categories from dataframe if provided
    if categories is None and df_path is not None:
        df = pd.read_csv(df_path)
        if 'Type' in df.columns:
            categories = df['Type'].tolist()
        elif 'BodyPart' in df.columns:
            categories = df['BodyPart'].tolist()
        else:
            print("No category column found in dataframe")
            categories = None
    
    # Apply dimensionality reduction
    print(f"Applying {method.upper()} dimensionality reduction on BERT embeddings...")
    
    if method == "tsne":
        reduced_embeddings = TSNE(n_components=2, random_state=42).fit_transform(embeddings)
    elif method == "pca":
        reduced_embeddings = PCA(n_components=2, random_state=42).fit_transform(embeddings)
    elif method == "umap":
        reduced_embeddings = umap.UMAP(n_components=2, random_state=42).fit_transform(embeddings)
    else:
        print(f"Unknown method {method}, using t-SNE")
        reduced_embeddings = TSNE(n_components=2, random_state=42).fit_transform(embeddings)
    
    # Create visualization
    plt.figure(figsize=(12, 10))
    
    if categories is not None:
        # Only use embeddings that we have categories for
        n_embeddings = min(len(reduced_embeddings), len(categories))
        reduced_embeddings = reduced_embeddings[:n_embeddings]
        categories = categories[:n_embeddings]
        
        # Remove NaN categories
        valid_indices = [i for i, cat in enumerate(categories) if isinstance(cat, str)]
        valid_embeddings = reduced_embeddings[valid_indices]
        valid_categories = [categories[i] for i in valid_indices]
        
        # Create color mapping
        unique_categories = list(set(valid_categories))
        color_map = {cat: i for i, cat in enumerate(unique_categories)}
        colors = [color_map[cat] for cat in valid_categories]
        
        # Create scatter plot with category colors
        scatter = plt.scatter(valid_embeddings[:, 0], valid_embeddings[:, 1], c=colors, cmap='viridis', 
                    alpha=0.7, s=50)
        
        # Add legend
        legend_elements = [plt.Line2D([0], [0], marker='o', color='w', 
                                     markerfacecolor=plt.cm.viridis(color_map[cat]/len(unique_categories)), 
                                     markersize=8, label=cat) 
                          for cat in unique_categories]
        plt.legend(handles=legend_elements, title="Categories", loc="best")
        
    else:
        # Simple scatter plot without categories
        plt.scatter(reduced_embeddings[:, 0], reduced_embeddings[:, 1], alpha=0.7)
    
    plt.title(f"BERT Embeddings Visualization using {method.upper()}")
    plt.tight_layout()
    
    # Save visualization
    os.makedirs(output_dir, exist_ok=True)
    category_type = "by_category" if categories is not None else "no_category"
    save_path = f"{output_dir}/bert_visualization_{category_type}_{method}.png"
    plt.savefig(save_path, dpi=300)
    plt.close()
    
    print(f"BERT visualization saved to {save_path}")
    return save_path

def visualize_embeddings_comparison(word2vec_path, bert_path, tagged_data_path, output_dir):
    """
    Create and save multiple embedding visualizations for comparison
    
    Args:
        word2vec_path: Path to Word2Vec model
        bert_path: Path to BERT embeddings
        tagged_data_path: Path to tagged exercises data
        output_dir: Directory to save visualizations
    """
    print("Loading embeddings for visualization...")
    word2vec_model, bert_embeddings = load_embeddings(word2vec_path, bert_path)
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Generate Word2Vec visualizations
    if word2vec_model is not None:
        print("Generating Word2Vec visualizations...")
        # Use different dimensionality reduction methods
        paths = []
        paths.append(visualize_word2vec(word2vec_model, output_dir, n_words=100, method="tsne"))
        paths.append(visualize_word2vec(word2vec_model, output_dir, n_words=100, method="pca"))
        paths.append(visualize_word2vec(word2vec_model, output_dir, n_words=100, method="umap"))
    
    # Generate BERT visualizations
    if bert_embeddings is not None:
        print("Generating BERT visualizations...")
        # Get categories from the tagged data
        paths = []
        paths.append(visualize_bert_by_category(bert_embeddings, output_dir, df_path=tagged_data_path, method="tsne"))
        paths.append(visualize_bert_by_category(bert_embeddings, output_dir, df_path=tagged_data_path, method="pca"))
        paths.append(visualize_bert_by_category(bert_embeddings, output_dir, df_path=tagged_data_path, method="umap"))
    
    print("Embedding visualizations completed!")
    return output_dir

if __name__ == "__main__":
    # Test the visualization functions
    visualize_embeddings_comparison(
        "data/embeddings/word2vec_model",
        "data/embeddings/bert_embeddings.npy",
        "data/tagged_exercises.csv",
        "data/embeddings/visualizations"
    )
import numpy as np
from gensim.models import KeyedVectors
from itertools import combinations

def load_word_vectors(model_path):
    """Load pre-trained Word2Vec model from .model file."""
    return KeyedVectors.load(model_path)

def preprocess_words(words):
    """Preprocess the input words (tokenization, lemmatization)."""
    return [word.lower() for word in words]

def get_word_embeddings(words, model):
    """Get the embeddings for the provided words."""
    embeddings = []
    for word in words:
        try:
            embeddings.append(model[word])
        except KeyError:
            print(f"Word '{word}' not found in the model, using zeros as embedding.")
            embeddings.append(np.zeros(model.vector_size))  # Fallback for unknown words
    return np.array(embeddings)

def calculate_average_similarity(embedding_group):
    """Calculate the average pairwise cosine similarity for a group of word embeddings."""
    similarity_matrix = np.inner(embedding_group, embedding_group) / (np.linalg.norm(embedding_group, axis=1).reshape(-1, 1) * np.linalg.norm(embedding_group, axis=1))
    num_pairs = len(embedding_group) * (len(embedding_group) - 1) / 2
    return np.sum(np.triu(similarity_matrix, 1)) / num_pairs

def group_similar_words(embeddings, words):
    """Group words into groups of 4 based on highest similarity."""
    groups = []
    
    remaining_indices = list(range(len(words)))  # Keep track of indices to avoid repetition
    
    while len(remaining_indices) >= 4:
        # Generate all possible combinations of 4 indices
        combinations_of_4 = list(combinations(remaining_indices, 4))
        
        best_group = None
        best_similarity = -float('inf')
        
        # Iterate through combinations and calculate their average similarity
        for combo in combinations_of_4:
            combo_embeddings = embeddings[list(combo)]
            avg_similarity = calculate_average_similarity(combo_embeddings)
            
            if avg_similarity > best_similarity:
                best_similarity = avg_similarity
                best_group = combo
        
        # Once the best group is found, add the words to the groups list
        groups.append([words[i] for i in best_group])
        
        # Remove those indices from the remaining pool
        remaining_indices = [i for i in remaining_indices if i not in best_group]
    
    # If there are any remaining words, just add them as a final group (if necessary)
    if remaining_indices:
        groups.append([words[i] for i in remaining_indices])
    
    return groups

def main():
    # Load the Word2Vec model
    model_path = '/Users/romeerpillay/Downloads/word2vec-google-news-300.model'  # Adjust this path
    model = load_word_vectors(model_path)

    # Example list of 16 words
    words = [
        "dinner", "dimple", "dollop", "drop",
        "daisy", "dash", "ding", "divot",
        "dent", "dale", "doc", "due",
        "delivery", "dab", "dream", "dory"
    ]
    
    # Preprocess words
    preprocessed_words = preprocess_words(words)

    # Get word embeddings
    embeddings = get_word_embeddings(preprocessed_words, model)

    # Group words into 4 groups based on similarity
    grouped_words = group_similar_words(embeddings, preprocessed_words)

    # Print the results
    print("Grouped Words:")
    for idx, group in enumerate(grouped_words):
        print(f"Group {idx + 1}: {', '.join(group)}")

if __name__ == "__main__":
    main()

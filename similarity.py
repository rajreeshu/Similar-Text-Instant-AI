from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

def find_similar_texts(old_data, new_data, top_n):
    vectorizer = TfidfVectorizer()
    combined_texts = old_data + new_data
    vectorized_texts = vectorizer.fit_transform(combined_texts)
    similarities = cosine_similarity(vectorized_texts[len(old_data):], vectorized_texts[:len(old_data)])
    results = []
    for idx, sim_scores in enumerate(similarities):
        top_indices = sim_scores.argsort()[-top_n:][::-1]
        results.append({"new_text": new_data[idx], "similar_texts": [(old_data[i], sim_scores[i]) for i in top_indices]})
    return results

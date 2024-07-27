from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

def combine_entities(entities_dict):
    combined_entities = {}
    for key, values in entities_dict.items():
        combined_entities[key] = ' '.join(values)
    return combined_entities

def calculate_cosinesim(cv_text, extracted_entities):
    vectorizer = TfidfVectorizer()

    combined_extracted = combine_entities(extracted_entities)

    extracted_text = ' '.join(combined_extracted.values())

    vectorizer.fit([cv_text, extracted_text])
    cv_vector = vectorizer.transform([cv_text])
    extracted_vector = vectorizer.transform([extracted_text])

    similarity = cosine_similarity(cv_vector, extracted_vector)[0][0]
    
    return similarity

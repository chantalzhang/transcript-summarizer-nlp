from rake_nltk import Rake
from keybert import KeyBERT

def generate_reference_keywords(file_path, num_keywords=20):
    with open(file_path, 'r', encoding='utf-8') as file:
        text = file.read()

    # TF-IDF Keywords
    tfidf_keywords = extract_keywords_tfidf(text, num_keywords)

    # RAKE Keywords
    rake = Rake()
    rake.extract_keywords_from_text(text)
    rake_keywords = rake.get_ranked_phrases()[:num_keywords]

    # KeyBERT Keywords
    kw_model = KeyBERT('all-MiniLM-L6-v2')
    keybert_keywords = [kw[0] for kw in kw_model.extract_keywords(text, top_n=num_keywords)]

    reference_keywords = list(set(tfidf_keywords + rake_keywords + keybert_keywords))
    return reference_keywords

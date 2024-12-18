import re
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation as LDA
import nltk

# nltk.download('punkt')
# nltk.download('stopwords')
# nltk.download('wordnet')

class PreProcessor:
    def __init__(self):
        self.lemmatizer = WordNetLemmatizer()
        self.stop_words = set(stopwords.words('english'))

        # Ignoring common phrases like "good morning"
        self.ignore_phrases = [
            "good morning", "hello class", "hey everyone", "today we will", 
            "welcome to", "let's get started", "sorry"
        ]
        self.ignore_pattern = re.compile(rf"\b({'|'.join(self.ignore_phrases)})\b", re.IGNORECASE)

    def noise_removal(self, text):
        # Removes filler words, repeated phrases, and speech artifacts
        filler_words = ["uh", "um", "yeah", "okay", "alright", "so"]
        filler_pattern = re.compile(rf"\b({'|'.join(filler_words)})\b", re.IGNORECASE)
        text = filler_pattern.sub("", text)  # Remove filler words
        text = re.sub(r"\b(\w+)( \1\b)+", r"\1", text)  # Remove repeated words
        text = self.ignore_pattern.sub("", text)  # Remove ignore phrases
        return text

    def sentence_segmentation(self, text):
        # Splits the text into sentences
        return sent_tokenize(text)

    def remove_stopwords(self, tokens):
        # Removes stopwords
        return [word for word in tokens if word.lower() not in self.stop_words]

    def lemmatize_tokens(self, tokens):
        # Lemmatizes tokens
        return [self.lemmatizer.lemmatize(word) for word in tokens]

    def stemming(self, tokens):
        # Stems tokens
        from nltk.stem.porter import PorterStemmer
        stemmer = PorterStemmer()
        return [stemmer.stem(word) for word in tokens]

    def simplify_sentences(self, sentences):
        # Simplifies sentences by removing stopwords and lemmatizing
        simplified = []
        for sentence in sentences:
            tokens = word_tokenize(sentence)
            tokens = self.remove_stopwords(tokens)
            tokens = self.lemmatize_tokens(tokens)
            simplified.append(" ".join(tokens))
        return simplified

    def extract_keywords(self, text, num_keywords=10):
        # Extracts important keywords using TF-IDF
        vectorizer = TfidfVectorizer(max_features=num_keywords, stop_words='english')
        tfidf_matrix = vectorizer.fit_transform([text])
        return vectorizer.get_feature_names_out()

    def topic_based_grouping(self, sentences):
        # Groups sentences into topics using LDA
        vectorizer = TfidfVectorizer(stop_words='english')
        tfidf_matrix = vectorizer.fit_transform(sentences)
        lda = LDA(n_components=2, random_state=42)
        lda.fit(tfidf_matrix)
        topics = {f"Topic {i+1}": [] for i in range(lda.n_components)}
        topic_assignments = lda.transform(tfidf_matrix).argmax(axis=1)
        for i, topic in enumerate(topic_assignments):
            topics[f"Topic {topic+1}"].append(sentences[i])
        return topics

    def clean_text(self, text):
        # Cleans text by removing special characters and extra whitespace
        text = re.sub(r"[^a-zA-Z0-9 .,!?'\n]", "", text)  # Remove special characters
        text = re.sub(r"\s+", " ", text).strip()  # Remove extra whitespace
        return text.lower()


# PIPELINE A: Minimalist Preprocessing for Embedding Evaluation
def pipeline_a(file_path):
    preprocessor = PreProcessor()
    with open(file_path, 'r', encoding='utf-8') as file:
        text = file.read()

    # Minimalist Preprocessing: Only noise removal and sentence segmentation
    text = preprocessor.noise_removal(text)
    sentences = preprocessor.sentence_segmentation(text)

    # Pass raw sentences to embedding model or further analysis
    return {"raw_sentences": sentences}


# PIPELINE B: Minimal Preprocessing with TF-IDF Keywords
def pipeline_b(file_path):
    preprocessor = PreProcessor()
    with open(file_path, 'r', encoding='utf-8') as file:
        text = file.read()

    # Standard Preprocessing
    text = preprocessor.noise_removal(text)
    sentences = preprocessor.sentence_segmentation(text)

    # Stopword Removal
    tokens = word_tokenize(" ".join(sentences))
    tokens_no_stopwords = preprocessor.remove_stopwords(tokens)

    # TF-IDF Keyword Extraction
    keywords = preprocessor.extract_keywords(" ".join(tokens_no_stopwords))

    return {"keywords": keywords, "sentences": sentences}


# PIPELINE C: Semantic Preprocessing with Lemmatization and Topic Grouping
def pipeline_c(file_path):
    preprocessor = PreProcessor()
    with open(file_path, 'r', encoding='utf-8') as file:
        text = file.read()

    # Standard Preprocessing
    text = preprocessor.noise_removal(text)
    sentences = preprocessor.sentence_segmentation(text)

    # Simplify Sentences (Stopword Removal + Lemmatization)
    simplified_sentences = preprocessor.simplify_sentences(sentences)

    # Topic-Based Grouping
    topics = preprocessor.topic_based_grouping(simplified_sentences)

    return {"topics": topics, "simplified_sentences": simplified_sentences}


# PIPELINE D: Aggressive Preprocessing with Stemming and TF-IDF
def pipeline_d(file_path):
    preprocessor = PreProcessor()
    with open(file_path, 'r', encoding='utf-8') as file:
        text = file.read()

    # Standard Preprocessing
    text = preprocessor.noise_removal(text)
    sentences = preprocessor.sentence_segmentation(text)

    # Stemming
    tokens = word_tokenize(" ".join(sentences))
    tokens_stemmed = preprocessor.stemming(tokens)

    # TF-IDF Keyword Extraction
    keywords = preprocessor.extract_keywords(" ".join(tokens_stemmed))

    return {"keywords": keywords, "stemmed_tokens": tokens_stemmed}


# PIPELINE E: Contextual Preprocessing for Structured Themes
def pipeline_e(file_path):
    preprocessor = PreProcessor()
    with open(file_path, 'r', encoding='utf-8') as file:
        text = file.read()

    # Preprocessing: Noise removal + Sentence Segmentation
    text = preprocessor.noise_removal(text)
    sentences = preprocessor.sentence_segmentation(text)

    # Simplify Sentences (Stopword Removal + Lemmatization)
    simplified_sentences = preprocessor.simplify_sentences(sentences)

    # Remove special characters for cleaner input
    cleaned_text = preprocessor.clean_text(" ".join(simplified_sentences))

    # Group into topics
    topics = preprocessor.topic_based_grouping(simplified_sentences)

    # Extract TF-IDF Keywords from cleaned text
    keywords = preprocessor.extract_keywords(cleaned_text)

    return {"topics": topics, "keywords": keywords, "simplified_sentences": simplified_sentences}

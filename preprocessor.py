import re
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation as LDA
import nltk

class PreProcessor:
# nltk.download('punkt')
# nltk.download('stopwords')
# nltk.download('wordnet')
    def __init__(self):
        self.lemmatizer = WordNetLemmatizer()
        self.stop_words = set(stopwords.words('english'))

        # Ignoring Common Phrases and Sorry. He says Sorry a lot
        self.ignore_phrases = [
            "good morning", "hello class", "hey everyone", "today we will", "welcome to", "let's get started", "Sorry"
        ]
        self.ignore_pattern = re.compile(rf"\b({'|'.join(self.ignore_phrases)})\b", re.IGNORECASE)

    # Removes filler words, phrases repeated, speech artifacts
    def noise_removal(self,text):
        filler_words = ["uh", "um", "yeah", "okay", "alright", "so"]
        filler_pattern = re.compile(rf"\b({'|'.join(filler_words)})\b", re.IGNORECASE)
        text = filler_pattern.sub("", text) # Removes filler words
        text = re.sub(r"\b(\w+)( \1\b)+", r"\1", text)# Remove repeated words or phrases
        text = self.ignore_pattern.sub("", text)
        return text

    def sentence_segmentation(self,text):
        return sent_tokenize(text)

    def remove_stopwords(self,tokens):
        return [word for word in tokens if word.lower() not in self.stop_words]

    def lemmatize_tokens(self,tokens):
        return [self.lemmatizer.lemmatize(word) for word in tokens]

    def stemming(self, tokens):
        from nltk.stem.porter import PorterStemmer
        stemmer = PorterStemmer()
        return [stemmer.stem(word) for word in tokens]

    # Simplifying complex sentences by focussing on main clauses
    def simplify_sentences(self, sentences):
        simplified = []
        for sentence in sentences:
            # Just tokenising and keeping meaningful words
            tokens = word_tokenize(sentence)
            tokens = self.remove_stopwords(tokens)
            tokens = self.lemmatize_tokens(tokens)
            simplified.append(" ".join(tokens))
        return simplified

    # Extracting key terms using TF-IDF. Do not try Gensim or Spacy together they always have dependency conflict 
    def extract_keywords(self, text, num_keywords=10):
        vectorizer = TfidfVectorizer(max_features=num_keywords, stop_words='english')
        tfidf_matrix = vectorizer.fit_transform([text])
        return vectorizer.get_feature_names_out()

    # Cluster sentences into topics using Latent Dirichlet Allocation
    # ^^ Could be a Talking point for report 
    def topic_based_grouping(self, sentences):
        vectorizer = TfidfVectorizer(stop_words='english')
        tfidf_matrix = vectorizer.fit_transform(sentences)
        # Training LDA model and grouping sentences by topic
        lda = LDA(n_components=2, random_state=42)
        lda.fit(tfidf_matrix)
        topics = {f"Topic {i+1}": [] for i in range(lda.n_components)}
        topic_assignments = lda.transform(tfidf_matrix).argmax(axis=1)
        for i, topic in enumerate(topic_assignments):
            topics[f"Topic {topic+1}"].append(sentences[i])

        return topics

    def clean_text(self, text):
        text = re.sub(r"[^a-zA-Z0-9 .,!?'\n]", "", text) 
        text = re.sub(r"\s+", " ", text).strip()
        return text.lower()

    # Preprocessing Pipeline
    def preprocess_transcript(self, file_path):
        with open(file_path, 'r', encoding='utf-8') as file:
            text = file.read()
        text = self.noise_removal(text)
        sentences = self.sentence_segmentation(text)
        simplified_sentences = self.simplify_sentences(sentences)
        keyword_list = self.extract_keywords(" ".join(simplified_sentences))
        thematic_sections = self.topic_based_grouping(simplified_sentences)
        cleaned_text = self.clean_text(" ".join(simplified_sentences))
        return {
            "keywords": keyword_list, "thematic_sections": thematic_sections, "cleaned_text": cleaned_text
        }
   

# IMPLEMENTED PREPROCESSING TECHNIQUES
"""
1. Noise Removal to remove filler words, repeated phrases and a lot of Sorries
2. Sentence Segmentation to split the input text into individual sentences
3. Stopword pretty standard
4. Lemmatisation p standard again
5. Stemming
6. TF-IDF to extract important keywords, could be tweaked
7. Topic Based Grouping by clustering sentences intp topics using LDA. Not sure if Danielle meant something like this when she talked about chunks
8. Removing Special Chars
"""   

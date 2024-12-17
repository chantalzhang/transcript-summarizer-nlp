import re
from nltk.tokenize import sent_tokenize
from transformers import AutoTokenizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation as LDA

class PreProcessor:
    def __init__(self):
        # Tokenizer for token count validation
        self.tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased") # no need to use api key for just token limit counting 

        self.filler_words = [
            # Hesitation markers and phrases that he uses a lot
            "uh", "um", "yeah", "okay", "alright", "so", "sorry",
            "like", "you know", "i mean", "kind of", "sort of",
            "basically", "actually", "pretty much",
            "hmm", "err", "well", "let's see", "let me think",
            
            # Lecture-specific fillers
            "moving on", "next", "now", "right", "anyway",
            "as i said", "as i mentioned", "like i said"
        ]
        
        self.ignore_phrases = [
            # Lecture openings/transitions
            "good morning", "let me share my screen", "hello class",
            "hey everyone", "today we will", "welcome to",
            "let's get started", "can everyone see my screen",
            "before we begin", "let's move forward",

            # Class management unrelated to content
            "any questions so far", "does that make sense",
            "raise your hand if", "let me know if",
            "is everyone following", "can everyone hear me",
            "we have time for", "in the remaining time",
            "we'll cover this next time", "in our next lecture",
            
            # Recap phrases
            "as we discussed last time", "remember from last class",
            "just to recap", "to summarize what we covered"
        ]
        
        self.ignore_pattern = re.compile(rf"\b({'|'.join(self.ignore_phrases)})\b", re.IGNORECASE)

    def noise_removal(self, text):
        """ Removes filler words, repeated words/phrases, and ignored phrases. """
        filler_pattern = re.compile(rf"\b({'|'.join(self.filler_words)})\b", re.IGNORECASE)
        text = filler_pattern.sub("", text)  # Remove filler words
        text = re.sub(r"\b(\w+)( \1\b)+", r"\1", text)  # Remove repeated words
        text = self.ignore_pattern.sub("", text)  # Remove ignored phrases
        return text

    def split_by_sentences(self, text):
        """ Splits text into sentences. """
        return sent_tokenize(text)

    def split_by_fixed_sentences(self, sentences, chunk_size):
        """ Split sentences into fixed-size chunks. """
        chunks = [sentences[i:i + chunk_size] for i in range(0, len(sentences), chunk_size)]
        return [" ".join(chunk) for chunk in chunks]

    # check token limit, only really necessary if we switch back to smaller model such as the hugging one with a 512 token limit
    def check_token_limit(self, chunks, max_tokens=8191): # text-embedding-3-small model has a limit of 8191 tokens per input
        """ Ensures no chunk exceeds the token limit by splitting further if needed. """
        valid_chunks = []
        for chunk in chunks:
            tokens = self.tokenizer.tokenize(chunk)
            if len(tokens) <= max_tokens:
                valid_chunks.append(chunk)
            else:
                # Split further by sentences if over the limit
                split_sentences = sent_tokenize(chunk)
                temp_chunk = ""
                for sentence in split_sentences:
                    if len(self.tokenizer.tokenize(temp_chunk + " " + sentence)) <= max_tokens:
                        temp_chunk += " " + sentence
                    else:
                        if temp_chunk:
                            valid_chunks.append(temp_chunk.strip())
                        temp_chunk = sentence
                if temp_chunk:
                    valid_chunks.append(temp_chunk.strip())
        return valid_chunks

def pipeline_sentence_level(file_path):
    """ Noise removal + sentence splitting. """
    preprocessor = PreProcessor()
    with open(file_path, 'r', encoding='utf-8') as file:
        text = file.read()
    text = preprocessor.noise_removal(text)
    sentences = preprocessor.split_by_sentences(text)
    return sentences

def pipeline_fixed_chunks(file_path, chunk_size):
    """ Noise removal + fixed sentence chunking + token limit enforcement. """
    preprocessor = PreProcessor()
    with open(file_path, 'r', encoding='utf-8') as file:
        text = file.read()
    text = preprocessor.noise_removal(text)
    sentences = preprocessor.split_by_sentences(text)
    chunks = preprocessor.split_by_fixed_sentences(sentences, chunk_size)
    valid_chunks = preprocessor.check_token_limit(chunks)
    return valid_chunks

def pipeline_topic_based(file_path, n_topics=5):
    """ Noise removal + topic-based grouping with token limit enforcement. """
    preprocessor = PreProcessor()
    with open(file_path, 'r', encoding='utf-8') as file:
        text = file.read()
    text = preprocessor.noise_removal(text)
    sentences = preprocessor.split_by_sentences(text)

    # LDA-based topic grouping
    vectorizer = TfidfVectorizer(stop_words='english')
    tfidf_matrix = vectorizer.fit_transform(sentences)
    lda = LDA(n_components=n_topics, random_state=42)
    lda.fit(tfidf_matrix)

    topics = {f"Topic {i+1}": [] for i in range(n_topics)}
    topic_assignments = lda.transform(tfidf_matrix).argmax(axis=1)
    for i, topic in enumerate(topic_assignments):
        topics[f"Topic {topic+1}"].append(sentences[i])

    # Enforce token limits per topic
    valid_topics = {}
    for topic, sentences in topics.items():
        chunks = [" ".join(sentences)]
        valid_topics[topic] = preprocessor.check_token_limit(chunks)
    return valid_topics

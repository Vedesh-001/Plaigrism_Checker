import nltk
from nltk.corpus import wordnet
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import string

# Download necessary NLTK data files
nltk.download('punkt')
nltk.download('wordnet')

def preprocess_text(text):
    # Convert text to lowercase and remove punctuation
    text = text.lower().translate(str.maketrans('', '', string.punctuation))
    # Tokenize text
    tokens = nltk.word_tokenize(text)
    return ' '.join(tokens)  # Return as a string for vectorizer

def extract_keywords(text):
    # Preprocess text
    preprocessed_text = preprocess_text(text)
    # Use TF-IDF for keyword extraction, with a max of 10 keywords
    tfidf = TfidfVectorizer(max_features=10)
    tfidf.fit_transform([preprocessed_text])
    keywords = tfidf.get_feature_names_out()
    return keywords

def find_synonyms(word):
    synonyms = set()
    for syn in wordnet.synsets(word):
        for lemma in syn.lemmas():
            synonyms.add(lemma.name())
    return synonyms

def calculate_similarity(text1, text2):
    # Preprocess both texts
    processed_text1 = preprocess_text(text1)
    processed_text2 = preprocess_text(text2)

    # Vectorize texts using TF-IDF
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform([processed_text1, processed_text2])

    # Calculate cosine similarity
    similarity = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])
    return similarity[0][0]

def check_plagiarism(text_to_check, corpus, threshold=0.7):
    # Extract keywords from the text to check (optional for additional analysis).5
    keywords_to_check = extract_keywords(text_to_check)
    print(f"Extracted Keywords: {keywords_to_check}")

    # Calculate similarity against each document in the corpus
    for i, doc in enumerate(corpus):
        similarity = calculate_similarity(text_to_check, doc)
        if similarity >= threshold:
            print(f"Potential plagiarism detected in document {i+1} with a similarity score of {similarity:.2f}")
        else:
            print(f"Document {i+1} similarity score is {similarity:.2f} - below threshold.")

# User inputs
text_to_check = input("Enter the text to check for plagiarism: ")

# Corpus input - enter documents directly
corpus = []
while True:
    doc = input("Enter a document for the corpus (or type 'done' to finish): ")
    if doc.lower() == 'done':
        break
    corpus.append(doc)

# Check for plagiarism
if len(corpus) > 0:
    threshold = input("Enter similarity threshold (default is 0.7): ")
    threshold = float(threshold) if threshold else 0.7  # Use default threshold if not provided
    check_plagiarism(text_to_check, corpus, threshold)
else:
    print("No documents in the corpus to check against.")

# Plaigrism_Checker
Plagiarism checker

# Preprocessing Text: Step1

Convert the text to lowercase.
Remove punctuation.
Tokenize the text into individual words.
Keyword Extraction: Step2

# 1.Use TF-IDF (Term Frequency-Inverse Document Frequency) to identify the most significant keywords in the text.

# Synonym Extraction: Step3

Use the WordNet corpus from NLTK to find synonyms of each keyword.

# Similarity Calculation: Step4

Use TF-IDF vectorization and cosine similarity to compute the similarity between the text to be checked and documents in the corpus.

# Threshold Check: step5

Compare the similarity score with a predefined threshold (e.g., 0.7). If the similarity is above the threshold, flag it as potential plagiarism.

How to Use This Code

Text to Check: Input the text you want to check for plagiarism in text_to_check.

# Corpus: The corpus consists of a list of documents against which the text will be compared. Threshold: Adjust the threshold according to your needs.

# ** Importing Libraries**

# NLTK: A library for natural language processing. We use it for text preprocessing, tokenization, and accessing wordnet for synonyms.

# wordnet: A lexical database of English words provided by NLTK, useful for finding synonyms.

# TfidfVectorizer: Part of sklearn, this tool converts text data into numerical vectors using the Term Frequency-Inverse Document Frequency (TF-IDF) method.

# cosine_similarity: A method to compute the cosine similarity between two vectors, which measures the similarity between them.

# string: A standard Python module used here to handle string operations, particularly for removing punctuation.

Downloading NLTK Data Files These lines ensure that the necessary datasets (punkt(will make it into a list) for tokenization and wordnet for synonyms) are downloaded and available for use.

# Text Preprocessing Function: Lowercasing: Converts all text to lowercase to ensure consistency.

# Remove Punctuation: Strips out punctuation marks to focus purely on the words.

# Tokenization: Splits the text into individual words (tokens) for further processing.

Keyword Extraction Function Preprocessing: The text is first preprocessed to clean and tokenize it.

# TF-IDF Vectorization: The text is vectorized using TF-IDF, which measures how important a word is in relation to the document and the corpus. max_features=10 limits the extraction to the 10 most important words (keywords).

Synonym Finder Function WordNet Synonyms: This function looks up the synonyms of a given word using WordNet. It returns a set of synonyms, which might be used for expanding the search or analysis later.

Similarity Calculation Function Preprocessing: Both texts are cleaned and tokenized.

# Vectorization: The cleaned texts are converted into numerical vectors using TF-IDF.

# Cosine Similarity: The similarity between the two vectors is calculated. Cosine similarity measures the cosine of the angle between two vectors, giving a value between -1 and 1, where 1 means identical.

Plagiarism Checking Function

# Extract Keywords: Extracts the most significant words from the text to be checked.

# Similarity Check: Iterates over each document in the corpus, calculates the similarity between the text to be checked and the current document, and compares it to the threshold.

# Threshold: If the similarity score is above the threshold (default is 0.7), the text is flagged as potential plagiarism.

User Input and Execution

# Text to Check: The user is prompted to input the text they want to check for plagiarism.

# Corpus Size: The user is asked how many documents are in the corpus.

# Document Inputs: The user is prompted to input each document in the corpus.

# Plagiarism Check: The script then checks the user-provided text against each document in the corpus.

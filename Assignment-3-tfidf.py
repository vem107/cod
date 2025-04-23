# pip install scikit-learn nltk

import nltk
from nltk.tokenize import word_tokenize
from nltk import pos_tag
from sklearn.feature_extraction.text import TfidfVectorizer

# Download NLTK resources
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')

# POS tagging
def pos_tag_nltk(text):
    words = word_tokenize(text)
    tagged = pos_tag(words)
    return tagged, words

# TF-IDF Vectorization
def get_tfidf_embeddings(sentence):
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform([sentence])
    return vectorizer, tfidf_matrix

# Main program
if __name__ == "__main__":
    user_text = input("Enter a sentence: ")

    print("\nPOS Tagging with NLTK:")
    tagged, words = pos_tag_nltk(user_text)
    print(tagged)

    # Get TF-IDF embeddings
    vectorizer, tfidf_matrix = get_tfidf_embeddings(user_text)

    # Print embeddings for each word
    print("\nTF-IDF Embeddings for each word:")
    words = vectorizer.get_feature_names_out()
    for i, word in enumerate(words):
        tfidf_value = tfidf_matrix[0, i]
        print(f"{word}: {tfidf_value}")

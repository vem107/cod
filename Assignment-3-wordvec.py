#pip install nltk
#pip install gensim ( worked on python 3.9.0)

import nltk
from nltk.tokenize import word_tokenize
from nltk import pos_tag
from gensim.models import Word2Vec

# Download NLTK resources
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')

# POS tagging
def pos_tag_nltk(text):
    words = word_tokenize(text)
    tagged = pos_tag(words)
    return tagged, words

# Train Word2Vec on the sentence
def train_word2vec(sentences):
    tokenized = [word_tokenize(sentence.lower()) for sentence in sentences]
    model = Word2Vec(sentences=tokenized, vector_size=10, window=2, min_count=1, sg=1)
    return model, tokenized[0]  # return model and tokenized word list

# Main program
if __name__ == "__main__":
    user_text = input("Enter a sentence: ")

    print("\nPOS Tagging with NLTK:")
    tagged, words = pos_tag_nltk(user_text)
    print(tagged)

    # Train Word2Vec
    w2v_model, tokenized_words = train_word2vec([user_text])

    # Print embeddings for every word
    print("\nWord2Vec Embeddings for each word:")
    for word in tokenized_words:
        vector = w2v_model.wv[word]
        print(f"{word}: {vector}\n")

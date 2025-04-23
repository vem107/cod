import numpy as np
import pandas as pd
import nltk
nltk.download('punkt')
from nltk.tokenize import word_tokenize, sent_tokenize
from langdetect import detect
import string

df = pd.read_csv('NewsCategorizer.csv')
df.head()

def remove_punctuation(text):
    return text.translate(str.maketrans('', '', string.punctuation))

def analyze_text(text):
    try:
        language = detect(text)
    except:
        language = 'unknown'

    # Remove punctuation
    text = remove_punctuation(text)

    words = word_tokenize(text)
    word_count = len(words)

    sentences = sent_tokenize(text)
    sentence_count = len(sentences)

    tokens = word_tokenize(text.lower())

    return {
        'language': language,
        'word_count': word_count,
        'sentence_count': sentence_count,
        'tokens': tokens
    }

# Process dataset function
def process_dataset(path):
    df = pd.read_csv(path)
    total = len(df)

    languages = []
    word_counts = []
    sentence_counts = []
    token_counts = []

    for idx, headline in enumerate(df['headline'], 1):
        results = analyze_text(headline)

        languages.append(results['language'])
        word_counts.append(results['word_count'])
        sentence_counts.append(results['sentence_count'])
        token_counts.append(len(results['tokens']))

        if idx % 5000 == 0:
            print(f"Processed {idx}/{total} articles")

    df['detected_language'] = languages
    df['word_count'] = word_counts
    df['sentence_count'] = sentence_counts
    df['token_count'] = token_counts
    
    return df

processed_dataset = process_dataset('NewsCategorizer.csv')
print(processed_dataset.head())

# Input of different language
sentence1 = "Bonjour le monde ! Comment ça va aujourd'hui ?"
sentence2 = "आभाळ आज खूप सुंदर दिसत आहे."
result1 = analyze_text(sentence1)
result2 = analyze_text(sentence2)
print(result1)
print(result2)

# Query
sentence = input("Enter the senttence")
results = analyze_text(sentence)
print(results)

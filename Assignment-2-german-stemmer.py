# pip install nltk
# nltk.download('punkt') if not work by pip install nltk

# 1.Menu Driven code using one stemmer 
from nltk.stem.snowball import SnowballStemmer
stemmer = SnowballStemmer("german")

# Function to perform stemming
def perform_stemming():
    user_input = input("Enter German words or sentences: ")
    german_words = user_input.split()
    print("\nOriginal Word -> Stemmed Word")
    for word in german_words:
        stemmed_word = stemmer.stem(word)
        print(f"{word} -> {stemmed_word}")
    print("\n")

# Display menu
def display_menu():
    print("Menu:")
    print("1. Enter words for stemming")
    print("2. Exit")

# Main function
def main():
    while True:
        display_menu()
        choice = input("Enter your choice (1 or 2): ")
        if choice == "1":
            perform_stemming()
        elif choice == "2":
            print("Exit!")
            break
        else:
            print("Invalid choice.")

# Entry point
if __name__ == "__main__":
    main()


# 2.Using Three stemmers
import nltk
from nltk.stem import SnowballStemmer, PorterStemmer, LancasterStemmer
from nltk.tokenize import word_tokenize

# Initialize the Snowball, Porter, and Lancaster Stemmers
snowball_stemmer_de = SnowballStemmer("german")
porter_stemmer = PorterStemmer()
lancaster_stemmer = LancasterStemmer()

# Sample words to stem (German)
words_de = ["laufen", "fliegen", "einfach", "glücklich", "springen", "schwimmen"]

# Apply stemming using different stemmers
snowball_stemmed_words_de = [snowball_stemmer_de.stem(word) for word in words_de]
porter_stemmed_words_de = [porter_stemmer.stem(word) for word in words_de]
lancaster_stemmed_words_de = [lancaster_stemmer.stem(word) for word in words_de]

# Print results
print("Using SnowballStemmer (German):")
for word, stemmed in zip(words_de, snowball_stemmed_words_de):
    print(f"Original: {word} -> Stemmed: {stemmed}")

print("\nUsing PorterStemmer (English):")
for word, stemmed in zip(words_de, porter_stemmed_words_de):
    print(f"Original: {word} -> Stemmed: {stemmed}")

print("\nUsing LancasterStemmer (English):")
for word, stemmed in zip(words_de, lancaster_stemmed_words_de):
    print(f"Original: {word} -> Stemmed: {stemmed}")


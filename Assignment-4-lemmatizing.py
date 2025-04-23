import nltk
from nltk.stem import WordNetLemmatizer

def lemmatize_word(word, pos):
    wnl = WordNetLemmatizer()
    return wnl.lemmatize(word, pos=pos)

def main():
    nltk.download('wordnet')
    
    print("Valid POS tags: 'n' (noun), 'v' (verb), 'a' (adjective), 'r' (adverb)")
    
    while True:
        print("\nMenu:")
        print("1. Lemmatize a word")
        print("2. Exit")
        choice = input("Enter your choice: ")
        
        if choice == '1':
            word = input("Enter a word: ")
            pos = input("Enter part of speech (n/v/a/r): ").lower()
            
            if pos not in ['n', 'v', 'a', 'r']:
                print("Invalid POS. Using default 'n' (noun).")
                pos = 'n'
                
            lemmatized_word = lemmatize_word(word, pos)
            print(f"Lemmatized form: {lemmatized_word}")
        
        elif choice == '2':
            print("Exiting program. Goodbye!")
            break
        
        else:
            print("Invalid choice. Please try again.")
            
if __name__ == "__main__":
    main()

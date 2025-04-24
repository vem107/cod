# pip install TextBlob
# pip install vaderSentiment

# 1.code 
from textblob import TextBlob
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

# Function to analyze sentiment using TextBlob
def analyze_sentiment(text):
    blob = TextBlob(text)
    sentiment = blob.sentiment.polarity
    if sentiment > 0:
        return "Positive Sentiment"
    elif sentiment < 0:
        return "Negative Sentiment"
    else:
        return "Neutral Sentiment"

# Function to analyze emotions using VADER
def analyze_emotions(text):
    analyzer = SentimentIntensityAnalyzer()
    sentiment_score = analyzer.polarity_scores(text)
    compound_score = sentiment_score['compound']

    if compound_score >= 0.05:
        return "Positive Emotion"
    elif compound_score <= -0.05:
        return "Negative Emotion"
    else:
        return "Neutral Emotion"

# Function to display results
def analyze_text(text):
    sentiment = analyze_sentiment(text)
    emotions = analyze_emotions(text)

    print("Sentiment Analysis:", sentiment)
    print("Emotion Analysis:", emotions)

# Main code to input text and perform analysis
if __name__ == "__main__":
    text = input("Enter the text for sentiment and emotion analysis: ")
    analyze_text(text)



# 2. Code 
from textblob import TextBlob
from nrclex import NRCLex

# Sentiment analysis using TextBlob
def analyze_sentiment(text):
    blob = TextBlob(text)
    polarity = blob.sentiment.polarity
    if polarity > 0:
        return "Positive"
    elif polarity < 0:
        return "Negative"
    else:
        return "Neutral"

# Emotion analysis using NRCLex
def analyze_emotions(text):
    emotion = NRCLex(text)
    return emotion.raw_emotion_scores

# Display results
def analyze_text(text):
    sentiment = analyze_sentiment(text)
    emotions = analyze_emotions(text)

    print("Sentiment:", sentiment)
    print("Emotions Detected:", emotions)

# Main
if __name__ == "__main__":
    text = input("Enter the text for sentiment and emotion analysis: ")
    analyze_text(text)


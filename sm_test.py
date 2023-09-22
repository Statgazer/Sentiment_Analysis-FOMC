

import spacy
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from textblob import TextBlob

# Load spaCy English model
nlp = spacy.load("en_core_web_sm")

# Load VADER sentiment analyzer
vader_analyzer = SentimentIntensityAnalyzer()

#Paths to your text files
desktop_paths = [
    "/Users/roshen_abraham/Desktop/PC/RA/DR. SUNNY WONG/FOMC Sentiment Analysis/sentiment  analysis project/Draft/FOMC_Meeting_2002_0129_0130.txt",
    "/Users/roshen_abraham/Desktop/PC/RA/DR. SUNNY WONG/FOMC Sentiment Analysis/sentiment  analysis project/Draft/FOMC_Meeting_2009_0127_0128.txt",
    "/Users/roshen_abraham/Desktop/PC/RA/DR. SUNNY WONG/FOMC Sentiment Analysis/sentiment  analysis project/Draft/FOMC_Meeting_2022_0125_0126.txt",
    "/Users/roshen_abraham/Desktop/PC/RA/DR. SUNNY WONG/FOMC Sentiment Analysis/sentiment  analysis project/Draft/FOMC_Meeting_2023_0131_0201.txt"
]

# Lists to store results
vader_results = []
textblob_results = []

# Iterate over each document
for desktop_path in desktop_paths:
    try:
        # Open and read the text file
        with open(desktop_path,"r") as file:
            text = file.read()

        # Preprocess text (you can add more preprocessing steps here)
        text = text.lower()

        # Perform sentiment analysis using VADER
        vader_sentiment_scores = vader_analyzer.polarity_scores(text)
        vader_results.append(vader_sentiment_scores['compound'])

        # Perform sentiment analysis using TextBlob
        textblob_analysis = TextBlob(text)
        textblob_sentiment = textblob_analysis.sentiment
        textblob_results.append(textblob_sentiment.polarity)

    


        # Print sentiment scores for both approaches
        print(f"Document: {desktop_path}")
        print(f"VADER Sentiment - Positive: {vader_sentiment_scores['pos']:.2f}, Negative: {vader_sentiment_scores['neg']:.2f}, Compound: {vader_sentiment_scores['compound']:.2f}")
        print(f"TextBlob Sentiment - Polarity: {textblob_sentiment.polarity:.2f}, Subjectivity: {textblob_sentiment.subjectivity:.2f}\n")

    except FileNotFoundError:
        print(f"File not found at path: {desktop_path}")
    except Exception as e:
        print(f"An error occurred: {str(e)}")

# Now you have sentiment analysis results displayed for each document with positive, negative, and compound scores for both VADER and TextBlob



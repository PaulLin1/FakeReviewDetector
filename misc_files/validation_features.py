import nltk
nltk.download('punkt')
nltk.download('vader_lexicon')
import pandas as pd
from nltk.sentiment import SentimentIntensityAnalyzer
from nltk.util import ngrams
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import CountVectorizer
from textstat import flesch_reading_ease

data = pd.read_csv('fake_5000 - Sheet1.csv')

for i in range(len(data.text_)):
  if data.label[i] == "CG":
    data.loc[i, "label"] = 1
  else:
    data.loc[i, "label"] = 0

def sentiment_analysis(text):
    sia = SentimentIntensityAnalyzer()
    sentiment_score = sia.polarity_scores(text)['compound']
    return sentiment_score

def n_grams(text):
    tokens = word_tokenize(text.lower())

    # Generate bi-grams
    tokenized_texts = list(ngrams(tokens, 2))
    vectorizer = CountVectorizer(ngram_range=(1, 2))

    # Concatenate tokens into strings for CountVectorizer input
    concatenated_texts = [" ".join(tokens) for tokens in tokenized_texts]

    # Transform the tokenized texts into word frequency and bi-gram representations
    X_word_freq_bigrams = vectorizer.fit_transform(concatenated_texts)

    return X_word_freq_bigrams

def fre_readability(text):
    return flesch_reading_ease(text)
    

def coherence_score(rating, sentiment_score):
    if sentiment_score > 0:
        if rating > 3:
            return 1
    return 0
    
def feature_combination(data):
    sentiment_scores = []
    fre_scores = []
    coherence_scores = []
    for text in range(len(data.text_)):
        sentiment_score = sentiment_analysis(data.text_[text])
        fre = fre_readability(data.text_[text])
        coherence = coherence_score(data.rating[text], sentiment_score)
        sentiment_scores.append(sentiment_score)
        fre_scores.append(fre)
        coherence_scores.append(coherence)
    data['sentiment_score'] = sentiment_scores
    data['fre'] = fre_scores
    data['coherence'] = coherence
    data.to_csv('validations_with_features.csv', index=False)
    return data
print(feature_combination(data))


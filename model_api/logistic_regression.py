import nltk
nltk.download('punkt')
nltk.download('vader_lexicon')
import pandas as pd
data = pd.read_csv('fake_5000 - Sheet1.csv')

texts = []
labels = []

for i in range(len(data.text_)):
  texts.append(data.text_[i])

  if data.label[i] == "CG":
    labels.append(1)
  else:
    labels.append(0)

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from nltk.sentiment import SentimentIntensityAnalyzer
from nltk.tokenize import word_tokenize
import numpy as np
from sklearn.metrics import accuracy_score

# Labels: 1 for fake reviews, 0 for genuine reviews
labels = np.array(labels)

# Tokenize and preprocess text data
tokenized_texts = []
for text in texts:
    tokenized_texts.append(word_tokenize(text.lower()))

# Word Frequency and N-grams
vectorizer = CountVectorizer(ngram_range=(1, 2))
X_word_freq = vectorizer.fit_transform([" ".join(tokens) for tokens in tokenized_texts])

# Sentiment Analysis
sid = SentimentIntensityAnalyzer()
sentiments = []
for text in texts:
  sentiments.append(sid.polarity_scores(text)["compound"])

# Combine features
X = X_word_freq
X = np.column_stack([X.toarray(), sentiments])

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, labels, test_size=0.2, random_state=42)

# Logistic Regression model
logreg_model = LogisticRegression()
logreg_model.fit(X_train, y_train)


def accuracy():
  #Prediction based on training
  y_pred = logreg_model.predict(X_test)

  accuracy = accuracy_score(y_test, y_pred)
  print(f"Accuracy {accuracy*100}%")


def detector(review):
   
  user_input_review = review
  # Tokenize and preprocess the user input
  user_input_tokens = word_tokenize(user_input_review.lower())

  # Filter out words not in the training vocabulary
  user_input_tokens_filtered = []
  for token in user_input_tokens:
    if token in vectorizer.vocabulary_:
      user_input_tokens_filtered.append(token)

  # Check if there are no tokens after filtering
  if not user_input_tokens_filtered:
      print("Error: No valid words found in the user input.")
  else:
      # Vectorize the user input using the same CountVectorizer
      user_input_vectorized = vectorizer.transform([" ".join(user_input_tokens_filtered)])

      # Add sentiment score for the user input
      user_input_sentiment = sid.polarity_scores(user_input_review)["compound"]

      #Put features together
      user_input_vectorized = np.column_stack([user_input_vectorized.toarray(), user_input_sentiment])

      # Make a prediction using the trained model
      prediction = logreg_model.predict(user_input_vectorized)
      # Output the prediction
      if prediction >= 0.8:
        return "Most likely Fake", prediction
      if prediction >= 0.5:
          return "Maybe Fake", prediction
      else:
          return "Looks Genuine", prediction

print(detector("Test"))
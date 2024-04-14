import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score
import matplotlib.pyplot as plt
import numpy as np

# Load the data from the file
data = pd.read_csv('review_with_features.csv')

# Randomize the order of data
#data = data.sample(frac=1, random_state=40)
# Split the data into features and labels
# Drop the 'LABEL_ENCODED', 'REVIEW_TEXT', and 'RATING' columns from X
X = data.drop(['LABEL_ENCODED', 'REVIEW_TEXT', 'RATING','Unnamed: 0'], axis=1)
y = data['LABEL_ENCODED']

# Scale the data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split the scaled data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)


# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create an instance of Logistic Regression model
model = LogisticRegression(max_iter=21000)

# Train the model on the training data
model.fit(X_train, y_train)

# Make predictions on the testing data
y_pred = model.predict(X_test)

# Calculate the accuracy of the model
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)


# Assuming 'model' is your logistic regression model
scores = cross_val_score(model, X, y, cv=4)
print("Cross-validated scores:", scores)
# Test the model on the validation data
val_data = pd.read_csv('validations_with_features.csv')
X_validation = val_data.drop(['label', 'text_', 'rating', 'category'], axis=1)

y_validation = val_data['label']

y_validation_pred = model.predict(X_validation)
# # Calculate the accuracy of the model on the validation data
validation_accuracy = accuracy_score(y_validation, y_validation_pred)
#print("Validation Accuracy:", validation_accuracy)

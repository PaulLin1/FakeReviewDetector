# Split the data into training, testing, and validation sets
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
X_train, X_validation, y_train, y_validation = train_test_split(X_train, y_train, test_size=0.25, random_state=42)

# Create an instance of Logistic Regression model
model = LogisticRegression(max_iter=21000)

# Train the model on the training data
model.fit(X_train, y_train)

# Make predictions on the testing data
y_pred = model.predict(X_test)

# Calculate the accuracy of the model on the testing data
accuracy = accuracy_score(y_test, y_pred)
print("Testing Accuracy:", accuracy)

# Make predictions on the validation data
y_validation_pred = model.predict(X_validation)

# Calculate the accuracy of the model on the validation data
validation_accuracy = accuracy_score(y_validation, y_validation_pred)
print("Validation Accuracy:", validation_accuracy)
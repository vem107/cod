import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout, Reshape, Input
from tensorflow.keras.optimizers import Adam

df = pd.read_csv('Sentiment.csv', encoding='ISO-8859-1')
df = df.dropna(subset=['text', 'sentiment'])

# 2. Encode Labels
le = LabelEncoder()
df['label'] = le.fit_transform(df['sentiment'])
num_classes = len(le.classes_)

MAX_FEATURES = 5000
tfidf_vectorizer = TfidfVectorizer(max_features=MAX_FEATURES,
                                   min_df=5,
                                   max_df=0.7,
                                   ngram_range=(1, 2))

# Fit and transform the text data
X_tfidf = tfidf_vectorizer.fit_transform(df['text']).toarray()

# 4. Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X_tfidf, df['label'],
                                                   test_size=0.2,
                                                   random_state=42,
                                                   stratify=df['label'])

# 5. Reshape data for LSTM (LSTM expects 3D input: [samples, time steps, features])
# For TF-IDF vectors, we'll treat each dimension as a time step with 1 feature
X_train_reshaped = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
X_test_reshaped = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)

# 6. Build LSTM Model
model = Sequential([
    Input(shape=(MAX_FEATURES, 1)),
    LSTM(128, return_sequences=True),
    Dropout(0.3),
    LSTM(64),
    Dropout(0.3),
    Dense(32, activation='relu'),
    Dense(num_classes, activation='softmax')
])

model.compile(loss='sparse_categorical_crossentropy',
              optimizer=Adam(learning_rate=0.001),
              metrics=['accuracy'])

model.summary()

# 7. Train Model
history = model.fit(X_train_reshaped, y_train,
                    epochs=2,
                    batch_size=32,
                    validation_data=(X_test_reshaped, y_test),
                    verbose=1)

# 8. Predict and Evaluate
y_pred_probs = model.predict(X_test_reshaped)
y_pred_classes = np.argmax(y_pred_probs, axis=1)

# 9. Confusion Matrix
cm = confusion_matrix(y_test, y_pred_classes)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=le.classes_)
plt.figure(figsize=(10, 8))
disp.plot(cmap='Blues')
plt.title("Confusion Matrix")
plt.tight_layout()
plt.show()

# 10. Classification Report
print("Classification Report:")
print(classification_report(y_test, y_pred_classes, target_names=le.classes_))

# 11. Plot Training History
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Model Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

plt.tight_layout()
plt.show()

def predict_sentiment(text, vectorizer=tfidf_vectorizer, model=model, label_encoder=le):
    # Transform the input text using the same TF-IDF vectorizer
    text_tfidf = vectorizer.transform([text]).toarray()
    
    # Reshape for LSTM
    text_reshaped = text_tfidf.reshape(text_tfidf.shape[0], text_tfidf.shape[1], 1)
    
    # Predict
    prediction = model.predict(text_reshaped)[0]
    predicted_class = np.argmax(prediction)
    
    # Get sentiment label and confidence
    sentiment = label_encoder.inverse_transform([predicted_class])[0]
    confidence = prediction[predicted_class] * 100
    
    return sentiment, confidence

# Example usage
test_text = "I absolutely loved this product! It exceeded all my expectations."
predicted_sentiment, confidence = predict_sentiment(test_text)
print(f"Text: {test_text}")
print(f"Predicted Sentiment: {predicted_sentiment} (Confidence: {confidence:.2f}%)")
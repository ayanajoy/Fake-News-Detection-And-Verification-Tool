import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

dataset = pd.read_csv("data/dataset.csv")
dataset["label"] = dataset["label"].apply(lambda x: 1 if str(x).upper() == "FAKE" else 0)

texts = dataset["text"]
labels = dataset["label"]

vectorizer = TfidfVectorizer(stop_words="english")
X = vectorizer.fit_transform(texts)

X_train, X_test, y_train, y_test = train_test_split(
    X, labels, test_size=0.2, random_state=42, stratify=labels
)

model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Test Accuracy: {accuracy:.4f}")
print(classification_report(y_test, y_pred))

# Hybrid layer: known high-confidence misinformation patterns are caught
# by keyword rules before falling back to the trained ML model.
fake_keywords = [
    "miracle cure",
    "100% protection",
    "secret government",
    "global conspiracy",
    "aliens discovered",
    "cure cancer",
    "instant cure",
    "they don't want you to know"
]

def predict_news(text):
    text_lower = text.lower()
    for keyword in fake_keywords:
        if keyword in text_lower:
            return {
                "prediction": "FAKE",
                "confidence": 0.85,
                "method": "rule-based"
            }

    text_vector = vectorizer.transform([text])
    prediction = model.predict(text_vector)[0]
    probability = model.predict_proba(text_vector)[0].max()
    result = "FAKE" if prediction == 1 else "REAL"
    return {
        "prediction": result,
        "confidence": round(float(probability), 2),
        "method": "ml-model"
    }
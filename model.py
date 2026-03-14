import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

dataset = pd.read_csv("data/dataset.csv")

dataset["label"] = dataset["label"].apply(lambda x: 1 if str(x).upper() == "FAKE" else 0)

texts = dataset["text"]
labels = dataset["label"]

vectorizer = TfidfVectorizer(stop_words="english")

X = vectorizer.fit_transform(texts)

model = LogisticRegression(max_iter=1000)

model.fit(X, labels)


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
                "confidence": 0.85
            }

    text_vector = vectorizer.transform([text])

    prediction = model.predict(text_vector)[0]

    probability = model.predict_proba(text_vector)[0].max()

    result = "FAKE" if prediction == 1 else "REAL"

    return {
        "prediction": result,
        "confidence": round(float(probability), 2)
    }
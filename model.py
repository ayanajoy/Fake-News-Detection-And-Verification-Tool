import re

fake_keywords = [
    "miracle cure",
    "100% protection",
    "secret government",
    "aliens discovered",
    "global conspiracy",
    "hidden truth",
    "scientists shocked",
    "they don't want you to know",
    "virus eliminated",
    "underground city"
]


def predict_news(text):

    text_lower = text.lower()

    score = 0

    suspicious_phrases = []

    for keyword in fake_keywords:
        if keyword in text_lower:
            score += 1
            suspicious_phrases.append(keyword)

    if score >= 2:
        prediction = "FAKE"
        confidence = 0.8
    else:
        prediction = "REAL"
        confidence = 0.6

    return {
        "prediction": prediction,
        "confidence": confidence
    }
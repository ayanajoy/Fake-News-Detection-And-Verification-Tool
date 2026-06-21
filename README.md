# Fake News Detection and Verification Tool

A web-based application that analyzes news articles and classifies them as **REAL or FAKE**, using a hybrid of rule-based pattern matching and a machine learning classifier. The system highlights suspicious phrases, explains why content is flagged, and verifies source credibility against a trusted-domain list.

---

## How It Works

The detection pipeline runs in two layers:

1. **Rule-based filter** — checks the input against a curated list of known high-confidence misinformation patterns (e.g. "miracle cure", "secret government", "aliens discovered"). If matched, the article is flagged FAKE immediately, no ML inference needed.
2. **ML classifier** — if no rule matches, a TF-IDF + Logistic Regression model trained on labeled news data makes the prediction.

This hybrid design keeps obvious cases fast and deterministic while letting the ML model handle everything else.

---

## Features

- Fake news classification (REAL / FAKE) via hybrid rule-based + ML pipeline
- Suspicious phrase detection with plain-language explanations
- Trusted news source verification against a JSON-backed domain allowlist
- Admin dashboard (Streamlit) with real-time classification analytics
- Dockerized for one-command setup

---

## Model Performance

Trained on a balanced 10,000-article subset of the [Kaggle Fake and Real News Dataset](https://www.kaggle.com/datasets/clmentbisaillon/fake-and-real-news-dataset) (5,000 REAL / 5,000 FAKE), evaluated on an 80/20 stratified train-test split:

| Metric | Score |
|---|---|
| Accuracy | 98.1% |
| Precision (REAL) | 0.97 |
| Recall (REAL) | 0.99 |
| Precision (FAKE) | 0.99 |
| Recall (FAKE) | 0.97 |

**Note:** This dataset has known formatting artifacts (REAL articles are Reuters wire stories with a consistent dateline format) that make REAL vs FAKE more separable by writing style than by content alone. Accuracy on this benchmark should be read as a measure of stylistic classification, not general misinformation detection across arbitrary news sources.

---

## Technologies Used

- Python
- Flask (Backend API)
- Streamlit (Admin Dashboard)
- Scikit-learn (TF-IDF + Logistic Regression)
- Pandas (Data Processing)
- Docker (Containerization)

---

## Example

**Input Article**
Scientists discovered a miracle cure that eliminates all viruses within two weeks.

**Output**
Prediction: FAKE

Method: rule-based

Suspicious Phrase: miracle cure

Explanation: Miracle cures are often misinformation.

---

## Running Locally (without Docker)

```bash
pip install -r requirements.txt
python model.py        # trains the model and prints accuracy/classification report
python app.py           # starts the Flask API on port 5000
streamlit run dashboard/admin_dashboard.py  # starts the dashboard on port 8501
```

## Running with Docker

Build the image:
```bash
docker build -t fake-news-app .
```

Run the container:
```bash
docker run -p 5000:5000 -p 8501:8501 fake-news-app
```

This starts both the Flask API (port 5000) and the Streamlit dashboard (port 8501) in a single container.

---

## Project Structure
.

├── app.py                      # Flask API

├── model.py                    # TF-IDF + Logistic Regression model, training & evaluation

├── explainability.py           # Phrase-matching explainability module

├── source_manager.py           # Trusted-source verification

├── dashboard/

│   └── admin_dashboard.py      # Streamlit admin dashboard

├── data/

│   ├── dataset.csv             # Training data (10,000 articles)

│   ├── trusted_sources.json    # Trusted domain allowlist

│   └── prepare_dataset.py      # Script used to build dataset.csv from source data

├── Dockerfile

├── requirements.txt

└── README.md

---

## Author

**Ayana Joy**

---

## License

This project is developed for **educational purposes**, under the MIT License.
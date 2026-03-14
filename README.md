# Fake News Detection and Verification Tool

A web-based application that analyzes news articles and detects whether they are **REAL or FAKE** using machine learning and explainability techniques. The system highlights suspicious phrases, provides explanations, and verifies source credibility.

---

## Features

- Fake news classification (REAL / FAKE)
- Suspicious phrase detection
- AI-based explanation of suspicious claims
- Trusted news source verification
- Admin dashboard with analytics
- Docker container support

---

## Technologies Used

- Python
- Flask (Backend API)
- Streamlit (Frontend Dashboard)
- Scikit-learn (Machine Learning)
- Pandas (Data Processing)
- Docker (Containerization)

---

## Example

**Input Article**
``` 
Scientists discovered a miracle cure that eliminates all viruses within two weeks.
```

**Output**
```
Prediction: FAKE
Suspicious Phrase: miracle cure
Explanation: Miracle cures are often associated with misinformation.
```

---

## Running with Docker

Build the Docker image:
``` 
docker build -t fake-news-app .
```

Run the container:
``` 
docker run -p 5000:5000 -p 8501:8501 fake-news-app
```


---

## Author

**Ayana Joy**  

---

## License

This project is developed for **educational purposes**.

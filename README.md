![CI](https://github.com/sreeja31-arha/Finance_AI-_API/actions/workflows/ci.yml/badge.svg)




# AI Expense Categorizer API

A REST API that takes a transaction description and predicts the expense category using machine learning.

Built with Flask and scikit-learn.

---

## What it does

You send a transaction description like `"Swiggy biryani order"` and the API returns the predicted category like `"Food"` with a confidence score.

---

## Tech Stack

- Python 3.11
- Flask — web API framework
- scikit-learn — TF-IDF vectorizer + Logistic Regression
- pandas — reading and processing training data
- pytest — 22 automated tests
- GitHub Actions — CI/CD pipeline

---

## Project Structure

expense-categorizer/
├── app/
│   └── api.py                  # Flask API with all routes
├── model/
│   ├── train.py                # trains the ML model
│   └── expense_model.joblib    # saved trained model
├── data/
│   └── transactions.csv        # labeled training data
├── tests/
│   └── test_api.py             # 22 pytest tests
├── .github/
│   └── workflows/
│       └── ci.yml              # GitHub Actions CI/CD
├── requirements.txt
└── README.md

---

## How to Run Locally

**1. Clone the repository**

git clone https://github.com/YOURNAME/expense-categorizer.git
cd expense-categorizer

**2. Create and activate virtual environment**

python -m venv venv
venv\Scripts\activate

**3. Install dependencies**
pip install -r requirements.txt

**4. Train the model**
python model/train.py

**5. Start the server**
python app/api.py

Server runs at `http://localhost:5000`

---

## API Endpoints

### GET /health
Check if the server is running.

**Response:**
```json
{
  "status": "ok",
  "model_loaded": true
}
```

---

### POST /predict
Send a transaction description and get the predicted category.

**Request body:**
```json
{
  "description": "Swiggy biryani order"
}
```

**Response:**
```json
{
  "description": "Swiggy biryani order",
  "category": "Food",
  "confidence": 0.76
}
```

---

### GET /categories
Get the list of all categories the model can predict.

**Response:**
```json
{
  "categories": ["Bills", "Entertainment", "Food", "Groceries", "Healthcare", "Transport"],
  "total": 6
}
```

---

## Running Tests
pytest tests/ -v

Expected output: **22 passed**

---

## How the ML Works
Transaction description (text)
↓
TF-IDF Vectorizer
converts text to numbers
↓
Logistic Regression
finds the best matching category
↓
Category + Confidence Score
**TF-IDF** gives higher scores to rare, descriptive words like
"biryani" or "petrol" and ignores common words like "and" or "the".

**Logistic Regression** learns which word patterns belong to
which categories from the training data.

---

## CI/CD

Every push to `main` automatically:
1. Installs dependencies
2. Trains the model
3. Runs all 22 tests

Powered by GitHub Actions.
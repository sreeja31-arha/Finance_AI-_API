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
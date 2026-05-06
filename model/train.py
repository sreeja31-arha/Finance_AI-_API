# import pandas as pd
# import joblib
# import os
# from sklearn.pipeline import Pipeline
# from sklearn.feature_extraction.text import TfidfVectorizer
# from sklearn.linear_model import LogisticRegression
# from sklearn.model_selection import train_test_split
# from sklearn.metrics import classification_report
# import data

# df =pd.read_csv("data/transaction.csv")

# x=df["description"]
# y=df["category"]

# x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2, random_state=42)

# pipeline=Pipeline([("tfidf", TfidfVectorizer(ngram_range=(1,2), lowercase=True)),
#                    ("clf",LogisticRegression(max_iter=1000))])

# pipeline.fit(x_train,y_train)

# y_pred=pipeline.predict(x_train)
# classification_report(y_test,y_pred)

# joblib.dump(pipeline,"model\expense_model.joblib")

import pandas as pd
import joblib
import os
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

df = pd.read_csv("data/transactions.csv")

X = df["description"]
y = df["category"]

print(f"Loaded {len(df)} rows")
print(f"Categories: {sorted(y.unique())}")

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print(f"Training rows: {len(X_train)}")
print(f"Testing rows:  {len(X_test)}")

pipeline = Pipeline([
    ("tfidf", TfidfVectorizer(ngram_range=(1, 2), lowercase=True)),
    ("clf",   LogisticRegression(max_iter=1000))
])

pipeline.fit(X_train, y_train)
print("Training complete!")

y_pred = pipeline.predict(X_test)
print(classification_report(y_test, y_pred))

os.makedirs("model", exist_ok=True)
joblib.dump(pipeline, r"model/expense_model.joblib")
print("Model saved to model/expense_model.joblib")
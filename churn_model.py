import pandas as pd
import pickle

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

df = pd.read_csv("customer_dataset.csv")

X = df[
    [
        "Purchase_Frequency",
        "Last_Purchase_Days",
        "Customer_Satisfaction",
        "Loyalty_Score"
    ]
]

y = df["Churn"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

model.fit(X_train, y_train)

pred = model.predict(X_test)

acc = accuracy_score(y_test, pred)

print("Churn Accuracy:", round(acc, 4))

pickle.dump(
    model,
    open("churn_model.pkl", "wb")
)

print("churn_model.pkl created")
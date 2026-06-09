import pandas as pd
import pickle

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

df = pd.read_csv("customer_dataset.csv")

encoder = LabelEncoder()

df["Customer_Segment"] = encoder.fit_transform(
    df["Customer_Segment"]
)

X = df[
    [
        "Age",
        "Annual_Budget",
        "Browsing_Time",
        "Discount_Sensitivity"
    ]
]

y = df["Customer_Segment"]

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

print("Accuracy:", round(acc, 4))

pickle.dump(
    model,
    open("model.pkl", "wb")
)

pickle.dump(
    encoder,
    open("segment_encoder.pkl", "wb")
)

print("model.pkl created")
print("segment_encoder.pkl created")
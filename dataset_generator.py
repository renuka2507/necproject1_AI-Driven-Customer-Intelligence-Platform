import pandas as pd
import numpy as np
import random

categories = ["Electronics", "Fashion", "Groceries", "Home", "Sports"]

premium_products = ["Smartphone", "Laptop", "Gaming Console"]
medium_products = ["Watch", "Shoes", "Headphones"]
budget_products = ["T-Shirt", "Grocery Essentials", "Household Products"]

data = []

for i in range(500):

    age = random.randint(18, 65)
    gender = random.choice(["Male", "Female"])

    budget = random.randint(5000, 100000)

    browsing_time = random.randint(5, 120)
    discount_sensitivity = random.randint(1, 10)

    purchase_frequency = random.randint(1, 20)
    last_purchase_days = random.randint(1, 180)
    satisfaction = random.randint(1, 10)
    loyalty_score = random.randint(1, 100)

    category = random.choice(categories)

    if budget > 60000:
        segment = "Luxury"
        purchased_product = random.choice(premium_products)

    elif budget >= 25000:
        segment = "Medium"
        purchased_product = random.choice(medium_products)

    else:
        segment = "Budget"
        purchased_product = random.choice(budget_products)

    purchase_amount = random.randint(
        int(budget * 0.05),
        int(budget * 0.30)
    )

    viewed = ",".join(
        random.sample(
            premium_products +
            medium_products +
            budget_products,
            3
        )
    )

    churn = 1 if (
        last_purchase_days > 120
        and loyalty_score < 40
    ) else 0

    data.append([
        f"CUST{i+1}",
        age,
        gender,
        budget,
        category,
        browsing_time,
        discount_sensitivity,
        viewed,
        purchased_product,
        purchase_amount,
        segment,
        purchase_frequency,
        last_purchase_days,
        satisfaction,
        loyalty_score,
        churn
    ])

df = pd.DataFrame(data, columns=[
    "Customer_ID",
    "Age",
    "Gender",
    "Annual_Budget",
    "Preferred_Category",
    "Browsing_Time",
    "Discount_Sensitivity",
    "Product_List_Viewed",
    "Purchased_Product",
    "Purchase_Amount",
    "Customer_Segment",
    "Purchase_Frequency",
    "Last_Purchase_Days",
    "Customer_Satisfaction",
    "Loyalty_Score",
    "Churn"
])

df = df.sample(frac=1).reset_index(drop=True)

df.to_csv("customer_dataset.csv", index=False)

print("Dataset created successfully")
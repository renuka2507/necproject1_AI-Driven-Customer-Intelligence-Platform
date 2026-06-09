import streamlit as st
import pandas as pd
import numpy as np
import pickle
import plotly.express as px
import plotly.figure_factory as ff
from sklearn.cluster import KMeans

# =====================================================
# PAGE CONFIG
# =====================================================

st.set_page_config(
    page_title="AI Customer Intelligence Platform",
    page_icon="📊",
    layout="wide"
)

st.title("📊 AI-Driven Customer Intelligence Platform")
st.markdown("---")

# =====================================================
# LOAD DATA
# =====================================================

# LOAD DATA
df = pd.read_csv("customer_dataset.csv")

# SAFETY CHECKS
import os

if not os.path.exists("customer_dataset.csv"):
    st.error("Dataset file missing!")

if not os.path.exists("model.pkl"):
    st.error("Model file missing!")

if not os.path.exists("churn_model.pkl"):
    st.error("Churn model file missing!")

if not os.path.exists("segment_encoder.pkl"):
    st.error("Encoder file missing!")

# LOAD MODELS
model = pickle.load(open("model.pkl", "rb"))
churn_model = pickle.load(open("churn_model.pkl", "rb"))
encoder = pickle.load(open("segment_encoder.pkl", "rb"))

# TABS (ONLY ONCE)
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "Dashboard",
    "Prediction",
    "Customer Segmentation",
    "Churn Analysis",
    "Recommendation Engine"
])

# =====================================================
# TAB 1 - DASHBOARD
# =====================================================
with tab1:

    st.header("📈 Business Dashboard")

    total_customers = len(df)
    avg_budget = round(df["Annual_Budget"].mean(), 2)
    avg_purchase = round(df["Purchase_Amount"].mean(), 2)
    total_revenue = round(df["Purchase_Amount"].sum(), 2)
    high_value_customers = len(df[df["Annual_Budget"] > 60000])

    churn_rate = round(df["Churn"].mean() * 100, 2)

    c1, c2, c3 = st.columns(3)

    c1.metric("Total Customers", total_customers)
    c2.metric("Average Budget", f"₹{avg_budget:,.0f}")
    c3.metric("Average Purchase", f"₹{avg_purchase:,.0f}")

    c4, c5, c6 = st.columns(3)

    c4.metric("Total Revenue", f"₹{total_revenue:,.0f}")
    c5.metric("High Value Customers", high_value_customers)
    c6.metric("Churn Rate", f"{churn_rate}%")

    st.markdown("---")

    col1, col2 = st.columns(2)

    with col1:
        fig = px.pie(
            df,
            names="Customer_Segment",
            title="Customer Segment Distribution"
        )

        st.plotly_chart(
            fig,
            use_container_width=True,
            key="dash_segment_pie"
        )

    with col2:
        fig = px.pie(
            df,
            names="Preferred_Category",
            title="Preferred Category Distribution"
        )

        st.plotly_chart(
            fig,
            use_container_width=True,
            key="dash_category_pie"
        )

    col3, col4 = st.columns(2)

    with col3:
        product_counts = (
            df["Purchased_Product"]
            .value_counts()
            .reset_index()
        )

        product_counts.columns = ["Product", "Count"]

        fig = px.bar(
            product_counts,
            x="Product",
            y="Count",
            title="Top Purchased Products"
        )

        st.plotly_chart(
            fig,
            use_container_width=True,
            key="dash_products_bar"
        )

    with col4:
        revenue_df = (
            df.groupby("Customer_Segment")["Purchase_Amount"]
            .sum()
            .reset_index()
        )

        fig = px.bar(
            revenue_df,
            x="Customer_Segment",
            y="Purchase_Amount",
            title="Revenue by Segment"
        )

        st.plotly_chart(
            fig,
            use_container_width=True,
            key="dash_revenue_bar"
        )

    col5, col6 = st.columns(2)

    with col5:
        fig = px.histogram(
            df,
            x="Annual_Budget",
            nbins=25,
            title="Annual Budget Distribution"
        )

        st.plotly_chart(
            fig,
            use_container_width=True,
            key="dash_budget_hist"
        )

    with col6:
        fig = px.histogram(
            df,
            x="Purchase_Amount",
            nbins=25,
            title="Purchase Amount Distribution"
        )

        st.plotly_chart(
            fig,
            use_container_width=True,
            key="dash_purchase_hist"
        )
#====================================================
# TAB 2 - PREDICTION
# =====================================================

with tab2:

    st.header("Customer Segment Prediction")

    age = st.number_input("Age", 18, 80, 25)
    budget = st.number_input("Annual Budget", 5000, 100000, 30000)
    browsing = st.number_input("Browsing Time", 1, 200, 30)
    discount = st.number_input("Discount Sensitivity", 1, 10, 5)

    if st.button("Predict Segment"):

        pred = model.predict([[age, budget, browsing, discount]])[0]
        segment = encoder.inverse_transform([pred])[0]

        st.success(f"Predicted Segment: {segment}")

        result_df = pd.DataFrame({
            "Feature": [
                "Age",
                "Annual Budget",
                "Browsing Time",
                "Discount Sensitivity",
                "Predicted Segment"
            ],
            "Value": [
                age,
                budget,
                browsing,
                discount,
                segment
            ]
        })

        st.subheader("Prediction Result")
        st.dataframe(result_df)

        probs = model.predict_proba([[age, budget, browsing, discount]])[0]

        prob_df = pd.DataFrame({
            "Segment": encoder.classes_,
            "Probability": probs
        })

        st.subheader("Prediction Probability")
        st.dataframe(prob_df)

        fig = px.bar(
            prob_df,
            x="Segment",
            y="Probability",
            title="Prediction Probability"
        )

        st.plotly_chart(
            fig,
            use_container_width=True,
            key="prediction_probability_chart"
        )

    st.subheader("Segment Statistics")

    segment_stats = df.groupby("Customer_Segment").agg({
        "Annual_Budget": "mean",
        "Purchase_Amount": "mean",
        "Customer_ID": "count"
    })

    st.dataframe(segment_stats)

    fig = px.pie(
        df,
        names="Customer_Segment",
        title="Customer Segment Distribution"
    )

    st.plotly_chart(
        fig,
        use_container_width=True,
        key="segment_pie_chart"
    )
    # =====================================================
# TAB 3 - CUSTOMER SEGMENTATION
# =====================================================
with tab3:

    st.header("📊 Customer Segmentation")

    X = df[["Annual_Budget", "Browsing_Time", "Purchase_Amount"]]

    kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)

    df["Cluster"] = kmeans.fit_predict(X)

    cluster_names = {
        0: "Budget Customers",
        1: "Medium Customers",
        2: "Luxury Customers"
    }

    df["Cluster_Name"] = df["Cluster"].map(cluster_names)

    st.subheader("Cluster Distribution")

    fig = px.pie(
        df,
        names="Cluster_Name",
        title="Customer Cluster Distribution"
    )

    st.plotly_chart(
        fig,
        use_container_width=True,
        key="cluster_pie_chart"
    )

    st.subheader("Cluster Scatter Plot")

    fig = px.scatter(
        df,
        x="Annual_Budget",
        y="Purchase_Amount",
        color="Cluster_Name",
        title="K-Means Customer Clusters"
    )

    st.plotly_chart(
        fig,
        use_container_width=True,
        key="cluster_scatter_chart"
    )

    st.subheader("Cluster Statistics")

    cluster_stats = df.groupby("Cluster_Name").agg({
        "Annual_Budget": "mean",
        "Purchase_Amount": "mean",
        "Customer_ID": "count"
    })

    st.dataframe(cluster_stats)

    revenue_cluster = df.groupby("Cluster_Name")["Purchase_Amount"].sum().reset_index()

    fig = px.bar(
        revenue_cluster,
        x="Cluster_Name",
        y="Purchase_Amount",
        title="Revenue Contribution by Cluster"
    )

    st.plotly_chart(
        fig,
        use_container_width=True,
        key="cluster_revenue_chart"
    )

# =====================================================
# TAB 4 - CHURN ANALYSIS
# =====================================================

with tab4:

    st.header("Churn Analysis")

    freq = st.number_input("Purchase Frequency", 1, 50, 5)
    days = st.number_input("Last Purchase Days", 1, 365, 60)
    satisfaction = st.number_input("Customer Satisfaction", 1, 10, 7)
    loyalty = st.number_input("Loyalty Score", 1, 100, 50)

    if st.button("Predict Churn"):

        churn_pred = churn_model.predict([[freq, days, satisfaction, loyalty]])[0]

        risk_level = "High Risk" if churn_pred == 1 else "Low Risk"

        if churn_pred == 1:
            st.error("High Churn Risk")
        else:
            st.success("Low Churn Risk")

        result_df = pd.DataFrame({
            "Metric": [
                "Purchase Frequency",
                "Last Purchase Days",
                "Customer Satisfaction",
                "Loyalty Score",
                "Risk Level"
            ],
            "Value": [
                freq,
                days,
                satisfaction,
                loyalty,
                risk_level
            ]
        })

        st.subheader("Prediction Result")
        st.dataframe(result_df)

    # ================= KPI SECTION =================

    total_churn = int(df["Churn"].sum())

    retention = round(((len(df) - total_churn) / len(df)) * 100, 2)

    c1, c2, c3 = st.columns(3)

    c1.metric("Total Churned", total_churn)
    c2.metric("Retention Rate", f"{retention}%")
    c3.metric("Average Loyalty", round(df["Loyalty_Score"].mean(), 2))

    st.markdown("---")

    # ================= CHART 1 =================

    fig = px.pie(
        df,
        names="Churn",
        title="Churn Distribution"
    )

    st.plotly_chart(
        fig,
        use_container_width=True,
        key="churn_pie_chart"
    )

    # ================= CHART 2 =================

    fig = px.scatter(
        df,
        x="Customer_Satisfaction",
        y="Loyalty_Score",
        color="Churn",
        title="Satisfaction vs Loyalty"
    )

    st.plotly_chart(
        fig,
        use_container_width=True,
        key="churn_scatter_chart"
    )

    # ================= CHART 3 =================

    fig = px.histogram(
        df,
        x="Loyalty_Score",
        nbins=20,
        title="Loyalty Score Distribution"
    )

    st.plotly_chart(
        fig,
        use_container_width=True,
        key="churn_loyalty_hist"
    )

    # ================= HIGH RISK CUSTOMERS =================

    high_risk = df[
        (df["Last_Purchase_Days"] > 120) &
        (df["Loyalty_Score"] < 40)
    ]

    st.subheader("High Risk Customers")

    st.dataframe(
        high_risk[
            [
                "Customer_ID",
                "Customer_Segment",
                "Loyalty_Score",
                "Last_Purchase_Days",
                "Churn"
            ]
        ]
    )


# =====================================================
# TAB 5 - RECOMMENDATION ENGINE
# =====================================================


with tab5:

    st.header("🛒 Recommendation Engine")

    customer_budget = st.number_input(
        "Customer Budget",
        min_value=5000,
        max_value=100000,
        value=30000
    )

    # ================= RULE-BASED RECOMMENDATION =================

    if customer_budget > 60000:

        products = [
            "Smartphone",
            "Laptop",
            "Gaming Console"
        ]

        score = 95

    elif customer_budget >= 25000:

        products = [
            "Watch",
            "Shoes",
            "Headphones"
        ]

        score = 88

    else:

        products = [
            "T-Shirt",
            "Grocery Essentials",
            "Household Products"
        ]

        score = 82

    # ================= DISPLAY PRODUCTS =================

    st.subheader("Recommended Products")

    for item in products:
        st.success(item)

    # ================= MATCH SCORE =================

    st.subheader("Customer Preference Match Score")

    st.progress(score / 100)

    st.metric("Match Score", f"{score}%")

    # ================= TABLE =================

    rec_df = pd.DataFrame({
        "Recommended Products": products
    })

    st.dataframe(rec_df)

# =====================================================
# DOWNLOAD DATASET
# =====================================================


st.markdown("---")

st.download_button(
    label="📥 Download Customer Dataset",
    data=df.to_csv(index=False),
    file_name="customer_dataset.csv",
    mime="text/csv"
)
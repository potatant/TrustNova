import streamlit as st
from PIL import Image
import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import cosine_similarity

# Set page configuration
st.set_page_config(page_title="TrustNova Loan Approval System", layout="wide")

# Display the TrustNova logo full-width at the top
st.image("assets/trustnova_logo.png", use_container_width=True)

# Sidebar navigation (Only one sidebar with radio button for feature selection)
st.sidebar.title("Select a feature")
option = st.sidebar.radio(
    "Select a feature",  # Label for the radio button
    ["Loan Approval Prediction", "Credit Score Calculator", "Bank Recommendation"],
    key="sidebar_radio"  # Unique key for the sidebar radio button
)

# Main content area
st.markdown(
    f"<h1 style='text-align: center; color: #2c2c2c;'>TrustNova Loan Approval System</h1>",
    unsafe_allow_html=True
)

st.markdown(
    f"<p style='text-align: center;'>Welcome to the TrustNova platform! Choose a feature below.</p>",
    unsafe_allow_html=True
)

# Logic for feature selection
if option == "Loan Approval Prediction":
    # --- Loan Approval Prediction ---
    st.header("Loan Approval Prediction")

    gender = st.selectbox("Gender", ["Male", "Female"])
    married = st.selectbox("Married", ["Yes", "No"])
    education = st.selectbox("Education", ["Graduate", "Not Graduate"])
    self_employed = st.selectbox("Self Employed", ["Yes", "No"])
    applicant_income = st.number_input("Applicant Income", min_value=0)
    coapplicant_income = st.number_input("Coapplicant Income", min_value=0)
    loan_amount = st.number_input("Loan Amount (in thousands)", min_value=0)
    loan_amount_term = st.number_input("Loan Amount Term (in days)", min_value=0)
    credit_history = st.selectbox("Credit History", ["Good", "Bad"])
    property_area = st.selectbox("Property Area", ["Urban", "Rural", "Semiurban"])

    if st.button("Predict Loan Approval"):
        # Dummy logic (replace with your ML model later)
        if applicant_income > 2500 and credit_history == "Good":
            st.success("✅ Loan Approved!")
        else:
            st.error("❌ Loan Rejected.")

elif option == "Credit Score Calculator":
    # --- Credit Score Calculator ---
    st.header("Credit Score Calculator")

    payment_history = st.slider("Payment History (%)", 0, 100, 80)
    credit_utilization = st.slider("Credit Utilization (%)", 0, 100, 30)
    length_of_credit = st.slider("Length of Credit History (years)", 0, 30, 5)
    credit_mix = st.selectbox("Credit Mix", ["Good", "Fair", "Poor"])
    new_credit = st.slider("New Credit Opened", 0, 10, 2)

    if st.button("Calculate Credit Score"):
        score = 300 + payment_history * 0.3 + (100 - credit_utilization) * 0.2 + length_of_credit * 5 + new_credit * 10
        if credit_mix == "Good":
            score += 50
        elif credit_mix == "Fair":
            score += 20
        st.success(f"📈 Estimated Credit Score: **{int(score)}**")

elif option == "Bank Recommendation":
    # --- Bank Recommendation Engine ---
    st.header("🏦 Bank Recommendation Engine")

    @st.cache_data
    def load_data():
        df = pd.read_csv("banks.csv")
        df.columns = df.columns.str.strip()
        return df

    def convert_to_number(text):
        text = str(text).upper().replace(" ", "")
        if "L" in text:
            return int(float(text.replace("L", "")) * 100000)
        return int(text.replace(",", ""))

    def preprocess_data(df):
        min_vals, max_vals = [], []
        for val in df["Loan Amount Range (INR)"]:
            parts = str(val).split("-")
            if len(parts) == 2:
                min_amt = convert_to_number(parts[0])
                max_amt = convert_to_number(parts[1])
            else:
                min_amt = max_amt = convert_to_number(parts[0])
            min_vals.append(min_amt)
            max_vals.append(max_amt)
        df["Min Amount"] = min_vals
        df["Max Amount"] = max_vals
        return df

    def recommend_loans(df, loan_type, loan_amount):
        df_filtered = df[
            (df['Loan Type'].str.lower() == loan_type.lower()) &
            (df['Min Amount'] <= loan_amount) &
            (df['Max Amount'] >= loan_amount)
        ]

        if df_filtered.empty:
            return pd.DataFrame(), False

        features = df_filtered[['Interest Rate (%)', 'Processing Time (days)']]
        scaler = StandardScaler()
        normalized = scaler.fit_transform(features)
        similarity = cosine_similarity(normalized)
        df_filtered['Similarity Score'] = similarity.mean(axis=1)

        top_banks = df_filtered.sort_values(by='Similarity Score', ascending=False)\
                               .drop_duplicates(subset=['Bank Name'])\
                               .head(3)

        return top_banks[['Bank Name', 'Loan Type', 'Interest Rate (%)', 'Processing Time (days)']], True

    df = preprocess_data(load_data())
    loan_types = df['Loan Type'].dropna().unique()
    selected_loan_type = st.selectbox("Select Loan Type", loan_types)

    loan_amount_range = st.selectbox("Select Loan Amount Range", [
        "1-3 Lakhs", "3-5 Lakhs", "5-10 Lakhs", "10-15 Lakhs", "15-20 Lakhs"
    ])
    range_map = {
        "1-3 Lakhs": (1, 3),
        "3-5 Lakhs": (3, 5),
        "5-10 Lakhs": (5, 10),
        "10-15 Lakhs": (10, 15),
        "15-20 Lakhs": (15, 20)
    }
    selected_range = range_map[loan_amount_range]
    avg_loan_amount = ((selected_range[0] + selected_range[1]) // 2) * 100000

    if st.button("Get Recommendations"):
        results, found = recommend_loans(df, selected_loan_type, avg_loan_amount)
        if found:
            st.success("🎯 Top Bank Recommendations")
        else:
            st.warning("⚠️ No exact match found. Showing all available banks.")
            results = df[['Bank Name', 'Loan Type', 'Interest Rate (%)', 'Processing Time (days)']]
        st.dataframe(results)

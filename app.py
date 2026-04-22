import streamlit as st
import pandas as pd
import pickle

st.set_page_config(page_title="Credit Risk", layout="wide")

st.title("💳 Credit Risk Prediction Dashboard")

# LOAD MODEL
model = pickle.load(open('model.pkl', 'rb'))
scaler = pickle.load(open('scaler.pkl', 'rb'))

mode = st.radio("Select Customer Type", ["New Customer", "Existing Customer"])

# ================= NEW CUSTOMER =================
if mode == "New Customer":

    st.header("🧾 New Customer")

    col1, col2 = st.columns(2)

    with col1:
        gender = st.selectbox("Gender", ["M", "F"])
        income = st.number_input("Income", value=100000)
        age = st.slider("Age", 18, 70, 30)

    with col2:
        credit = st.number_input("Loan Amount", value=500000)
        children = st.number_input("Children", 0, 10, 0)

    st.subheader("Financial Details")

    col3, col4 = st.columns(2)
    total_loans = col3.number_input("Number of Loans", 0, 20, 1)
    total_debt = col4.number_input("Total Debt", value=100000)

    if st.button("Predict Risk"):

        # MODEL INPUT
        data = {
            'CODE_GENDER': gender,
            'AMT_INCOME_TOTAL': income,
            'AMT_CREDIT': credit,
            'DAYS_BIRTH': -age * 365,
            'CNT_CHILDREN': children,
            'TOTAL_LOANS': total_loans,
            'TOTAL_DEBT': total_debt,
            'TOTAL_CREDIT': credit
        }

        df = pd.DataFrame([data])
        df = pd.get_dummies(df)
        df = df.reindex(columns=scaler.feature_names_in_, fill_value=0)
        df = df.fillna(0)

        prob = model.predict_proba(scaler.transform(df))[0][1]

        # RULE-BASED SCORING
        income_ratio = credit / (income + 1)
        debt_ratio = total_debt / (credit + 1)

        score = 0

        if income_ratio > 6:
            score += 2
        elif income_ratio > 3:
            score += 1

        if debt_ratio > 0.7:
            score += 2
        elif debt_ratio > 0.3:
            score += 1

        if total_loans > 10:
            score += 2
        elif total_loans > 3:
            score += 1

        if age < 25:
            score += 1
        elif age > 60:
            score += 2

        if score >= 5:
            risk = "High"
        elif score >= 3:
            risk = "Medium"
        else:
            risk = "Low"

        # OUTPUT
        st.subheader("📊 Result")

        c1, c2 = st.columns(2)
        c1.metric("Model Probability", f"{prob:.2f}")
        c2.metric("Risk Score", score)

        if risk == "High":
            st.error("🔴 HIGH RISK")
        elif risk == "Medium":
            st.warning("🟡 MEDIUM RISK")
        else:
            st.success("🟢 LOW RISK")

        # VISUALS
        st.subheader("📊 Loan vs Debt")
        chart = pd.DataFrame({
            "Amount": [credit, total_debt]
        }, index=["Loan", "Debt"])
        st.bar_chart(chart)

        st.subheader("📈 Debt Ratio")
        st.progress(min(debt_ratio, 1.0))
        st.write(f"{debt_ratio:.2f}")

# ================= EXISTING CUSTOMER =================
else:

    st.header("👤 Existing Customer")

    app_df = pd.read_csv('data/application_train.csv')

    customer_id = st.selectbox("Customer ID", app_df['SK_ID_CURR'])

    cust = app_df[app_df['SK_ID_CURR'] == customer_id]

    col1, col2 = st.columns(2)
    col1.metric("Income", int(cust['AMT_INCOME_TOTAL'].values[0]))
    col2.metric("Credit", int(cust['AMT_CREDIT'].values[0]))

    if st.button("Predict Risk"):

        df = cust.drop(['TARGET'], axis=1)
        df = pd.get_dummies(df)
        df = df.reindex(columns=scaler.feature_names_in_, fill_value=0)
        df = df.fillna(0)   # 🔥 FIX NaN ERROR

        prob = model.predict_proba(scaler.transform(df))[0][1]

        if prob > 0.6:
            risk = "High"
        elif prob > 0.4:
            risk = "Medium"
        else:
            risk = "Low"

        st.subheader("📊 Result")
        st.metric("Probability", f"{prob:.2f}")

        if risk == "High":
            st.error("🔴 HIGH RISK")
        elif risk == "Medium":
            st.warning("🟡 MEDIUM RISK")
        else:
            st.success("🟢 LOW RISK")
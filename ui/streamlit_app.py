from __future__ import annotations

import os
from typing import Any, Dict

import requests
import streamlit as st

DEFAULT_API_URL = os.getenv("CHURN_API_URL", "http://127.0.0.1:8000")

st.set_page_config(
    page_title="Telco Churn Predictor",
    page_icon="📉",
    layout="centered",
)

st.title("Telco Customer Churn Predictor")
st.caption(
    "Demo UI for a production-style ML system: Streamlit → FastAPI → scikit-learn Pipeline. "
    "This tool estimates churn probability based on a customer profile."
)

with st.sidebar:
    st.subheader("API Settings")
    api_base_url = st.text_input("FastAPI base URL", value=DEFAULT_API_URL).strip()
    st.write("Health check:")
    try:
        r = requests.get(f"{api_base_url}/health", timeout=3)
        if r.ok and r.json().get("status") == "ok":
            st.success(f"OK ({r.json().get('model_version', 'unknown')})")
        else:
            st.warning(f"Unexpected response: {r.text}")
    except Exception as e:
        st.error(f"Cannot reach API: {e}")

st.divider()
st.subheader("Customer Inputs")

# --- Inputs ---
col1, col2 = st.columns(2)

with col1:
    gender = st.selectbox("gender", ["Female", "Male"])
    SeniorCitizen = st.selectbox("SeniorCitizen", [0, 1], help="0 = No, 1 = Yes")
    Partner = st.selectbox("Partner", ["Yes", "No"])
    Dependents = st.selectbox("Dependents", ["Yes", "No"])
    tenure = st.number_input("tenure (months)", min_value=0, max_value=200, value=12, step=1)

with col2:
    PhoneService = st.selectbox("PhoneService", ["Yes", "No"])
    MultipleLines = st.selectbox("MultipleLines", ["Yes", "No", "No phone service"])
    InternetService = st.selectbox("InternetService", ["DSL", "Fiber optic", "No"])
    Contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
    PaperlessBilling = st.selectbox("PaperlessBilling", ["Yes", "No"])

st.markdown("### Internet add-on services")

col3, col4 = st.columns(2)
with col3:
    OnlineSecurity = st.selectbox("OnlineSecurity", ["Yes", "No", "No internet service"])
    OnlineBackup = st.selectbox("OnlineBackup", ["Yes", "No", "No internet service"])
    DeviceProtection = st.selectbox("DeviceProtection", ["Yes", "No", "No internet service"])

with col4:
    TechSupport = st.selectbox("TechSupport", ["Yes", "No", "No internet service"])
    StreamingTV = st.selectbox("StreamingTV", ["Yes", "No", "No internet service"])
    StreamingMovies = st.selectbox("StreamingMovies", ["Yes", "No", "No internet service"])

st.markdown("### Billing")

col5, col6 = st.columns(2)
with col5:
    PaymentMethod = st.selectbox(
        "PaymentMethod",
        [
            "Electronic check",
            "Mailed check",
            "Bank transfer (automatic)",
            "Credit card (automatic)",
        ],
    )

with col6:
    MonthlyCharges = st.number_input("MonthlyCharges", min_value=0.0, value=70.0, step=1.0)
    TotalCharges = st.number_input(
        "TotalCharges",
        min_value=0.0,
        value=1000.0,
        step=10.0,
        help="If unknown/new customer, you can set this to 0 and the model will still run.",
    )

payload: Dict[str, Any] = {
    "gender": gender,
    "SeniorCitizen": int(SeniorCitizen),
    "Partner": Partner,
    "Dependents": Dependents,
    "tenure": int(tenure),
    "PhoneService": PhoneService,
    "MultipleLines": MultipleLines,
    "InternetService": InternetService,
    "OnlineSecurity": OnlineSecurity,
    "OnlineBackup": OnlineBackup,
    "DeviceProtection": DeviceProtection,
    "TechSupport": TechSupport,
    "StreamingTV": StreamingTV,
    "StreamingMovies": StreamingMovies,
    "Contract": Contract,
    "PaperlessBilling": PaperlessBilling,
    "PaymentMethod": PaymentMethod,
    "MonthlyCharges": float(MonthlyCharges),
    "TotalCharges": float(TotalCharges),
}

st.divider()

btn = st.button("Predict churn", type="primary")

if btn:
    try:
        with st.spinner("Calling model..."):
            resp = requests.post(f"{api_base_url}/predict", json=payload, timeout=10)
        if not resp.ok:
            st.error(f"API error ({resp.status_code}): {resp.text}")
        else:
            result = resp.json()
            prob = float(result["churn_probability"])
            pred = int(result["churn_prediction"])
            model_version = result.get("model_version", "unknown")

            st.subheader("Result")
            st.metric("Churn probability", f"{prob:.2%}")
            st.write(f"Prediction: **{'Churn' if pred == 1 else 'No churn'}**")
            st.caption(f"Model version: {model_version}")

            # Lightweight explanation
            st.markdown("### Interpretation (high level)")
            if prob >= 0.7:
                st.info(
                    "High churn risk. In many businesses, this would trigger retention outreach "
                    "(e.g., discounts, support check-in, contract incentives)."
                )
            elif prob >= 0.5:
                st.warning(
                    "Moderate churn risk. Consider targeted interventions or monitoring, depending on cost/benefit."
                )
            else:
                st.success("Lower churn risk. No action may be required, depending on your retention strategy.")

            with st.expander("Show request/response JSON"):
                st.json({"request": payload, "response": result})

    except requests.exceptions.RequestException as e:
        st.error(f"Network error calling API: {e}")
    except Exception as e:
        st.error(f"Unexpected error: {e}")

st.divider()
st.caption(
    "Disclaimer: This is a demo model built for portfolio purposes. "
    "Do not use as-is for production decisions without proper validation, monitoring, and governance."
)

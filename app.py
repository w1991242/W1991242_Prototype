import streamlit as st
import pandas as pd
import numpy as np
import joblib
import altair as alt

# ===============================
# App configuration
# ===============================
st.set_page_config(
    page_title="Credit Card Fraud Detection – IPD Prototype",
    layout="wide"
)

# ===============================
# Styling (dark-mode safe)
# ===============================
st.markdown(
    """
    <style>
    .section-title {
        color: #e5e7eb;   /* light grey for dark backgrounds */
        font-weight: 600;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# ===============================
# Expected feature order
# ===============================
EXPECTED_FEATURES = (
    ["Time"]
    + [f"V{i}" for i in range(1, 29)]
    + ["Amount"]
)

# ===============================
# Load model and scaler
# ===============================
model = joblib.load("fraud_model.pkl")
scaler = joblib.load("scaler.pkl")

# ===============================
# Title
# ===============================
st.title("💳 Credit Card Fraud Detection – Prototype")

st.write(
    "This prototype demonstrates how a trained machine learning model is "
    "integrated into a web-based system to identify potentially fraudulent "
    "credit card transactions."
)

st.divider()

# ===============================
# Sidebar controls
# ===============================
st.sidebar.title("Detection Settings")

threshold = st.sidebar.slider(
    "Fraud Probability Threshold",
    min_value=0.01,
    max_value=0.50,
    value=0.05,
    step=0.01
)

st.sidebar.caption(
    "Lower thresholds increase fraud sensitivity but may increase false positives."
)

# ===============================
# Upload section
# ===============================
st.markdown("### <span class='section-title'>1. Upload Transaction Data</span>", unsafe_allow_html=True)

uploaded_file = st.file_uploader(
    "Upload a CSV file containing transaction records",
    type=["csv"]
)

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)

    st.subheader("Data Preview")
    st.dataframe(df.head())

    # ===============================
    # Prepare features
    # ===============================
    if "Class" in df.columns:
        X = df.drop("Class", axis=1)
    else:
        X = df.copy()

    try:
        # ===============================
        # Feature validation
        # ===============================
        missing = set(EXPECTED_FEATURES) - set(X.columns)
        if missing:
            st.error(f"Missing required features: {missing}")
            st.stop()

        X = X.loc[:, EXPECTED_FEATURES]

        # ===============================
        # Scaling
        # ===============================
        X_scaled = scaler.transform(X)

        # ===============================
        # Prediction
        # ===============================
        fraud_prob = model.predict_proba(X_scaled)[:, 1]
        predictions = (fraud_prob >= threshold).astype(int)

        df["Fraud_Probability"] = fraud_prob
        df["Prediction_Label"] = pd.Series(
            predictions, index=df.index
        ).map({0: "Legitimate", 1: "Fraud"})

        st.divider()

        # ===============================
        # Summary
        # ===============================
        st.markdown("### <span class='section-title'>2. Prediction Summary</span>", unsafe_allow_html=True)

        fraud_count = (df["Prediction_Label"] == "Fraud").sum()
        legit_count = (df["Prediction_Label"] == "Legitimate").sum()
        fraud_rate = (fraud_count / len(df)) * 100

        col1, col2, col3 = st.columns(3)
        col1.metric("🚨 Fraudulent", fraud_count)
        col2.metric("✅ Legitimate", legit_count)
        col3.metric("📉 Fraud Rate (%)", f"{fraud_rate:.2f}")

        # ===============================
        # Clean professional chart (Altair)
        # ===============================
        st.markdown("### <span class='section-title'>Detection Overview</span>", unsafe_allow_html=True)

        chart_df = pd.DataFrame({
            "Type": ["Legitimate", "Fraud"],
            "Count": [legit_count, fraud_count]
        })

        chart = (
            alt.Chart(chart_df)
            .mark_bar(cornerRadius=4)
            .encode(
                x=alt.X("Count:Q", title="Number of Transactions"),
                y=alt.Y("Type:N", title=None, sort="-x"),
                color=alt.Color(
                    "Type:N",
                    scale=alt.Scale(
                        domain=["Legitimate", "Fraud"],
                        range=["#9ca3af", "#dc2626"]  # grey / red
                    ),
                    legend=None
                )
            )
            .properties(height=160)
        )

        st.altair_chart(chart, use_container_width=True)

        st.divider()

        # ===============================
        # Results preview
        # ===============================
        st.markdown("### <span class='section-title'>3. Prediction Results (Preview)</span>", unsafe_allow_html=True)

        def highlight(val):
            if val == "Fraud":
                return "color: #dc2626; font-weight: bold;"
            elif val == "Legitimate":
                return "color: #16a34a; font-weight: bold;"
            return ""

        styled_preview = df.head(500).style.applymap(
            highlight,
            subset=["Prediction_Label"]
        )

        st.dataframe(styled_preview, use_container_width=True)

        st.caption(
            "Only the first 500 rows are shown for performance reasons. "
            "The full results can be downloaded below."
        )

        # ===============================
        # Download
        # ===============================
        csv_output = df.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="⬇️ Download Full Predictions (CSV)",
            data=csv_output,
            file_name="fraud_predictions.csv",
            mime="text/csv"
        )

        st.caption(
            "Fraudulent transactions are highlighted in red and legitimate "
            "transactions in green. Uploaded data is not stored."
        )

    except Exception as e:
        st.error(
            "Error processing the file. Please ensure the uploaded CSV "
            "matches the expected dataset format."
        )
        st.exception(e)
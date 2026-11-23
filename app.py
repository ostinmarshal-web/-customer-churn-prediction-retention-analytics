
import streamlit as st
import pandas as pd
import joblib
from pathlib import Path

# ==========================================
# PAGE CONFIGURATION
# ==========================================
st.set_page_config(
    page_title="Retention Intelligence",
    page_icon="📊",
    layout="wide"
)

BASE_DIR = Path(__file__).parent
MODEL_PATH = BASE_DIR / "customer_churn_model.pkl"
DATA_PATH = BASE_DIR / "WA_Fn-UseC_-Telco-Customer-Churn.csv"


# ==========================================
# CUSTOM CSS
# ==========================================
st.markdown("""
<style>
.block-container {
    padding-top: 1.8rem;
    padding-bottom: 2rem;
}

[data-testid="stMetric"] {
    background-color: #172033;
    border: 1px solid #263550;
    padding: 16px;
    border-radius: 12px;
}

[data-testid="stMetricLabel"] {
    color: #AAB6CC;
}

div.stButton > button {
    border-radius: 10px;
    min-height: 45px;
    font-weight: 600;
}

</style>
""", unsafe_allow_html=True)


# ==========================================
# LOAD MODEL
# ==========================================
@st.cache_resource
def load_artifact():
    return joblib.load(MODEL_PATH)


# ==========================================
# LOAD DATA
# ==========================================
@st.cache_data
def load_dataset():

    data = pd.read_csv(DATA_PATH)

    data["TotalCharges"] = pd.to_numeric(
        data["TotalCharges"],
        errors="coerce"
    )

    data = data.dropna(subset=["TotalCharges"])

    return data


artifact = load_artifact()

model = artifact["model"]
threshold = artifact["threshold"]
expected_features = artifact["features"]

df = load_dataset()


# ==========================================
# SIDEBAR
# ==========================================
st.sidebar.title("📊 Retention Intelligence")
st.sidebar.caption("Customer analytics platform")

page = st.sidebar.radio(
    "Navigation",
    ["Overview", "Risk Segmentation", "Predict Churn"]
)

st.sidebar.divider()

st.sidebar.caption("Prediction Engine")
st.sidebar.write("Machine Learning Model")
st.sidebar.caption(f"Decision Threshold: {threshold:.3f}")


# ==========================================
# OVERVIEW DASHBOARD
# ==========================================
if page == "Overview":

    st.title("Customer Retention Dashboard")

    st.write(
        "Understand customer churn patterns, subscription behavior "
        "and revenue trends."
    )

    # ------------------------------------------
    # KPI CALCULATIONS
    # ------------------------------------------
    total_customers = len(df)

    churned = df[df["Churn"] == "Yes"]
    retained = df[df["Churn"] == "No"]

    churn_count = len(churned)

    churn_rate = (
        churn_count / total_customers * 100
        if total_customers > 0
        else 0
    )

    avg_monthly_charges = df["MonthlyCharges"].mean()

    churned_monthly_revenue = churned["MonthlyCharges"].sum()

    # ------------------------------------------
    # KPI DISPLAY
    # ------------------------------------------
    st.subheader("Business Overview")

    col1, col2, col3, col4 = st.columns(4)

    col1.metric(
        "Total Customers",
        f"{total_customers:,}"
    )

    col2.metric(
        "Churn Rate",
        f"{churn_rate:.1f}%"
    )

    col3.metric(
        "Avg. Monthly Charges",
        f"${avg_monthly_charges:,.2f}"
    )

    col4.metric(
        "Churned Monthly Revenue",
        f"${churned_monthly_revenue:,.2f}"
    )

    st.caption(
        "Churned monthly revenue represents monthly charges from "
        "customers who have already churned in this dataset."
    )

    st.divider()

    # ------------------------------------------
    # CHURN DISTRIBUTION
    # ------------------------------------------
    st.subheader("Churn Analysis")

    chart_col1, chart_col2 = st.columns(2)

    with chart_col1:

        st.markdown("#### Customer Distribution")

        churn_distribution = df["Churn"].value_counts()

        st.bar_chart(churn_distribution)

    # ------------------------------------------
    # CHURN BY CONTRACT
    # ------------------------------------------
    with chart_col2:

        st.markdown("#### Churn Rate by Contract")

        contract_churn = (
            df.groupby("Contract")["Churn"]
            .apply(lambda x: (x == "Yes").mean() * 100)
            .sort_values(ascending=False)
        )

        st.bar_chart(contract_churn)

    st.divider()

    # ------------------------------------------
    # CUSTOMER INSIGHTS
    # ------------------------------------------
    st.subheader("Customer Insights")

    insight_col1, insight_col2 = st.columns(2)

    with insight_col1:

        st.markdown("#### Average Monthly Charges")

        monthly_charges = df.groupby("Churn")["MonthlyCharges"].mean()

        st.bar_chart(monthly_charges)

    with insight_col2:

        st.markdown("#### Customer Tenure")

        tenure_distribution = (
            df.groupby("Churn")["tenure"]
            .mean()
        )

        st.bar_chart(tenure_distribution)

    # ------------------------------------------
    # DATA EXPLORER
    # ------------------------------------------
    with st.expander("Explore Customer Dataset"):

        st.dataframe(
            df,
            use_container_width=True
        )


# ==========================================
# RISK SEGMENTATION
# ==========================================
elif page == "Risk Segmentation":
        st.title("🎯 Customer Risk Segmentation")
        st.write(
            "Identify customers who are most likely to leave "
            "and prioritize retention efforts."
        )

        # Calculate churn probability for every customer
        risk_df = df.copy()

        risk_df["ChurnProbability"] = model.predict_proba(
            risk_df[expected_features]
        )[:, 1]

        # Assign risk categories
        risk_df["RiskLevel"] = pd.cut(
            risk_df["ChurnProbability"],
            bins=[-0.001, 0.30, 0.60, 1.00],
            labels=["Low Risk", "Medium Risk", "High Risk"]
        )

        # Risk category counts
        low_risk = (risk_df["RiskLevel"] == "Low Risk").sum()
        medium_risk = (risk_df["RiskLevel"] == "Medium Risk").sum()
        high_risk = (risk_df["RiskLevel"] == "High Risk").sum()

        # Display risk metrics
        col1, col2, col3 = st.columns(3)

        col1.metric("🟢 Low Risk Customers", low_risk)
        col2.metric("🟡 Medium Risk Customers", medium_risk)
        col3.metric("🔴 High Risk Customers", high_risk)

        st.divider()

        # Risk distribution chart
        st.subheader("Customer Risk Distribution")

        risk_distribution = (
    risk_df["RiskLevel"]
    .value_counts()
    .reindex(["Low Risk", "Medium Risk", "High Risk"], fill_value=0)
)

        st.bar_chart(risk_distribution)

        # High-risk customer analysis
        st.subheader("🚨 High-Risk Customer Analysis")

        high_risk_df = risk_df[
            risk_df["RiskLevel"] == "High Risk"
        ].sort_values(
            by="ChurnProbability",
            ascending=False
        )

        if not high_risk_df.empty:

            # Monthly charges associated with high-risk customers
            high_risk_revenue = high_risk_df["MonthlyCharges"].sum()

            col1, col2 = st.columns(2)

            col1.metric(
                "High-Risk Customers",
                len(high_risk_df)
            )

            col2.metric(
                "Monthly Charges at Risk",
                f"${high_risk_revenue:,.2f}"
            )

            st.caption(
                "Monthly charges at risk represent the combined monthly "
                "charges of high-risk customers, not guaranteed revenue loss."
            )

            st.subheader("Priority Retention List")
            contract_filter = st.selectbox(
            "Filter by Contract Type",
            ["All Contracts", "Month-to-month", "One year", "Two year"]
        )
            if contract_filter == "All Contracts":
                filtered_customers = high_risk_df.copy()
        else:
            filtered_customers = high_risk_df[
            high_risk_df["Contract"] == contract_filter
            ]

            display_columns = [
                "customerID",
                "tenure",
                "Contract",
                "MonthlyCharges",
                "ChurnProbability",
                "RiskLevel"
            ]

            st.dataframe(
                high_risk_df[display_columns],
                use_container_width=True,
                hide_index=True
            )

            # Download high-risk customer data
            csv_data = high_risk_df[display_columns].to_csv(
                index=False
            ).encode("utf-8")

            st.download_button(
                label="📥 Download High-Risk Customers",
                data=csv_data,
                file_name="high_risk_customers.csv",
                mime="text/csv"
            )


        st.divider()

        st.info(
            "💡 Retention strategy: Contact high-risk customers first. "
            "Consider personalized offers, contract upgrades, pricing reviews, "
            "and improved customer support."
        )
elif page == "Predict Churn":

    st.title("Customer Churn Prediction")

    st.write(
        "Enter customer information to estimate their likelihood "
        "of leaving the service."
    )

    # ------------------------------------------
    # CUSTOMER FORM
    # ------------------------------------------
    with st.form("customer_form"):

        st.subheader("Customer Profile")

        col1, col2 = st.columns(2)

        with col1:

            gender = st.selectbox(
                "Gender",
                ["Male", "Female"]
            )

            senior = st.selectbox(
                "Senior Citizen",
                [0, 1]
            )

            partner = st.selectbox(
                "Has Partner?",
                ["Yes", "No"]
            )

            dependents = st.selectbox(
                "Has Dependents?",
                ["Yes", "No"]
            )

            tenure = st.number_input(
                "Tenure (months)",
                min_value=0,
                max_value=100,
                value=12
            )

            phone = st.selectbox(
                "Phone Service",
                ["Yes", "No"]
            )

            multiple = st.selectbox(
                "Multiple Lines",
                ["No", "Yes", "No phone service"]
            )

            internet = st.selectbox(
                "Internet Service",
                ["DSL", "Fiber optic", "No"]
            )

            security = st.selectbox(
                "Online Security",
                ["Yes", "No", "No internet service"]
            )

            backup = st.selectbox(
                "Online Backup",
                ["Yes", "No", "No internet service"]
            )

        with col2:

            protection = st.selectbox(
                "Device Protection",
                ["Yes", "No", "No internet service"]
            )

            support = st.selectbox(
                "Tech Support",
                ["Yes", "No", "No internet service"]
            )

            tv = st.selectbox(
                "Streaming TV",
                ["Yes", "No", "No internet service"]
            )

            movies = st.selectbox(
                "Streaming Movies",
                ["Yes", "No", "No internet service"]
            )

            contract = st.selectbox(
                "Contract",
                ["Month-to-month", "One year", "Two year"]
            )

            paperless = st.selectbox(
                "Paperless Billing",
                ["Yes", "No"]
            )

            payment = st.selectbox(
                "Payment Method",
                [
                    "Electronic check",
                    "Mailed check",
                    "Bank transfer (automatic)",
                    "Credit card (automatic)"
                ]
            )

            monthly = st.number_input(
                "Monthly Charges",
                min_value=0.0,
                value=70.0
            )

            total = st.number_input(
                "Total Charges",
                min_value=0.0,
                value=840.0
            )

        submitted = st.form_submit_button(
            "Analyze Customer",
            use_container_width=True
        )

    # ------------------------------------------
    # PREDICTION
    # ------------------------------------------
    if submitted:

        customer = {
            "gender": gender,
            "SeniorCitizen": senior,
            "Partner": partner,
            "Dependents": dependents,
            "tenure": tenure,
            "PhoneService": phone,
            "MultipleLines": multiple,
            "InternetService": internet,
            "OnlineSecurity": security,
            "OnlineBackup": backup,
            "DeviceProtection": protection,
            "TechSupport": support,
            "StreamingTV": tv,
            "StreamingMovies": movies,
            "Contract": contract,
            "PaperlessBilling": paperless,
            "PaymentMethod": payment,
            "MonthlyCharges": monthly,
            "TotalCharges": total
        }

        input_df = pd.DataFrame([customer])

        input_df = input_df[expected_features]

        probability = model.predict_proba(input_df)[0, 1]

        prediction = int(probability >= threshold)

        # ------------------------------------------
        # RESULTS
        # ------------------------------------------
        st.divider()

        st.subheader("Prediction Results")

        result_col1, result_col2 = st.columns(2)

        with result_col1:

            st.metric(
                "Churn Probability",
                f"{probability:.1%}"
            )

            st.progress(float(probability))

        with result_col2:

            if prediction == 1:

                st.error(
                    "⚠️ Customer predicted to churn"
                )

            else:

                st.success(
                    "✅ Customer predicted to stay"
                )

        # ------------------------------------------
        # RISK INTERPRETATION
        # ------------------------------------------
        if prediction == 1:

            st.warning(
                "This customer's estimated churn probability "
                "exceeds the model's decision threshold."
            )

            st.subheader("Retention Recommendations")

            if contract == "Month-to-month":

                st.write(
                    "• Consider offering an annual contract incentive."
                )

            if security == "No":

                st.write(
                    "• Consider bundling Online Security services."
                )

            if support == "No":

                st.write(
                    "• Offer proactive technical support."
                )

            if monthly >= 70:

                st.write(
                    "• Review pricing and consider a personalized offer."
                )

            st.write(
                "• Contact the customer to understand their concerns."
            )

        else:

            st.info(
                "This customer is currently predicted to stay. "
                "Continue monitoring satisfaction and engagement."
            )

        st.caption(
            f"Decision threshold: {threshold:.3f}. "
            "This prediction is an estimate, not a guarantee."
        )

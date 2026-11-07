from io import BytesIO

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4, landscape
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.platypus import (
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)

# --------------------------------------------------
# PAGE CONFIGURATION
# --------------------------------------------------
st.set_page_config(
    page_title="FraudGuard AI",
    page_icon="🛡️",
    layout="wide",
)

# --------------------------------------------------
# CUSTOM DARK THEME
# --------------------------------------------------
st.markdown(
    """
    <style>
    .stApp {
        background-color: #0b1220;
        color: #ffffff;
    }
    [data-testid="stSidebar"] {
        background-color: #111c30;
    }
    h1, h2, h3, p, label {
        color: #ffffff !important;
    }
    [data-testid="stMetric"] {
        background: #17263d;
        padding: 16px;
        border-radius: 12px;
        border: 1px solid #263b59;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# --------------------------------------------------
# MODEL LOADING
# --------------------------------------------------
@st.cache_resource
def load_fraud_model():
    return joblib.load("fraud_model.pkl")


try:
    model = load_fraud_model()
    model_load_error = None
except Exception as error:
    model = None
    model_load_error = str(error)


# --------------------------------------------------
# REQUIRED MODEL FEATURES
# --------------------------------------------------
REQUIRED_COLUMNS = [
    "Time",
    "V1", "V2", "V3", "V4", "V5", "V6",
    "V7", "V8", "V9", "V10", "V11", "V12",
    "V13", "V14", "V15", "V16", "V17", "V18",
    "V19", "V20", "V21", "V22", "V23", "V24",
    "V25", "V26", "V27", "V28",
    "Amount",
]


# --------------------------------------------------
# PDF REPORT GENERATOR
# --------------------------------------------------
def create_pdf_report(transactions):
    buffer = BytesIO()
    document = SimpleDocTemplate(
        buffer,
        pagesize=landscape(A4),
        rightMargin=30,
        leftMargin=30,
        topMargin=30,
        bottomMargin=30,
    )
    styles = getSampleStyleSheet()
    elements = [
        Paragraph("FraudGuard AI - Fraud Detection Report", styles["Title"]),
        Spacer(1, 16),
    ]

    total = len(transactions)
    fraud_count = int((transactions["Prediction"] == 1).sum())
    legitimate_count = total - fraud_count
    average_probability = (
        float(transactions["Fraud Probability (%)"].mean())
        if total else 0.0
    )

    summary = [
        ["Total Transactions", "Potential Fraud", "Legitimate", "Average Fraud Probability"],
        [str(total), str(fraud_count), str(legitimate_count), f"{average_probability:.2f}%"],
    ]
    summary_table = Table(summary, colWidths=[150, 150, 150, 190])
    summary_table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#17365D")),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                ("BACKGROUND", (0, 1), (-1, 1), colors.HexColor("#EAF0F8")),
                ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
                ("PADDING", (0, 0), (-1, -1), 7),
                ("ALIGN", (0, 0), (-1, -1), "CENTER"),
            ]
        )
    )
    elements.extend([summary_table, Spacer(1, 18)])

    table_data = [["Amount", "Prediction", "Fraud Probability", "Result", "Risk Level"]]
    for _, row in transactions.head(25).iterrows():
        table_data.append(
            [
                f'{float(row["Amount"]):,.2f}',
                str(int(row["Prediction"])),
                f'{float(row["Fraud Probability (%)"]):.2f}%',
                str(row["Result"]),
                str(row["Risk Level"]),
            ]
        )

    detail_table = Table(
        table_data,
        repeatRows=1,
        colWidths=[100, 90, 130, 170, 110],
    )
    detail_table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#17365D")),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
                ("PADDING", (0, 0), (-1, -1), 6),
                ("FONTSIZE", (0, 0), (-1, -1), 8),
            ]
        )
    )
    elements.append(Paragraph("First 25 Predicted Transactions", styles["Heading2"]))
    elements.append(detail_table)
    elements.append(Spacer(1, 10))
    elements.append(
        Paragraph(
            "Note: Model outputs are estimates for review, not proof of fraud.",
            styles["Italic"],
        )
    )

    document.build(elements)
    buffer.seek(0)
    return buffer


# --------------------------------------------------
# SAMPLE DASHBOARD DATA (DEMO ONLY)
# --------------------------------------------------
@st.cache_data
def create_demo_data():
    rng = np.random.default_rng(42)
    count = 1000
    return pd.DataFrame(
        {
            "Transaction_ID": range(1001, 1001 + count),
            "Amount": rng.uniform(100, 50000, count).round(2),
            "Hour": rng.integers(0, 24, count),
            "Fraud": rng.choice([0, 1], count, p=[0.95, 0.05]),
        }
    )


# --------------------------------------------------
# SHARED HELPER
# --------------------------------------------------
def get_prediction_results():
    return st.session_state.get("prediction_results")


# --------------------------------------------------
# SIDEBAR NAVIGATION
# --------------------------------------------------
st.sidebar.title("🛡️ FraudGuard AI")
st.sidebar.caption("AI-Powered Financial Risk Monitoring")

pages = [
    "Dashboard",
    "Fraud Prediction",
    "Financial Analytics",
    "CSV Upload",
    "Admin Login",
    "Admin Dashboard",
    "PDF Reports",
]
page = st.sidebar.radio("Navigation", pages)

if model_load_error:
    st.sidebar.warning("Model file not loaded")
else:
    st.sidebar.success("AI model loaded")


# --------------------------------------------------
# DASHBOARD
# --------------------------------------------------
if page == "Dashboard":
    st.title("🛡️ Credit Card Fraud Detection")
    st.caption("AI-powered Financial Risk Monitoring")

    transactions = get_prediction_results()

    if transactions is not None:
        total = len(transactions)
        fraud_count = int((transactions["Prediction"] == 1).sum())
        legitimate_count = total - fraud_count
        fraud_amount = float(
            transactions.loc[
                transactions["Prediction"] == 1, "Amount"
            ].sum()
        )

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Total Transactions", f"{total:,}")
        c2.metric("Potential Fraud", f"{fraud_count:,}")
        c3.metric("Legitimate", f"{legitimate_count:,}")
        c4.metric("Potential Fraud Value", f"{fraud_amount:,.2f}")

        st.info("Dashboard metrics are based on your latest uploaded prediction file.")

        chart_data = pd.DataFrame(
            {
                "Transaction Type": ["Legitimate", "Potential Fraud"],
                "Count": [legitimate_count, fraud_count],
            }
        )
        fig = px.pie(
            chart_data,
            names="Transaction Type",
            values="Count",
            hole=0.5,
            color="Transaction Type",
            color_discrete_map={
                "Legitimate": "#16a085",
                "Potential Fraud": "#e74c3c",
            },
        )
        st.plotly_chart(fig, use_container_width=True)

        st.subheader("Recent Predictions")
        st.dataframe(transactions.tail(10), use_container_width=True, hide_index=True)

    else:
        demo_data = create_demo_data()
        demo_total = len(demo_data)
        demo_fraud = int(demo_data["Fraud"].sum())
        demo_legitimate = demo_total - demo_fraud
        demo_fraud_value = float(
            demo_data.loc[demo_data["Fraud"] == 1, "Amount"].sum()
        )

        st.warning(
            "You are viewing synthetic demonstration data. "
            "Upload a transaction file in Fraud Prediction to view actual model results."
        )
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Demo Transactions", f"{demo_total:,}")
        c2.metric("Demo Fraud Cases", f"{demo_fraud:,}")
        c3.metric("Demo Legitimate", f"{demo_legitimate:,}")
        c4.metric("Demo Fraud Value", f"{demo_fraud_value:,.2f}")

        left, right = st.columns(2)
        with left:
            demo_chart = pd.DataFrame(
                {
                    "Type": ["Legitimate", "Potential Fraud"],
                    "Count": [demo_legitimate, demo_fraud],
                }
            )
            fig = px.pie(
                demo_chart,
                names="Type",
                values="Count",
                hole=0.5,
            )
            st.plotly_chart(fig, use_container_width=True)

        with right:
            st.subheader("Demo Fraud by Transaction Hour")
            fig = px.histogram(
                demo_data[demo_data["Fraud"] == 1],
                x="Hour",
                nbins=24,
            )
            st.plotly_chart(fig, use_container_width=True)


# --------------------------------------------------
# FRAUD PREDICTION
# --------------------------------------------------
elif page == "Fraud Prediction":
    st.title("AI Fraud Prediction")
    st.write(
        "Upload a CSV containing the 30 model features: "
        "Time, V1–V28 and Amount."
    )

    if model is None:
        st.error(
            "The trained model could not be loaded. "
            "Check that fraud_model.pkl is in the same folder as app.py."
        )
        st.stop()

    uploaded_file = st.file_uploader(
        "Upload transaction CSV",
        type=["csv"],
        key="fraud_prediction_upload",
    )

    if uploaded_file is not None:
        try:
            transactions = pd.read_csv(uploaded_file)

            missing_columns = [
                column
                for column in REQUIRED_COLUMNS
                if column not in transactions.columns
            ]

            if missing_columns:
                st.error(
                    "This CSV is missing required model columns: "
                    + ", ".join(missing_columns)
                )
            elif transactions.empty:
                st.warning("The uploaded CSV contains no transaction rows.")
            else:
                X = transactions[REQUIRED_COLUMNS].copy()

                for column in REQUIRED_COLUMNS:
                    X[column] = pd.to_numeric(X[column], errors="coerce")

                if X.isnull().any().any():
                    bad_columns = X.columns[X.isnull().any()].tolist()
                    st.error(
                        "Some required feature values are empty or non-numeric. "
                        "Please check: " + ", ".join(bad_columns)
                    )
                else:
                    predictions = model.predict(X)
                    probabilities = model.predict_proba(X)[:, 1]

                    transactions["Prediction"] = predictions.astype(int)
                    transactions["Fraud Probability (%)"] = (
                        probabilities * 100
                    ).round(2)
                    transactions["Result"] = transactions["Prediction"].map(
                        {
                            0: "Legitimate",
                            1: "Potential Fraud",
                        }
                    )
                    transactions["Risk Level"] = (
                        transactions["Fraud Probability (%)"].apply(
                            lambda probability: (
                                "High Risk"
                                if probability >= 75
                                else "Medium Risk"
                                if probability >= 50
                                else "Low Risk"
                            )
                        )
                    )

                    st.session_state["prediction_results"] = transactions

                    total = len(transactions)
                    fraud_count = int((transactions["Prediction"] == 1).sum())
                    legitimate_count = total - fraud_count
                    average_probability = float(
                        transactions["Fraud Probability (%)"].mean()
                    )

                    st.success("Predictions completed successfully!")
                    c1, c2, c3, c4 = st.columns(4)
                    c1.metric("Total Transactions", total)
                    c2.metric("Potential Fraud", fraud_count)
                    c3.metric("Legitimate", legitimate_count)
                    c4.metric("Average Fraud Probability", f"{average_probability:.2f}%")

                    st.subheader("Prediction Results")
                    st.dataframe(
                        transactions,
                        use_container_width=True,
                        hide_index=True,
                    )

                    csv_report = transactions.to_csv(index=False).encode("utf-8")
                    st.download_button(
                        "Download Prediction CSV",
                        data=csv_report,
                        file_name="fraud_detection_results.csv",
                        mime="text/csv",
                    )

                    pdf_report = create_pdf_report(transactions)
                    st.download_button(
                        "Download PDF Report",
                        data=pdf_report,
                        file_name="FraudGuard_AI_Report.pdf",
                        mime="application/pdf",
                    )

        except Exception as error:
            st.error(f"Unable to process this CSV: {error}")

    else:
        st.info("Choose a compatible CSV file to begin fraud prediction.")


# --------------------------------------------------
# FINANCIAL ANALYTICS
# --------------------------------------------------
elif page == "Financial Analytics":
    st.title("Financial Analytics")
    transactions = get_prediction_results()

    if transactions is None:
        st.info(
            "No prediction results are available yet. "
            "Upload a compatible CSV in Fraud Prediction first."
        )
    else:
        total = len(transactions)
        fraud_count = int((transactions["Prediction"] == 1).sum())
        legitimate_count = total - fraud_count
        fraud_rate = (fraud_count / total * 100) if total else 0.0
        average_probability = (
            float(transactions["Fraud Probability (%)"].mean())
            if total else 0.0
        )

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Total Transactions", total)
        c2.metric("Potential Fraud", fraud_count)
        c3.metric("Legitimate", legitimate_count)
        c4.metric("Fraud Rate", f"{fraud_rate:.2f}%")

        st.metric("Average Fraud Probability", f"{average_probability:.2f}%")
        st.progress(min(fraud_rate / 100, 1.0))

        st.subheader("Fraud vs Legitimate Transactions")
        outcome_data = pd.DataFrame(
            {
                "Transaction Type": ["Potential Fraud", "Legitimate"],
                "Count": [fraud_count, legitimate_count],
            }
        )
        fig = px.pie(
            outcome_data,
            names="Transaction Type",
            values="Count",
            hole=0.5,
            color="Transaction Type",
            color_discrete_map={
                "Potential Fraud": "#e74c3c",
                "Legitimate": "#16a085",
            },
        )
        st.plotly_chart(fig, use_container_width=True)

        st.subheader("Risk Level Summary")
        risk_order = ["High Risk", "Medium Risk", "Low Risk"]
        risk_summary = (
            transactions["Risk Level"]
            .value_counts()
            .reindex(risk_order, fill_value=0)
        )
        c1, c2, c3 = st.columns(3)
        c1.metric("High Risk", int(risk_summary["High Risk"]))
        c2.metric("Medium Risk", int(risk_summary["Medium Risk"]))
        c3.metric("Low Risk", int(risk_summary["Low Risk"]))

        risk_chart_data = risk_summary.rename_axis("Risk Level").reset_index(name="Transactions")
        fig = px.bar(
            risk_chart_data,
            x="Risk Level",
            y="Transactions",
            color="Risk Level",
            color_discrete_map={
                "High Risk": "#e74c3c",
                "Medium Risk": "#f39c12",
                "Low Risk": "#16a085",
            },
        )
        st.plotly_chart(fig, use_container_width=True)

        st.subheader("Risk Distribution Percentage")
        risk_percentage = (
            risk_summary / risk_summary.sum() * 100
            if risk_summary.sum() else risk_summary.astype(float)
        )
        percentage_data = risk_percentage.rename("Percentage (%)").reset_index()
        percentage_data = percentage_data.rename(columns={"index": "Risk Level"})
        st.dataframe(percentage_data, use_container_width=True, hide_index=True)

        st.subheader("Fraud Probability Distribution")
        fig = px.histogram(
            transactions,
            x="Fraud Probability (%)",
            nbins=20,
            title="Predicted Fraud Probability",
        )
        st.plotly_chart(fig, use_container_width=True)

        st.subheader("Transaction Amount Analysis")
        fig = px.histogram(
            transactions,
            x="Amount",
            color="Result",
            barmode="overlay",
            title="Transaction Amounts by Prediction Result",
        )
        st.plotly_chart(fig, use_container_width=True)

        st.subheader("Filter Transactions by Risk")
        selected_risk = st.selectbox(
            "Select Risk Level",
            ["All", "High Risk", "Medium Risk", "Low Risk"],
        )
        if selected_risk == "All":
            filtered_transactions = transactions
        else:
            filtered_transactions = transactions[
                transactions["Risk Level"] == selected_risk
            ]

        st.write(f"Showing {len(filtered_transactions)} transactions")
        st.dataframe(
            filtered_transactions,
            use_container_width=True,
            hide_index=True,
        )

        filtered_csv = filtered_transactions.to_csv(index=False).encode("utf-8")
        st.download_button(
            "Download Filtered CSV",
            data=filtered_csv,
            file_name="filtered_fraud_transactions.csv",
            mime="text/csv",
        )


# --------------------------------------------------
# CSV UPLOAD AND DATA QUALITY
# --------------------------------------------------
elif page == "CSV Upload":
    st.title("Transaction CSV Upload")
    uploaded_file = st.file_uploader(
        "Choose a CSV file",
        type=["csv"],
        key="csv_quality_upload",
    )

    if uploaded_file is not None:
        try:
            uploaded_data = pd.read_csv(uploaded_file)
            st.success("CSV uploaded successfully!")

            col1, col2 = st.columns(2)
            col1.metric("Total Rows", len(uploaded_data))
            col2.metric("Total Columns", len(uploaded_data.columns))

            st.subheader("Dataset Preview")
            st.dataframe(uploaded_data.head(20), use_container_width=True)

            st.subheader("Dataset Summary Statistics")
            st.dataframe(
                uploaded_data.describe(include="all").transpose(),
                use_container_width=True,
            )

            st.subheader("Missing Values Analysis")
            missing_values = uploaded_data.isnull().sum()
            missing_report = pd.DataFrame(
                {
                    "Column": missing_values.index,
                    "Missing Values": missing_values.values,
                }
            )
            st.dataframe(missing_report, use_container_width=True)
            st.metric("Total Missing Values", int(missing_values.sum()))

            st.subheader("Duplicate Transaction Analysis")
            duplicate_count = int(uploaded_data.duplicated().sum())
            st.metric("Duplicate Transactions", duplicate_count)

            if duplicate_count:
                st.warning("Duplicate records were found.")
            else:
                st.success("No duplicate records were found.")

            cleaned_data = uploaded_data.drop_duplicates()
            cleaned_csv = cleaned_data.to_csv(index=False).encode("utf-8")
            st.download_button(
                "Download Cleaned CSV",
                data=cleaned_csv,
                file_name="cleaned_transactions.csv",
                mime="text/csv",
            )

            st.subheader("Column Names")
            st.write(list(uploaded_data.columns))

        except Exception as error:
            st.error(f"Unable to read CSV: {error}")
    else:
        st.info("Upload a CSV file to inspect its data quality.")


# --------------------------------------------------
# ADMIN LOGIN
# --------------------------------------------------
elif page == "Admin Login":
    st.title("Admin Login")
    st.write("Sign in to access the FraudGuard AI administration panel.")

    username = st.text_input("Admin Username")
    password = st.text_input("Admin Password", type="password")

    if st.button("Login", type="primary"):
        try:
            admin_secrets = st.secrets["admin"]
            correct_username = admin_secrets["username"]
            correct_password = admin_secrets["password"]

            if username == correct_username and password == correct_password:
                st.session_state["admin_logged_in"] = True
                st.success("Admin login successful!")
            else:
                st.error("Invalid username or password.")
        except Exception:
            st.error(
                "Admin credentials are not configured. "
                "Check .streamlit/secrets.toml."
            )

    if st.session_state.get("admin_logged_in", False):
        st.success("You are signed in as the administrator.")
        if st.button("Logout"):
            st.session_state["admin_logged_in"] = False
            st.rerun()


# --------------------------------------------------
# ADMIN DASHBOARD
# --------------------------------------------------
elif page == "Admin Dashboard":
    st.title("FraudGuard AI - Admin Dashboard")

    if not st.session_state.get("admin_logged_in", False):
        st.warning("Please log in as Admin to access this page.")
    else:
        st.success("Welcome to the Admin Dashboard!")

        if st.button("Logout from Admin Dashboard"):
            st.session_state["admin_logged_in"] = False
            st.rerun()

        c1, c2 = st.columns(2)
        c1.metric("Model Status", "Loaded" if model is not None else "Unavailable")
        c2.metric("Application Status", "Active")

        transactions = get_prediction_results()
        if transactions is None:
            st.info("No transaction predictions are available yet.")
        else:
            total = len(transactions)
            fraud_count = int((transactions["Prediction"] == 1).sum())
            legitimate_count = total - fraud_count
            average_probability = (
                float(transactions["Fraud Probability (%)"].mean())
                if total else 0.0
            )

            st.subheader("Transaction Overview")
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Total Transactions", total)
            c2.metric("Potential Fraud", fraud_count)
            c3.metric("Legitimate Transactions", legitimate_count)
            c4.metric("Average Fraud Probability", f"{average_probability:.2f}%")

            st.subheader("Risk Summary")

        if "prediction_results" in st.session_state:
            admin_df = st.session_state["prediction_results"]

            if "Risk Level" in admin_df.columns:
                risk_counts = (
                    admin_df["Risk Level"]
                    .astype(str)
                    .str.strip()
                    .str.title()
                    .value_counts()
                    .reindex(["High", "Medium", "Low"], fill_value=0)
                )

                col1, col2, col3 = st.columns(3)

                col1.metric("High Risk", int(risk_counts["High"]))
                col2.metric("Medium Risk", int(risk_counts["Medium"]))
                col3.metric("Low Risk", int(risk_counts["Low"]))

                st.bar_chart(risk_counts)

            else:
                st.warning("Risk Level column is not available.")

        else:
            st.info("Upload transactions and run fraud prediction to view the risk summary.")

            st.subheader("Fraud Transactions Download")

        if "prediction_results" in st.session_state:
            admin_df = st.session_state["prediction_results"]

            if "Prediction" in admin_df.columns:
                fraud_transactions = admin_df[
                    admin_df["Prediction"] == 1
                ]

                st.write(
                    f"Total Fraud Transactions: {len(fraud_transactions)}"
                )

                if not fraud_transactions.empty:
                    fraud_csv = fraud_transactions.to_csv(
                        index=False
                    ).encode("utf-8")

                    st.download_button(
                        label="Download Fraud Transactions CSV",
                        data=fraud_csv,
                        file_name="fraud_transactions.csv",
                        mime="text/csv"
                    )
                else:
                    st.info("No fraud transactions found.")
            else:
                st.warning("Prediction column is not available.")
        else:
            st.info("Run fraud prediction first to download fraud transactions.")

            st.subheader("Fraud Alert Summary")
            if fraud_count:
                st.error(
                    f"Alert: {fraud_count} transactions were flagged as potential fraud."
                )
            else:
                st.success("No transactions were flagged as potential fraud.")

            st.subheader("Recent Potential Fraud Transactions")
            fraud_transactions = transactions[
                transactions["Prediction"] == 1
            ]

            if fraud_transactions.empty:
                st.info("No potential fraud transactions to display.")
            else:
                st.dataframe(
                    fraud_transactions.head(10),
                    use_container_width=True,
                    hide_index=True,
                )
                fraud_csv = fraud_transactions.to_csv(index=False).encode("utf-8")
                st.download_button(
                    "Download Fraud Transactions",
                    data=fraud_csv,
                    file_name="fraud_transactions.csv",
                    mime="text/csv",
                )


# --------------------------------------------------
# PDF REPORTS
# --------------------------------------------------
elif page == "PDF Reports":
    st.title("FraudGuard AI - PDF Reports")
    st.write("Generate a PDF summary from your latest prediction results.")

    transactions = get_prediction_results()
    if transactions is None:
        st.info(
            "No prediction results are available. "
            "Upload a compatible transaction CSV in Fraud Prediction first."
        )
    else:
        total = len(transactions)
        fraud_count = int((transactions["Prediction"] == 1).sum())
        legitimate_count = total - fraud_count
        average_probability = (
            float(transactions["Fraud Probability (%)"].mean())
            if total else 0.0
        )

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Total Transactions", total)
        c2.metric("Potential Fraud", fraud_count)
        c3.metric("Legitimate", legitimate_count)
        c4.metric("Average Fraud Probability", f"{average_probability:.2f}%")

        st.subheader("Report Preview")
        st.dataframe(transactions.head(10), use_container_width=True, hide_index=True)

        pdf_report = create_pdf_report(transactions)
        st.download_button(
            "Download FraudGuard AI PDF Report",
            data=pdf_report,
            file_name="FraudGuard_AI_Report.pdf",
            mime="application/pdf",
            type="primary",
        )

st.sidebar.divider()
st.sidebar.caption("FraudGuard AI | Academic Project Prototype")
st.sidebar.caption("Predictions are estimates and require human review.")

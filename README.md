# Customer Churn Prediction & Retention Analytics

An interactive machine learning application for analyzing customer churn, identifying high-risk customers, and supporting data-driven customer retention strategies.

Built with Python, Scikit-learn, Pandas, and Streamlit.

## Overview

Customer churn refers to customers discontinuing their relationship with a company or service. Identifying customers who are likely to churn helps businesses understand customer behavior and develop targeted retention strategies.

This project combines a machine learning-based churn prediction model with an interactive analytics dashboard to explore churn patterns, segment customers by risk, and prioritize retention efforts.

## Features

### 📊 Customer Retention Dashboard
- Total customer count
- Overall churn rate
- Average monthly charges
- Monthly charges associated with churned customers
- Customer churn distribution visualizations
- Churn analysis by contract type
- Customer insights based on monthly charges and tenure

### 🎯 Customer Risk Segmentation
- Predict churn probabilities for customers in the dataset
- Categorize customers into Low, Medium, and High Risk segments
- Identify high-risk customers for retention prioritization
- Filter high-risk customers by contract type
- View prioritized customer information
- Download high-risk customer data as a CSV report

### 🔮 Customer Churn Prediction
- Interactive customer information input form
- Predict the probability of customer churn
- Classify customers according to the model's decision threshold
- Display personalized retention recommendations based on customer characteristics

## Machine Learning

The project uses a trained Scikit-learn machine learning pipeline for churn prediction.

The existing model artifact is loaded by the application to generate churn probabilities and predictions.

Model evaluation conducted in the original project included:

| Model | Accuracy | ROC-AUC |
|---|---:|---:|
| Logistic Regression | 80.55% | 0.8421 |
| Random Forest | 78.28% | 0.8232 |

Logistic Regression demonstrated better performance in the reported model comparison.

## Risk Segmentation

Customers are segmented according to their predicted churn probability:

| Risk Category | Predicted Churn Probability |
|---|---|
| Low Risk | ≤ 30% |
| Medium Risk | > 30% to 60% |
| High Risk | > 60% |

The segmentation enables businesses to prioritize customers who may require retention interventions.

## Dataset

The project uses the IBM Telco Customer Churn dataset.

The dataset contains customer information such as:

- Customer tenure
- Contract type
- Monthly charges
- Total charges
- Internet and phone services
- Payment methods
- Customer demographics
- Churn status

Dataset file:

`WA_Fn-UseC_-Telco-Customer-Churn.csv`

## Technology Stack

- Python
- Pandas
- Scikit-learn
- Joblib
- Streamlit

## Project Structure

```text
customer-churn-prediction/
│
├── app.py
├── customer_churn_model.pkl
├── WA_Fn-UseC_-Telco-Customer-Churn.csv
├── requirements.txt
├── README.md
│
└── screenshots/
    ├── overview.png
    ├── risk_segmentation.png
    └── churn_prediction.png
```

## Installation and Setup

### 1. Clone the repository

```bash
git clone https://github.com/YOUR-USERNAME/customer-churn-prediction.git
```

### 2. Navigate to the project directory

```bash
cd customer-churn-prediction
```

### 3. Create a virtual environment

```bash
python -m venv .venv
```

### 4. Activate the virtual environment

Windows:

```bash
.venv\Scripts\activate
```

### 5. Install dependencies

```bash
pip install -r requirements.txt
```

### 6. Run the application

```bash
streamlit run app.py
```

The application will open in your browser.

## Application Pages

### Overview
Provides an overview of customer churn metrics, contract-based churn analysis, and customer insights.

### Risk Segmentation
Identifies customers with elevated churn probabilities, categorizes risk levels, and provides a prioritized list of high-risk customers.

### Predict Churn
Allows users to enter customer details and obtain a churn prediction with probability-based insights.

## Business Value

The application can help businesses:

- Identify customers who may be at risk of leaving
- Understand churn patterns across contract types
- Prioritize retention efforts
- Explore customer characteristics associated with churn
- Support data-driven retention decisions

## Limitations

- Predictions depend on the trained model and the data used to develop it.
- Risk categories are based on probability thresholds and are not guarantees of customer behavior.
- Monthly charges associated with high-risk customers represent potential exposure, not confirmed revenue loss.
- The application is intended for demonstration and analytical purposes and should not be treated as a production customer decision system.

## Disclaimer

This project is intended for educational, portfolio, and demonstration purposes.

---

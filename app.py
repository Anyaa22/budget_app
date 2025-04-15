import streamlit as st
import pandas as pd
import plotly.express as px
import joblib
from sklearn.model_selection import train_test_split

# Set Streamlit page config
st.set_page_config(page_title="💰 Smart Budget & Savings Predictor", layout="wide")

# Load dataset
@st.cache_data
def load_data():
    df = pd.read_csv("rural_budget_allocation_dataset.csv")
    return df

df = load_data()

# Load trained model
@st.cache_resource
def load_model():
    return joblib.load("prediction_model2.pkl")

model = load_model()

# Show app title
st.title("💰 Smart Budget & Savings Predictor")

# Dataset Viewer
with st.expander("📄 View Dataset"):
    st.dataframe(df, use_container_width=True)

# Stats
st.markdown("### 📊 Summary Statistics")
st.write(df.describe())

# Charts
st.markdown("### 📈 Expense vs Budget Overview")
expense_cols = ['HousingExpense', 'TransportationExpense', 'FoodExpense', 'UtilitiesExpense', 'EntertainmentExpense']
budget_cols = ['HousingBudget', 'TransportationBudget', 'FoodBudget', 'UtilitiesBudget', 'EntertainmentBudget']

col1, col2 = st.columns(2)
col1.subheader("Actual Expenses")
col1.bar_chart(df[expense_cols])

col2.subheader("Budget Allocations")
col2.bar_chart(df[budget_cols])

# Predict Actual Savings using loaded model
st.markdown("### 🔮 Predict Actual Savings")

# Get exact feature list from model
expected_features = list(model.feature_names_in_)  # Ensures correct order & match

# Prepare features and target
X = df[expected_features]
y = df['Savings']

# Split for test evaluation
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Sidebar Input
st.sidebar.header("📥 Input Values for Prediction")

def user_input():
    input_data = {}

    st.sidebar.subheader("💵 Income")
    input_data['Income'] = st.sidebar.number_input("Income", min_value=0, value=int(df['Income'].mean()))

    st.sidebar.subheader("🏠 Expenses")
    input_data['HousingExpense'] = st.sidebar.number_input("Housing Expense", min_value=0, value=int(df['HousingExpense'].mean()))
    input_data['TransportationExpense'] = st.sidebar.number_input("Transportation Expense", min_value=0, value=int(df['TransportationExpense'].mean()))
    input_data['FoodExpense'] = st.sidebar.number_input("Food Expense", min_value=0, value=int(df['FoodExpense'].mean()))
    input_data['UtilitiesExpense'] = st.sidebar.number_input("Utilities Expense", min_value=0, value=int(df['UtilitiesExpense'].mean()))
    input_data['EntertainmentExpense'] = st.sidebar.number_input("Entertainment Expense", min_value=0, value=int(df['EntertainmentExpense'].mean()))

    st.sidebar.subheader("📊 Budget Allocations")
    input_data['HousingBudget'] = st.sidebar.number_input("Housing Budget", min_value=0, value=int(df['HousingBudget'].mean()))
    input_data['TransportationBudget'] = st.sidebar.number_input("Transportation Budget", min_value=0, value=int(df['TransportationBudget'].mean()))
    input_data['FoodBudget'] = st.sidebar.number_input("Food Budget", min_value=0, value=int(df['FoodBudget'].mean()))
    input_data['UtilitiesBudget'] = st.sidebar.number_input("Utilities Budget", min_value=0, value=int(df['UtilitiesBudget'].mean()))
    input_data['EntertainmentBudget'] = st.sidebar.number_input("Entertainment Budget", min_value=0, value=int(df['EntertainmentBudget'].mean()))
    input_data['SavingsBudget'] = st.sidebar.number_input("Savings Budget", min_value=0, value=int(df['SavingsBudget'].mean()))

    return pd.DataFrame([input_data])

input_df = user_input()

# Show the input used for prediction
st.write("🔍 Prediction Input Preview", input_df)

# Prediction button
if st.sidebar.button("📊 Predict Now"):
    prediction = model.predict(input_df)[0]
    st.sidebar.success(f"💰 Predicted Actual Savings: ${prediction:.2f}")

# Model Performance Chart
st.markdown("### 📊 Model Performance on Test Set")
test_predictions = model.predict(X_test)
comparison_df = pd.DataFrame({
    "Actual": y_test,
    "Predicted": test_predictions
})
st.line_chart(comparison_df.reset_index(drop=True))

# Download Button
with st.expander("⬇️ Download Sample Prediction Data"):
    full_predictions = df.copy()
    full_predictions["PredictedSavings"] = model.predict(X)
    csv = full_predictions.to_csv(index=False).encode("utf-8")
    st.download_button("Download Full Predictions", data=csv, file_name="savings_predictions.csv", mime="text/csv")

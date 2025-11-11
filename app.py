import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

st.set_page_config(page_title="AI Business Dashboard", layout="wide")

st.title("🚀 AI-Based Business Analytics Dashboard")

# ---------------------------------------------
# AI MODEL SECTION
# ---------------------------------------------
st.header("🤖 AI Model for Demand Prediction")

uploaded_file = st.file_uploader("Upload Dataset (CSV/Excel)", type=["csv", "xlsx"])

if uploaded_file is not None:
    try:
        if uploaded_file.name.endswith(".csv"):
            df = pd.read_csv(uploaded_file)
        else:
            df = pd.read_excel(uploaded_file)

        st.write("### Preview of Uploaded Data:")
        st.dataframe(df.head())

        columns = list(df.columns)
        target_col = st.selectbox("Select Target Column (Demand/Output)", columns)
        feature_cols = st.multiselect("Select Feature Columns (Input Variables)", [c for c in columns if c != target_col])

        if feature_cols:
            X = df[feature_cols]
            y = df[target_col]

            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

            model = LinearRegression()
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)

            r2 = r2_score(y_test, y_pred)

            st.write("### Model Evaluation:")
            st.write(f"✅ **R² Score:** {r2:.2f}")

            # Plot predictions
            fig, ax = plt.subplots(figsize=(7, 4))
            ax.scatter(y_test, y_pred, color="purple")
            ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], color="black", linestyle="--")
            ax.set_xlabel("Actual Values")
            ax.set_ylabel("Predicted Values")
            ax.set_title("Actual vs Predicted (AI Model)")
            st.pyplot(fig)

            # Demand insight
            mean_pred = np.mean(y_pred)
            high_demand_index = np.argmax(y_pred)
            high_demand_product = X_test.iloc[high_demand_index]
            st.write("### 🔍 AI Insights:")
            st.success(f"Highest predicted demand corresponds to: **{high_demand_product.to_dict()}**")
            st.info(f"Average predicted demand across dataset: **{mean_pred:.2f}**")

    except Exception as e:
        st.error(f"Error processing data: {e}")

# ---------------------------------------------
# BREAK-EVEN ANALYSIS
# ---------------------------------------------
st.header("💰 Break-even Analysis")

uploaded_file_be = st.file_uploader("Upload Break-even Data (CSV/Excel)", type=["csv", "xlsx"], key="break_even")

if uploaded_file_be is not None:
    try:
        if uploaded_file_be.name.endswith(".csv"):
            df_be = pd.read_csv(uploaded_file_be)
        else:
            df_be = pd.read_excel(uploaded_file_be)

        st.write("### Uploaded Data Preview:")
        st.dataframe(df_be.head())

        fixed_cost = df_be["Fixed_Cost"].iloc[0]
        variable_cost = df_be["Variable_Cost"].iloc[0]
        selling_price = df_be["Selling_Price"].iloc[0]

        # Slider for max quantity range
        max_qty = st.slider("Select Maximum Quantity Range", 500, 5000, 1000, step=100)
        qty = np.arange(0, max_qty, 10)

        total_cost = fixed_cost + variable_cost * qty
        total_revenue = selling_price * qty
        breakeven_point = fixed_cost / (selling_price - variable_cost)

        # Plot the graph
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.plot(qty, total_cost, label="Total Cost", color="red")
        ax.plot(qty, total_revenue, label="Total Revenue", color="green")
        ax.axvline(x=breakeven_point, color="blue", linestyle="--", label=f"Break-even Point ({breakeven_point:.2f} units)")
        ax.set_xlabel("Quantity Produced/Sold")
        ax.set_ylabel("Revenue / Cost (₹)")
        ax.set_title("Break-even Analysis")
        ax.legend()
        st.pyplot(fig)

        # Text summary
        st.subheader("📈 Break-even Summary")
        st.write(f"At approximately **{breakeven_point:.2f} units**, the company reaches its break-even point.")

        if breakeven_point < qty[-1]:
            st.success(f"Beyond {breakeven_point:.2f} units, **the company will operate in profit.**")
        else:
            st.warning(f"The break-even point ({breakeven_point:.2f} units) lies beyond the selected range — **the company will run in loss** within this range.")

    except Exception as e:
        st.error(f"Error processing break-even data: {e}")



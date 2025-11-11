import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error

st.set_page_config(page_title="Break-even & Demand Forecast Tool", layout="wide")

st.title("Break-even Analysis &  AI-based Demand Forecasting Tool - IE Project")
st.markdown("Upload your data files below to perform break-even and demand analysis.")

# --- SIDEBAR ---
st.sidebar.header("📁 Upload CSV Files")

# Upload for Break-even
break_even_file = st.sidebar.file_uploader("Upload Break-even data (CSV)", type=['csv'])
# Upload for Demand
demand_file = st.sidebar.file_uploader("Upload Demand dataset (CSV)", type=['csv'])

# --- BREAK-EVEN ANALYSIS ---
st.header("📉 Break-even Analysis")

if break_even_file is not None:
    try:
        df = pd.read_csv(break_even_file)
        st.subheader("Uploaded Data")
        st.dataframe(df)

        fixed_cost = df['Fixed_Cost'].iloc[0]
        variable_cost = df['Variable_Cost_per_Unit'].iloc[0]
        selling_price = df['Selling_Price_per_Unit'].iloc[0]

        break_even_units = fixed_cost / (selling_price - variable_cost)
        st.write(f"**Break-even Point:** {break_even_units:.2f} units")

        units = np.linspace(0, break_even_units * 2, 100)
        total_cost = fixed_cost + variable_cost * units
        total_revenue = selling_price * units

        fig, ax = plt.subplots()
        ax.plot(units, total_cost, label='Total Cost', color='red')
        ax.plot(units, total_revenue, label='Total Revenue', color='green')
        ax.axvline(break_even_units, color='blue', linestyle='--', label='Break-even Point')
        ax.set_xlabel("Units Sold")
        ax.set_ylabel("Cost / Revenue")
        ax.set_title("Break-even Chart")
        ax.legend()
        st.pyplot(fig)

        st.info(f"At less than {break_even_units:.2f} units, company runs **in loss**. Beyond that, it makes **profit.**")

    except Exception as e:
        st.error(f"Error reading break-even data: {e}")
else:
    st.warning("Upload break_even.csv to see analysis.")

# --- DEMAND ANALYSIS ---
st.header("🧠 AI-based Demand Forecasting")

if demand_file is not None:
    try:
        demand_df = pd.read_csv(demand_file,encoding='latin1')
        st.subheader("Uploaded Demand Dataset")
        st.dataframe(demand_df.head())

        # Auto select numeric columns
        numeric_cols = demand_df.select_dtypes(include=np.number).columns.tolist()

        if len(numeric_cols) < 2:
            st.error("Dataset must contain at least two numeric columns (e.g., marketing spend, sales, etc.).")
        else:
            x_col = st.selectbox("Select Feature (X - input)", numeric_cols, index=0)
            y_col = st.selectbox("Select Target (Y - output)", numeric_cols, index=1)

            X = demand_df[[x_col]].values
            y = demand_df[y_col].values

            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            model = LinearRegression()
            model.fit(X_train, y_train)

            y_pred = model.predict(X_test)

            r2 = r2_score(y_test, y_pred)
            rmse = np.sqrt(mean_squared_error(y_test, y_pred))

            st.write(f"**R² Score:** {r2:.3f}")
            st.write(f"**RMSE:** {rmse:.3f}")
            st.subheader("📊 Product Demand Insights")

# Check if necessary columns exist
if 'Product Name' in demand_df.columns and 'Quantity' in demand_df.columns:
    demand_summary = (
        demand_df.groupby('Product Name')['Quantity']
        .sum()
        .sort_values(ascending=False)
        .head(5)
        .reset_index()
    )

    st.write("### 🔝 Top 5 High-Demand Products")
    st.dataframe(demand_summary)

    top_product = demand_summary.iloc[0]['Product Name']
    top_demand = demand_summary.iloc[0]['Quantity']

    st.success(f"🏆 The highest demand is for **{top_product}** with total quantity {top_demand}.")
else:
    st.warning("Couldn't find 'Product Name' or 'Quantity' column to compute demand insights.")

            fig, ax = plt.subplots()
            ax.scatter(y_test, y_pred, alpha=0.7, color='purple')
            ax.set_xlabel("Actual Demand")
            ax.set_ylabel("Predicted Demand")
            ax.set_title("Actual vs Predicted Demand")
            st.pyplot(fig)

            st.success("AI model successfully trained and tested!")

    except Exception as e:
        st.error(f"Error during AI demand modeling: {e}")
else:
    st.warning("Upload demand_data.csv (like the Kaggle Superstore dataset) to run AI analysis.")



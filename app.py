import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error

st.set_page_config(page_title="Break-even & Demand Forecast Tool", layout="wide")

st.title("📊 Break-even Analysis & 🧠 AI-based Demand Forecasting Tool")
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
        variable_cost = df['Variable_Cost'].iloc[0]
        selling_price = df['Selling_Price'].iloc[0]

        break_even_units = fixed_cost / (selling_price - variable_cost)
        st.write(f"**Break-even Point:** {break_even_units:.2f} units")

        units = np.linspace(0, break_even_units * 2, 100)
        total_cost = fixed_cost + variable_cost * units
        total_revenue = selling_price * units

        fig, ax = plt.subplots()
        ax.plot(units, total_cost, label='Total Cost', color='red')
        ax.plot(units, total_revenue, label='Total Revenue', color='green')
        ax.axvline(break_even_units, color='blue', linestyle='--', label=f'Break-even Point ({break_even_units:.2f} units)')
        ax.set_xlabel("Units Sold")
        ax.set_ylabel("Cost / Revenue")
        ax.set_title("Break-even Chart")
        ax.legend()
        st.pyplot(fig)

        # Summary text below the graph
        st.markdown("### 📈 Break-even Summary")
        st.info(f"At approximately **{break_even_units:.2f} units**, the company reaches its break-even point.")

        if break_even_units < units[-1] / 2:
            st.success(f"✅ Beyond {break_even_units:.2f} units, **the company will operate in profit.**")
        else:
            st.warning(f"⚠️ The break-even point ({break_even_units:.2f} units) lies beyond the expected range — **the company may face loss** in current production levels.")

    except Exception as e:
        st.error(f"Error reading break-even data: {e}")
else:
    st.warning("Upload break_even.csv to see analysis.")

# --- DEMAND ANALYSIS ---
st.header("🧠 AI-based Demand Forecasting")

if demand_file is not None:
    try:
        demand_df = pd.read_csv(demand_file, encoding='latin1')
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

            fig, ax = plt.subplots()
            ax.scatter(y_test, y_pred, alpha=0.7, color='purple')
            ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=2)
            ax.set_xlabel("Actual Demand")
            ax.set_ylabel("Predicted Demand")
            ax.set_title("Actual vs Predicted Demand")
            st.pyplot(fig)

            # Show highest demand prediction insight
            highest_demand_index = np.argmax(y_pred)
            st.markdown("### 🔍 AI Insights")
            st.success(f"Highest predicted demand occurs when **{x_col} = {X_test[highest_demand_index][0]:.2f}**, "
                       f"with a predicted demand of **{y_pred[highest_demand_index]:.2f} units.**")

    except Exception as e:
        st.error(f"Error during AI demand modeling: {e}")
else:
    st.warning("Upload demand_data.csv (like the Kaggle Superstore dataset) to run AI analysis.")



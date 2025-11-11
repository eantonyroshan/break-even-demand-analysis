import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error

st.set_page_config(page_title="Break-even & Demand Forecast Tool", layout="wide")

st.title(" An automated tool to perform Break-even Analysis & Demand Forecasting with Ai")
st.markdown("Upload your data files below to perform break-even and demand analysis.")

# --- SIDEBAR ---
st.sidebar.header("📁 Upload CSV Files")

# Upload for Break-even
break_even_file = st.sidebar.file_uploader("Upload Break-even data (CSV)", type=['csv'])
# Upload for Demand
demand_file = st.sidebar.file_uploader("Upload Demand dataset (CSV)", type=['csv'])

# ---------------------------------------------
# BREAK-EVEN ANALYSIS
# ---------------------------------------------
st.header("📉 Break-even Analysis")

if break_even_file is not None:
    try:
        df = pd.read_csv(break_even_file)
        st.subheader("Uploaded Data")
        st.dataframe(df)

        fixed_cost = df['Fixed_Cost'].iloc[0]
        variable_cost = df['Variable_Cost_per_Unit'].iloc[0]
        selling_price = df['Selling_Price_per_Unit'].iloc[0]

        # Slider to adjust the maximum sales quantity
        max_qty = st.slider("Select Maximum Selling Quantity Range", min_value=500, max_value=10000, value=2000, step=100)

        # Compute break-even point
        break_even_units = fixed_cost / (selling_price - variable_cost)
        st.write(f"### 💡 Current Break-even Point: {break_even_units:.2f} units")

        # Compute ideal break-even point (assuming 10% increase in price or 10% decrease in cost)
        ideal_selling_price = selling_price * 1.1
        ideal_variable_cost = variable_cost * 0.9
        ideal_break_even_units = fixed_cost / (ideal_selling_price - ideal_variable_cost)
        st.write(f"### 🌟 Ideal Break-even Point (if efficiency improves): {ideal_break_even_units:.2f} units")

        # Generate chart
        qty = np.linspace(0, max_qty, 200)
        total_cost = fixed_cost + variable_cost * qty
        total_revenue = selling_price * qty

        fig, ax = plt.subplots(figsize=(8, 5))
        ax.plot(qty, total_cost, label='Total Cost', color='red')
        ax.plot(qty, total_revenue, label='Total Revenue', color='green')
        ax.axvline(break_even_units, color='blue', linestyle='--', label=f'Break-even ({break_even_units:.0f} units)')
        ax.axvline(ideal_break_even_units, color='orange', linestyle='--', label=f'Ideal Break-even ({ideal_break_even_units:.0f} units)')
        ax.set_xlabel("Units Sold")
        ax.set_ylabel("Cost / Revenue")
        ax.set_title("Break-even Chart with Adjustable Range")
        ax.legend()
        st.pyplot(fig)

        # Summary text
        st.markdown("### 📈 Break-even Summary")
        st.info(f"At approximately **{break_even_units:.2f} units**, the company currently breaks even.")
        if break_even_units < qty[-1]:
            st.success(f"✅ Beyond {break_even_units:.2f} units, the company will operate in profit.")
        else:
            st.warning(f"⚠️ Within the current production range (up to {max_qty} units), the company still runs in loss.")

    except Exception as e:
        st.error(f"Error reading break-even data: {e}")
else:
    st.warning("Upload break_even.csv to see analysis.")

# ---------------------------------------------
# AI-BASED DEMAND FORECASTING
# ---------------------------------------------
st.header("🧠 AI-based Demand Forecasting")

if demand_file is not None:
    try:
        demand_df = pd.read_csv(demand_file, encoding='latin1')
        st.subheader("Uploaded Demand Dataset")
        st.dataframe(demand_df.head())

        # Auto select numeric columns
        numeric_cols = demand_df.select_dtypes(include=np.number).columns.tolist()
        non_numeric_cols = demand_df.select_dtypes(exclude=np.number).columns.tolist()

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

            # Plot actual vs predicted
            fig, ax = plt.subplots(figsize=(6, 4))
            ax.scatter(y_test, y_pred, alpha=0.7, color='purple')
            ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=2)
            ax.set_xlabel("Actual Demand")
            ax.set_ylabel("Predicted Demand")
            ax.set_title("Actual vs Predicted Demand")
            st.pyplot(fig)

            st.success("✅ AI model successfully trained and tested!")

            # Show top 5 highest predicted demand items
            demand_df = demand_df.copy()
            demand_df['Predicted_Demand'] = model.predict(demand_df[[x_col]])

            st.markdown("### 🔍 Top 5 Products with Highest Predicted Demand")
            if any(col.lower() in ['product', 'item', 'name'] for col in non_numeric_cols):
                id_col = [col for col in non_numeric_cols if col.lower() in ['product', 'item', 'name']][0]
                top5 = demand_df[[id_col, 'Predicted_Demand']].sort_values(by='Predicted_Demand', ascending=False).head(5)
                st.dataframe(top5)
            else:
                top5 = demand_df[['Predicted_Demand']].sort_values(by='Predicted_Demand', ascending=False).head(5)
                st.dataframe(top5)

    except Exception as e:
        st.error(f"Error during AI demand modeling: {e}")
else:
    st.warning("Upload demand_data.csv (like the Kaggle Superstore dataset) to run AI analysis.")



import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

# -------------------------------
# Streamlit Page Config
# -------------------------------
st.set_page_config(page_title="Break-even & Demand Analysis", layout="wide")
st.title("Break-even and AI-based Demand Forecasting Tool - IE Project")
st.markdown("An automated tool performs **break-even analysis** and **AI-based demand forecasting** using uploaded datasets.")

# -------------------------------
# Break-even Analysis Section
# -------------------------------
st.header("💰 Break-even Analysis")

breakeven_file = st.file_uploader("Upload Break-even data (CSV/Excel)", type=["csv", "xlsx"])

if breakeven_file is not None:
    try:
        if breakeven_file.name.endswith(".csv"):
            data = pd.read_csv(breakeven_file)
        else:
            data = pd.read_excel(breakeven_file)
        
        st.write("### Preview of Uploaded Data")
        st.dataframe(data.head())

        # Expecting columns: Fixed_Cost, Variable_Cost, Selling_Price, Quantity
        fixed_cost = data["Fixed_Cost"].iloc[0]
        variable_cost = data["Variable_Cost"].iloc[0]
        selling_price = data["Selling_Price"].iloc[0]
        quantity = np.arange(0, data["Quantity"].iloc[-1] + 1)

        total_cost = fixed_cost + variable_cost * quantity
        total_revenue = selling_price * quantity
        breakeven_point = fixed_cost / (selling_price - variable_cost)

        st.write(f"**Break-even Quantity:** {breakeven_point:.2f} units")

        fig, ax = plt.subplots(figsize=(8, 5))
        ax.plot(quantity, total_cost, label="Total Cost", color="red")
        ax.plot(quantity, total_revenue, label="Total Revenue", color="green")
        ax.axvline(x=breakeven_point, color="blue", linestyle="--", label="Break-even Point")
        ax.set_xlabel("Quantity")
        ax.set_ylabel("Money (₹)")
        ax.set_title("Break-even Analysis")
        ax.legend()
        st.pyplot(fig)

        if quantity[-1] < breakeven_point:
            st.warning("🚨 Company will run at a loss — increase sales or reduce cost.")
        else:
            st.success("✅ Profit generated beyond the break-even point!")

    except Exception as e:
        st.error(f"Error reading break-even data: {e}")

# -------------------------------
# Demand Forecasting (AI Model)
# -------------------------------
st.header("🤖 AI-based Demand Forecasting")

demand_file = st.file_uploader("Upload Demand Dataset (CSV)", type=["csv"])

if demand_file is not None:
    try:
        data = pd.read_csv(demand_file, encoding='latin1')
        st.write("### Preview of Demand Dataset")
        st.dataframe(data.head())

        # Automatically detect features (X) and target (y)
        numeric_cols = data.select_dtypes(include=[np.number]).columns.tolist()

        if len(numeric_cols) < 2:
            st.error("Dataset must have at least two numeric columns.")
        else:
            target_col = st.selectbox("Select Target Column (Demand)", numeric_cols)
            feature_cols = [col for col in numeric_cols if col != target_col]

            X = data[feature_cols]
            y = data[target_col]

            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

            model = LinearRegression()
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)

            r2 = r2_score(y_test, y_pred)
            st.write(f"**R² Score:** {r2:.2f}")

            # ----- Plot actual vs predicted demand -----
            fig, ax = plt.subplots(figsize=(7, 4))
            ax.scatter(y_test, y_pred, color='lightblue')
            ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
            ax.set_xlabel("Actual Demand")
            ax.set_ylabel("Predicted Demand")
            ax.set_title("AI Model: Actual vs Predicted Demand")
            st.pyplot(fig)

            # ----- Find which product has highest predicted demand -----
            if 'product' in data.columns or 'Product' in data.columns:
                product_col = 'product' if 'product' in data.columns else 'Product'
                predicted_df = data.copy()
                predicted_df['Predicted_Demand'] = model.predict(X)
                top_products = predicted_df.groupby(product_col)['Predicted_Demand'].mean().sort_values(ascending=False).head(5)
                st.subheader("🔥 Top Products by Predicted Demand")
                st.write(top_products)
            else:
                st.info("Column 'product' not found — skipping product-level analysis.")

    except Exception as e:
        st.error(f"Error during AI model training: {e}")

# -------------------------------
# Footer
# -------------------------------
st.markdown("---")
st.caption("ROSHAN & VISHAAL | Powered by Streamlit + Scikit-Learn")


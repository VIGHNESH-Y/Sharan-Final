import pandas as pd
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error, r2_score
import xgboost as xgb
import joblib

# ------------------- DATA LOADING -------------------
@st.cache_data
def load_data():
    df = pd.read_csv("final_sales_data.csv")
    return df

# Load data
df = load_data()

# ------------------- STREAMLIT APP TITLE -------------------
st.markdown("""
    <h1 style='text-align: center; color: #2E86C1;'>📊 Business Sales Forecasting Tool</h1>
    <p style='text-align: center; color: gray;'>Predict your future sales with Machine Learning</p>
    <hr style='border:1px solid #2E86C1;'>
    """, unsafe_allow_html=True)

# ------------------- EDA -------------------
st.subheader("🔎 Explore Your Sales Data")
st.dataframe(df.head())

# ------------------- FEATURES & TARGET -------------------
df['Date'] = pd.to_datetime(df['Date'])
df['Month'] = df['Date'].dt.month

features = ['Price', 'Promotion', 'Holiday_Flag', 'Competitor_Price', 'Month']
X = df[features]
y = df['Sales']

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# ------------------- TRAIN TEST SPLIT -------------------
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# ------------------- MODELS -------------------
models = {
    "Linear Regression": LinearRegression(),
    "Random Forest": RandomForestRegressor(n_estimators=100),
    "XGBoost": xgb.XGBRegressor(objective='reg:squarederror', n_estimators=100)
}

results = {}

st.subheader("🤖 Model Performance")

for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mape = mean_absolute_percentage_error(y_test, y_pred) * 100
    r2 = r2_score(y_test, y_pred)

    results[name] = {"RMSE": rmse, "MAPE": mape, "R2 Score": r2}

# Show results
results_df = pd.DataFrame(results).T
st.dataframe(results_df.style.highlight_max(color='#AED6F1', axis=0))

# Best model
best_model_name = results_df['R2 Score'].idxmax()
best_model = models[best_model_name]
joblib.dump(best_model, 'best_sales_forecasting_model.pkl')

st.success(f"🏆 Best Model: {best_model_name}")

# ------------------- PREDICTION UI -------------------
st.subheader("📈 Predict Your Future Sales")

st.markdown("<h4 style='color: #2E86C1;'>Enter Your Business Scenario:</h4>", unsafe_allow_html=True)

col1, col2 = st.columns(2)

with col1:
    price = st.number_input("Product Selling Price (₹)", min_value=0.0, value=120.0)
    competitor_price = st.number_input("Competitor Price (₹)", min_value=0.0, value=125.0)

with col2:
    promotion_text = st.selectbox("Running Promotion/Discount?", ["Yes", "No"])
    promotion = 1 if promotion_text == "Yes" else 0

    holiday_text = st.selectbox("Is It Holiday Season?", ["Yes", "No"])
    holiday_flag = 1 if holiday_text == "Yes" else 0

month = st.number_input("Month (1 = January, 12 = December)", min_value=1, max_value=12, value=5)

# Predict
user_input = scaler.transform([[price, promotion, holiday_flag, competitor_price, month]])
predicted_sales = best_model.predict(user_input)[0]

st.markdown(f"<h3 style='color: green;'>🔮 Predicted Sales: {predicted_sales:.2f} Units</h3>", unsafe_allow_html=True)

# ------------------- ACTUAL vs PREDICTED -------------------
st.subheader("📊 Actual vs Predicted Sales Chart")

best_y_pred = best_model.predict(X_test)
sns.set_theme(style="darkgrid")  # Apply seaborn theme
plt.figure(figsize=(10, 5))
plt.plot(y_test.values, label='Actual Sales', color='#2980B9', linewidth=2)
plt.plot(best_y_pred, label='Predicted Sales', color='#F39C12', linewidth=2, linestyle='--')
plt.fill_between(range(len(y_test.values)), y_test.values, best_y_pred, color='#85C1E9', alpha=0.3)
plt.legend()
plt.title(f"{best_model_name}: Actual vs Predicted Sales", fontsize=14)
st.pyplot(plt)


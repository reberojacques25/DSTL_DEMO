import streamlit as st
import pandas as pd
import numpy as np
import altair as alt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import json
import requests

# -------------------------------------------------------------
#                 🔐 GOOGLE GEMINI API KEY SETUP
# -------------------------------------------------------------
# Replace "YOUR_GEMINI_API_KEY" with your actual Google API key
GEMINI_API_KEY = "AIzaSyDzuPPWuXy_vjK-kOa_IHZH5HkVVyFhdNM"

def call_gemini(prompt):
    """Call Google Gemini API and return a response."""
    try:
        url = "https://generativelanguage.googleapis.com/v1beta/models/gemini-pro:generateContent?key=" + GEMINI_API_KEY
        headers = {"Content-Type": "application/json"}
        data = {"contents":[{"parts":[{"text": prompt}]}]}
        response = requests.post(url, headers=headers, json=data)
        r = response.json()
        
        return r["candidates"][0]["content"]["parts"][0]["text"]
    except Exception as e:
        return "Gemini API Error: " + str(e)

# -------------------------------------------------------------
#                   🎨 STREAMLIT UI
# -------------------------------------------------------------
st.set_page_config(page_title="DataSpace AI Platform", layout="wide")

st.title("🚀 DataSpace AI Analytics Prototype")
st.write("A near-complete prototype simulating the DataSpace platform.")

# -------------------------------------------------------------
#                   🧑‍💼 MULTI-TENANT LOGIN
# -------------------------------------------------------------
st.sidebar.header("🔐 Tenant Login (Simulation)")
tenant = st.sidebar.selectbox(
    "Select a Tenant:",
    ["Company A", "Company B", "Company C"]
)
st.sidebar.success(f"Logged in as: **{tenant}**")

# -------------------------------------------------------------
#                   📤 DATA UPLOAD
# -------------------------------------------------------------
st.sidebar.header("📤 Upload Dataset")
file = st.sidebar.file_uploader("Upload CSV or Excel", type=["csv", "xlsx"])

if not file:
    st.warning("Upload a dataset to continue.")
    st.stop()

# Load data
df = pd.read_csv(file) if file.name.endswith(".csv") else pd.read_excel(file)
st.subheader("📌 Auto Detected Schema")
st.json({col: str(df[col].dtype) for col in df.columns})

st.subheader("📊 First Rows of the Dataset")
st.dataframe(df.head())

# -------------------------------------------------------------
#                   🧪 BASIC ANALYSIS
# -------------------------------------------------------------
st.subheader("📈 Automated Insights (Data Profiling)")
numerics = df.select_dtypes(include=np.number).columns.tolist()
cats = df.select_dtypes(include=["object"]).columns.tolist()
dates = df.select_dtypes(include=["datetime"]).columns.tolist()

st.write("### 🔢 **Numeric Columns**")
st.write(numerics)

st.write("### 🏷 **Categorical Columns**")
st.write(cats)

# Summary statistics
if len(numerics) > 0:
    st.subheader("📊 Summary Statistics")
    st.dataframe(df[numerics].describe())

# -------------------------------------------------------------
#                   📉 AUTO VISUALIZATIONS
# -------------------------------------------------------------
st.subheader("📈 Automated Visualizations")

if len(numerics) >= 1:
    numcol = st.selectbox("Select numeric column to visualize:", numerics)
    chart = alt.Chart(df).mark_line().encode(
        x=df.index,
        y=numcol
    ).properties(height=300)
    st.altair_chart(chart, use_container_width=True)

if len(cats) >= 1:
    catcol = st.selectbox("Select categorical column:", cats)
    cat_chart = alt.Chart(df).mark_bar().encode(
        x=catcol,
        y="count()"
    ).properties(height=300)
    st.altair_chart(cat_chart, use_container_width=True)

# -------------------------------------------------------------
#                       🤖 AI INSIGHTS
# -------------------------------------------------------------
st.subheader("🤖 AI Insight Generator (Google Gemini)")

if st.button("Generate AI Insights"):
    prompt = f"""
    You are a senior business analyst. Analyze this dataset and provide insights.
    Dataset columns: {list(df.columns)}
    Numeric columns: {numerics}
    Categorical columns: {cats}

    Provide:
    - Key patterns
    - Potential risks
    - Opportunities
    - Business recommendations
    """
    ai_insight = call_gemini(prompt)
    st.success("AI Insights Generated")
    st.write(ai_insight)

# -------------------------------------------------------------
#                📈 SIMPLE MACHINE LEARNING DEMO
# -------------------------------------------------------------
st.subheader("🧠 Predictive Model (Auto-Regression Demo)")

if len(numerics) >= 2:
    target = st.selectbox("Select target column:", numerics)
    features = [col for col in numerics if col != target]

    X = df[features].fillna(0)
    y = df[target].fillna(0)

    model = LinearRegression()
    model.fit(X, y)
    preds = model.predict(X)

    st.write("### 📌 Model Coefficients")
    for f, c in zip(features, model.coef_):
        st.write(f"- **{f}** → {c:.4f}")

    st.write("### 🔮 Predictions (first 10 rows)")
    st.write(preds[:10])
else:
    st.info("Not enough numeric columns to run ML prediction.")

# -------------------------------------------------------------
#                ✔ ANALYST APPROVAL WORKFLOW
# -------------------------------------------------------------
st.subheader("✔ Analyst Recommendation Approval")

if st.checkbox("Show AI Draft Recommendation"):
    st.info("Mock Draft Insight: Sales have increased in high-demand regions. Consider optimizing logistics.")

if st.button("Approve"):
    st.success("Recommendation Approved and Published (Mock).")

if st.button("Reject"):
    st.error("Recommendation Rejected (Mock).")

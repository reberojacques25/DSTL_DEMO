import streamlit as st
import pandas as pd
import numpy as np
import altair as alt
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error, accuracy_score, classification_report
import requests

# -------------------------------------------------------------
# 🔐 GOOGLE GEMINI API KEY SETUP
# -------------------------------------------------------------
GEMINI_API_KEY = None
try:
    GEMINI_API_KEY = st.secrets.get("GEMINI_API_KEY")
except Exception:
    GEMINI_API_KEY = None

def call_gemini(prompt):
    if not GEMINI_API_KEY:
        return "Gemini API key not configured."
    try:
        url = "https://generativelanguage.googleapis.com/v1beta/models/gemini-pro:generateContent?key=" + GEMINI_API_KEY
        headers = {"Content-Type": "application/json"}
        data = {"contents":[{"parts":[{"text": prompt}]}]}
        response = requests.post(url, headers=headers, json=data, timeout=30)
        r = response.json()
        return r["candidates"][0]["content"]["parts"][0]["text"]
    except Exception as e:
        return "Gemini API Error: " + str(e)

# -------------------------------------------------------------
# UI SETUP
# -------------------------------------------------------------
st.set_page_config(page_title="DataSpace AI Platform", layout="wide")
st.title("🚀 DataSpace AI Analytics Prototype")
st.write("A near-complete prototype simulating the DataSpace platform.")

# -------------------------------------------------------------
# MULTI-TENANT LOGIN
# -------------------------------------------------------------
st.sidebar.header("🔐 Tenant Login (Simulation)")
tenant = st.sidebar.selectbox("Select a Tenant:", ["Company A", "Company B", "Company C"])
st.sidebar.success(f"Logged in as: **{tenant}**")

# -------------------------------------------------------------
# DATA UPLOAD
# -------------------------------------------------------------
st.sidebar.header("📤 Upload Dataset")
file = st.sidebar.file_uploader("Upload CSV or Excel", type=["csv", "xlsx"])

if not file:
    st.warning("Upload a dataset to continue.")
    st.stop()

try:
    df = pd.read_csv(file) if file.name.endswith(".csv") else pd.read_excel(file)
except Exception as e:
    st.error(f"Failed to load file: {e}")
    st.stop()

df.columns = [str(c) for c in df.columns]
df_reset = df.reset_index().rename(columns={"index": "_row_index"})

st.subheader("📌 Auto Detected Schema")
st.json({col: str(df[col].dtype) for col in df.columns})

st.subheader("📊 First Rows of the Dataset")
st.dataframe(df.head())

# -------------------------------------------------------------
# DATA PROFILING
# -------------------------------------------------------------
st.subheader("📈 Automated Insights (Data Profiling)")

numerics = df.select_dtypes(include=[np.number]).columns.tolist()
cats = df.select_dtypes(include=["object", "category"]).columns.tolist()

st.write("### 🔢 Numeric Columns")
st.write(numerics)

st.write("### 🏷 Categorical Columns")
st.write(cats)

if len(numerics) > 0:
    st.subheader("📊 Summary Statistics")
    st.dataframe(df[numerics].describe())

# -------------------------------------------------------------
# VISUALIZATION FIXED FOR ALTair
# -------------------------------------------------------------
st.subheader("📈 Automated Visualizations")
chart_df = df_reset.copy()

if len(numerics) >= 1:
    numcol = st.selectbox("Select numeric column to visualize:", numerics, index=0)
    chart_df[numcol] = pd.to_numeric(chart_df[numcol], errors="coerce")

    try:
        chart = alt.Chart(chart_df).mark_line().encode(
            x=alt.X("_row_index:Q", title="Row Index"),
            y=alt.Y(f"{numcol}:Q", title=numcol),
            tooltip=[alt.Tooltip("_row_index:Q"), alt.Tooltip(f"{numcol}:Q")]
        ).properties(height=350)
        st.altair_chart(chart, use_container_width=True)
    except Exception as e:
        st.error("Chart error: " + str(e))

if len(cats) >= 1:
    catcol = st.selectbox("Select categorical column:", cats, index=0)

    try:
        cat_chart = alt.Chart(chart_df).mark_bar().encode(
            x=alt.X(f"{catcol}:N", sort='-y', title=catcol),
            y=alt.Y("count()", title="count"),
            tooltip=[alt.Tooltip(f"{catcol}:N"), alt.Tooltip("count():Q")]
        ).properties(height=350)
        st.altair_chart(cat_chart, use_container_width=True)
    except Exception as e:
        st.error("Chart error: " + str(e))

# -------------------------------------------------------------
# AI INSIGHT GENERATION
# -------------------------------------------------------------
st.subheader("🤖 AI Insight Generator (Google Gemini)")
if st.button("Generate AI Insights"):
    prompt = f"""
    You are a senior business analyst. Analyze this dataset and provide insights.
    Dataset columns: {list(df.columns)}
    Numeric columns: {numerics}
    Categorical columns: {cats}
    """
    ai_insight = call_gemini(prompt)
    st.success("AI Insights Generated")
    st.write(ai_insight)

# -------------------------------------------------------------
# MACHINE LEARNING SECTION
# -------------------------------------------------------------
st.subheader("🧠 Predictive Modeling (Linear / Decision Trees Demo)")

all_features = numerics + cats

if len(all_features) >= 1:
    target = st.selectbox("Select target column:", df.columns.tolist())

    X = df[all_features].copy()
    y_raw = df[target].copy()

    encoders = {}
    X_encoded = pd.DataFrame(index=X.index)

    for col in X.columns:
        if col in numerics:
            X_encoded[col] = pd.to_numeric(X[col], errors="coerce").fillna(0)
        else:
            le = LabelEncoder()
            X_encoded[col] = le.fit_transform(X[col].astype(str))
            encoders[col] = le

    is_num = pd.api.types.is_numeric_dtype(y_raw)

    if is_num:
        y = pd.to_numeric(y_raw, errors="coerce").fillna(0)

        model_lr = LinearRegression()
        model_lr.fit(X_encoded, y)
        preds_lr = model_lr.predict(X_encoded)

        st.write("### 📌 Linear Regression Results")
        for f, c in zip(X_encoded.columns, model_lr.coef_):
            st.write(f"- **{f}** → {c:.4f}")

        st.write(f"R²: {r2_score(y, preds_lr):.4f}")
        st.write(f"RMSE: {mean_squared_error(y, preds_lr, squared=False):.4f}")

        tree = DecisionTreeRegressor()
        tree.fit(X_encoded, y)
        preds_tree = tree.predict(X_encoded)

        st.write("### 📌 Decision Tree Regressor RMSE")
        st.write(mean_squared_error(y, preds_tree, squared=False))

    else:
        le_y = LabelEncoder()
        y = le_y.fit_transform(y_raw.astype(str))

        tree = DecisionTreeClassifier()
        tree.fit(X_encoded, y)
        preds = tree.predict(X_encoded)

        st.write("### 📌 Decision Tree Classifier Accuracy")
        st.write(accuracy_score(y, preds))

else:
    st.info("Upload dataset with numeric or categorical fields to model.")

# -------------------------------------------------------------
# ANALYST APPROVAL WITH EDITING OPTION
# -------------------------------------------------------------
st.subheader("✔ Analyst Recommendation Approval")

draft_text = "Mock Draft Insight: Sales have increased in high-demand regions. Consider optimizing logistics."

show_draft = st.checkbox("Show AI Draft Recommendation")
if show_draft:
    st.info(draft_text)

if "modifying" not in st.session_state:
    st.session_state.modifying = False

colA, colB = st.columns(2)
with colA:
    if st.button("Approve & Modify"):
        st.session_state.modifying = True
with colB:
    if st.button("Reject"):
        st.session_state.modifying = False
        st.error("Recommendation rejected.")

if st.session_state.modifying:
    st.write("✍️ Edit and finalize the approved recommendation below:")
    modified = st.text_area("Refine Recommendation", value=draft_text, height=150)

    if st.button("Submit Final Version"):
        st.success("Final recommendation submitted successfully!")
        st.write("### 📌 Final Published Version")
        st.write(modified)
        st.session_state.modifying = False

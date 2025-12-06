# app.py
"""
DataSpace AI Platform - Modular version
- Separates analytics, structured insight generation, and AI polishing
- Robust handling for numeric/categorical data
- Altair charts with proper schema typing (avoids SchemaValidationError)
- ML demos (linear regression + decision tree) using encoded categorical features
- Editable analyst approval workflow with versioning and download
- Optional Google Gemini polishing (graceful fallback if key missing)
"""

import streamlit as st
import pandas as pd
import numpy as np
import altair as alt
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error, accuracy_score, classification_report
import json
import requests
import io
from datetime import datetime

# ------------------------------
# Config / Gemini
# ------------------------------
st.set_page_config(page_title="DataSpace AI Platform (Modular)", layout="wide")
GEMINI_API_KEY = None
try:
    GEMINI_API_KEY = st.secrets.get("GEMINI_API_KEY")
except Exception:
    GEMINI_API_KEY = None

def call_gemini(prompt: str, timeout=25) -> str:
    """
    Call Google Gemini generative API if key present. Graceful fallback otherwise.
    """
    if not GEMINI_API_KEY:
        return "Gemini not configured — using local polishing."
    try:
        url = "https://generativelanguage.googleapis.com/v1beta/models/gemini-pro:generateContent?key=" + GEMINI_API_KEY
        headers = {"Content-Type": "application/json"}
        data = {"contents":[{"parts":[{"text": prompt}]}]}
        resp = requests.post(url, headers=headers, json=data, timeout=timeout)
        resp.raise_for_status()
        r = resp.json()
        return r["candidates"][0]["content"]["parts"][0]["text"]
    except Exception as e:
        return f"Gemini API Error: {e}"

# ------------------------------
# Utility Helpers
# ------------------------------
def safe_read_file(uploaded_file):
    try:
        if uploaded_file.name.lower().endswith(".csv"):
            return pd.read_csv(uploaded_file)
        else:
            return pd.read_excel(uploaded_file)
    except Exception as e:
        raise RuntimeError(f"Failed to read file: {e}")

def detect_columns(df: pd.DataFrame):
    numerics = df.select_dtypes(include=[np.number]).columns.tolist()
    cats = df.select_dtypes(include=["object", "category", "bool"]).columns.tolist()
    # detect datetime-like columns (attempt)
    dates = [c for c in df.columns if pd.api.types.is_datetime64_any_dtype(df[c])]
    for c in df.columns:
        if c not in dates:
            try:
                parsed = pd.to_datetime(df[c], errors="coerce")
                if parsed.notna().sum() / max(1, len(parsed)) > 0.5:
                    dates.append(c)
            except Exception:
                pass
    return numerics, cats, dates

# ------------------------------
# Core Analytics Functions
# ------------------------------
def compute_basic_summary(df: pd.DataFrame):
    """
    Returns a structured summary dict with statistics that the insight generator can use.
    """
    numerics, cats, dates = detect_columns(df)
    summary = {}
    summary['n_rows'] = int(df.shape[0])
    summary['n_columns'] = int(df.shape[1])
    summary['numeric_columns'] = numerics
    summary['categorical_columns'] = cats
    summary['datetime_columns'] = dates

    # numeric aggregates
    num_summary = {}
    for c in numerics:
        series = pd.to_numeric(df[c], errors="coerce")
        num_summary[c] = {
            "count": int(series.count()),
            "mean": float(series.mean()) if series.count() > 0 else None,
            "median": float(series.median()) if series.count() > 0 else None,
            "std": float(series.std()) if series.count() > 1 else None,
            "min": float(series.min()) if series.count() > 0 else None,
            "max": float(series.max()) if series.count() > 0 else None,
        }
    summary['numeric_summary'] = num_summary

    # top categories for categorical columns
    cat_summary = {}
    for c in cats:
        top = df[c].astype(str).value_counts(dropna=True).head(5).to_dict()
        cat_summary[c] = top
    summary['categorical_summary_top'] = cat_summary

    # simple trend check on first numeric column if datetime present
    trend = {}
    if dates and numerics:
        dt_col = dates[0]
        num_col = numerics[0]
        try:
            tmp = df[[dt_col, num_col]].copy()
            tmp[dt_col] = pd.to_datetime(tmp[dt_col], errors="coerce")
            tmp = tmp.dropna(subset=[dt_col])
            if tmp.shape[0] >= 3:
                tmp_sorted = tmp.sort_values(dt_col)
                first = tmp_sorted[num_col].iloc[0]
                last = tmp_sorted[num_col].iloc[-1]
                trend['first'] = float(first) if pd.notna(first) else None
                trend['last'] = float(last) if pd.notna(last) else None
                if pd.notna(first) and pd.notna(last) and first != 0:
                    trend['growth_rate'] = float((last - first) / (abs(first)))
                else:
                    trend['growth_rate'] = None
                trend['datetime_col'] = dt_col
                trend['numeric_col'] = num_col
        except Exception:
            trend = {}
    summary['trend'] = trend

    # anomaly flags: columns with high missingness or high cardinality
    issues = {}
    missingness = {c: float(df[c].isna().mean()) for c in df.columns}
    high_missing = {k:v for k,v in missingness.items() if v > 0.4}
    issues['high_missing'] = high_missing
    high_cardinality = {}
    for c in df.columns:
        try:
            unique_frac = df[c].nunique() / max(1, len(df))
            if unique_frac > 0.9 and df[c].nunique() > 20:
                high_cardinality[c] = float(unique_frac)
        except Exception:
            pass
    issues['high_cardinality'] = high_cardinality

    summary['issues'] = issues
    return summary

# ------------------------------
# Structured Insight Generator
# ------------------------------
def generate_structured_insight(summary: dict):
    """
    Use deterministic, rule-based logic to produce structured insights.
    AI should only polish these later.
    """
    insights = {
        "key_patterns": [],
        "drivers": [],
        "risks": [],
        "opportunities": [],
        "recommendations": []
    }

    # Basic dataset health
    if summary['n_rows'] == 0:
        insights['risks'].append("Dataset is empty — cannot produce meaningful analysis.")
        return insights

    # Missingness risk
    if summary['issues'].get('high_missing'):
        for col, frac in summary['issues']['high_missing'].items():
            insights['risks'].append(f"Column '{col}' has {frac*100:.1f}% missing values — may bias results.")

    # High cardinality warning
    for col in summary['issues'].get('high_cardinality', {}):
        insights['risks'].append(f"Column '{col}' has very high cardinality — encoding may inflate model complexity.")

    # Numeric patterns
    for c, stats in summary.get('numeric_summary', {}).items():
        if stats["count"] == 0:
            continue
        if stats["std"] is None:
            continue
        mean = stats["mean"]
        std = stats["std"]
        if std is not None and mean is not None and std > 0:
            cv = std / mean if mean != 0 else None
            if cv is not None and cv > 1.0:
                insights['key_patterns'].append(f"'{c}' shows high variability (CV={cv:.2f}).")
            elif cv is not None and cv < 0.1:
                insights['key_patterns'].append(f"'{c}' is very stable (low variability).")

    # Trend-based driver/opportunity
    trend = summary.get('trend', {})
    if trend:
        gr = trend.get('growth_rate')
        num_col = trend.get('numeric_col')
        dt_col = trend.get('datetime_col')
        if gr is not None:
            if gr > 0.05:
                insights['key_patterns'].append(f"{num_col} is increasing over time (growth_rate={gr:.2%}).")
                insights['opportunities'].append(f"Scale up operations related to '{num_col}'.")
            elif gr < -0.05:
                insights['risks'].append(f"{num_col} is declining over time (growth_rate={gr:.2%}).")
                insights['recommendations'].append("Investigate root causes for decline in the metric.")
            else:
                insights['key_patterns'].append(f"{num_col} shows flat/no significant trend over time.")

    # Categorical top contributors (drivers)
    for cat_col, top in summary.get('categorical_summary_top', {}).items():
        if top:
            top_item, top_count = next(iter(top.items()))
            insights['drivers'].append(f"Top {cat_col}: '{top_item}' (count={top_count}).")
            if top_count / max(1, summary['n_rows']) > 0.5:
                insights['recommendations'].append(f"Consider targeted strategies for majority '{top_item}' in column '{cat_col}'.")

    # Generic recommendations based on size
    if summary['n_rows'] < 50:
        insights['risks'].append("Small dataset size — results may not generalize.")
        insights['recommendations'].append("Collect more labeled data for robust modeling.")
    else:
        insights['recommendations'].append("Run targeted models on cleaned features (see ML section).")

    # Consolidate: if no insights found
    if not any(insights.values()):
        insights['key_patterns'].append("No strong automated patterns detected. Consider deeper analysis or domain-specific metrics.")

    return insights

# ------------------------------
# AI Polishing (natural language) - uses structured insight to create narrative
# ------------------------------
def polish_with_ai(structured_insight: dict, summary: dict, use_gemini: bool = True):
    """
    Convert structured insight into a human-friendly narrative.
    If Gemini not configured or returns an error, fall back to local formatting.
    """
    base_prompt = {
        "summary_snapshot": {
            "rows": summary.get('n_rows'),
            "cols": summary.get('n_columns'),
            "numeric_columns": summary.get('numeric_columns'),
            "categorical_columns": summary.get('categorical_columns')
        },
        "insights": structured_insight
    }
    # Build a concise prompt
    prompt = f"""You are a senior business analyst. Given the following structured analytics snapshot, produce:
1) A short executive summary (3-5 short bullet sentences)
2) 3 prioritized recommendations
3) A one-paragraph risk summary.

Analytics Snapshot (JSON):
{json.dumps(base_prompt, indent=2, default=str)}
"""
    if use_gemini and GEMINI_API_KEY:
        polished = call_gemini(prompt)
        if polished and not polished.startswith("Gemini API Error"):
            return polished
        # fall back to local if error
    # Local deterministic polishing
    lines = []
    si = structured_insight
    lines.append("Executive summary:")
    # key patterns first
    if si.get("key_patterns"):
        for kp in si["key_patterns"][:5]:
            lines.append(f"- {kp}")
    else:
        lines.append("- No major patterns automatically detected.")

    lines.append("\nTop recommendations:")
    recs = si.get("recommendations", [])[:5]
    if recs:
        for r in recs:
            lines.append(f"- {r}")
    else:
        lines.append("- No automatic recommendations generated.")

    lines.append("\nRisk summary:")
    if si.get("risks"):
        for r in si["risks"][:5]:
            lines.append(f"- {r}")
    else:
        lines.append("- No major risks automatically flagged.")

    lines.append("\nOpportunities & drivers:")
    for d in si.get("drivers", [])[:5]:
        lines.append(f"- {d}")
    for op in si.get("opportunities", [])[:5]:
        lines.append(f"- {op}")

    return "\n".join(lines)

# ------------------------------
# ML helpers (encode and run simple models)
# ------------------------------
def encode_features(df: pd.DataFrame, numeric_cols, categorical_cols):
    encoders = {}
    X_enc = pd.DataFrame(index=df.index)
    for c in numeric_cols:
        X_enc[c] = pd.to_numeric(df[c], errors="coerce").fillna(0)
    for c in categorical_cols:
        le = LabelEncoder()
        # cast to string to avoid NaN issues
        X_enc[c] = le.fit_transform(df[c].astype(str))
        encoders[c] = le
    return X_enc, encoders

# ------------------------------
# Streamlit UI
# ------------------------------
st.title("🚀 DataSpace AI Analytics (Structured Pipeline)")

st.sidebar.header("🔐 Tenant & Upload")
tenant = st.sidebar.selectbox("Select a Tenant (simulation):", ["Company A", "Company B", "Company C"])
st.sidebar.write(f"Active tenant: **{tenant}**")

uploaded_file = st.sidebar.file_uploader("Upload CSV / XLSX dataset", type=["csv", "xlsx"])
if uploaded_file is None:
    st.info("Please upload a dataset to begin. Example: a shipments or sales dataset.")
    st.stop()

# Load dataset
try:
    df = safe_read_file(uploaded_file)
except Exception as e:
    st.error(str(e))
    st.stop()

# Normalize columns and create index for plotting
df.columns = [str(c) for c in df.columns]
df_reset = df.reset_index().rename(columns={"index": "_row_index"})

# Show schema and sample
st.subheader("📌 Dataset Snapshot")
st.write(f"Rows: {df.shape[0]}, Columns: {df.shape[1]}")
st.dataframe(df.head(8))
st.json({col: str(df[col].dtype) for col in df.columns})

# Compute analytics
with st.spinner("Computing analytics..."):
    summary = compute_basic_summary(df)
    structured_insight = generate_structured_insight(summary)

# Display structured analytics
st.subheader("📈 Structured Analytics Summary")
col1, col2 = st.columns([2,1])
with col1:
    st.write("**Top numeric summaries (sample)**")
    # show a compact table for numeric summary
    if summary['numeric_summary']:
        num_df = pd.DataFrame(summary['numeric_summary']).T
        st.dataframe(num_df.style.format(precision=3))
    else:
        st.info("No numeric columns detected.")

    st.write("**Top categorical values (sample)**")
    if summary['categorical_summary_top']:
        sample_cat = {k: list(v.items())[:3] for k,v in summary['categorical_summary_top'].items()}
        st.json(sample_cat)
    else:
        st.info("No categorical columns detected.")

with col2:
    st.write("**Detected issues**")
    st.json(summary.get('issues', {}))

# Visualizations (safe)
st.subheader("📊 Visualizations")
viz_cols = st.multiselect("Choose up to 2 columns to visualize (first numeric, second categorical recommended):", df.columns.tolist(), default=df.columns.tolist()[:2])
if viz_cols:
    # ensure row index present and numeric
    chart_df = df_reset.copy()
    chart_df["_row_index"] = pd.to_numeric(chart_df["_row_index"], errors="coerce")
    # if first selected is numeric, line chart
    if len(viz_cols) >= 1 and viz_cols[0] in summary['numeric_columns']:
        numcol = viz_cols[0]
        chart_df[numcol] = pd.to_numeric(chart_df[numcol], errors="coerce")
        try:
            line = alt.Chart(chart_df).mark_line().encode(
                x=alt.X("_row_index:Q", title="Row index"),
                y=alt.Y(f"{numcol}:Q", title=numcol),
                tooltip=[alt.Tooltip("_row_index:Q"), alt.Tooltip(f"{numcol}:Q")]
            ).properties(height=300)
            st.altair_chart(line, use_container_width=True)
        except Exception as e:
            st.error(f"Numeric chart error: {e}")

    # if second selected is categorical, bar count
    if len(viz_cols) >= 2 and viz_cols[1] in summary['categorical_columns']:
        catcol = viz_cols[1]
        try:
            bar = alt.Chart(chart_df).mark_bar().encode(
                x=alt.X(f"{catcol}:N", sort='-y', title=catcol),
                y=alt.Y("count()", title="count"),
                tooltip=[alt.Tooltip(f"{catcol}:N"), alt.Tooltip("count():Q")]
            ).properties(height=300)
            st.altair_chart(bar, use_container_width=True)
        except Exception as e:
            st.error(f"Categorical chart error: {e}")
else:
    st.info("Choose columns to visualize.")

# Show the structured insights (deterministic)
st.subheader("🧾 Structured Insights (rule-based)")
st.write("Key patterns:")
for kp in structured_insight.get("key_patterns", []):
    st.write(f"- {kp}")

st.write("Drivers:")
for d in structured_insight.get("drivers", []):
    st.write(f"- {d}")

st.write("Risks:")
for r in structured_insight.get("risks", []):
    st.write(f"- {r}")

st.write("Opportunities:")
for o in structured_insight.get("opportunities", []):
    st.write(f"- {o}")

st.write("Recommendations:")
for rec in structured_insight.get("recommendations", []):
    st.write(f"- {rec}")

# AI polishing (optional)
st.subheader("🤖 Polished Narrative (AI-enhanced)")
use_gemini = st.checkbox("Use Gemini for polishing (requires GEMINI_API_KEY in Streamlit secrets)", value=False)
if st.button("Generate Polished Narrative"):
    with st.spinner("Polishing narrative..."):
        polished = polish_with_ai(structured_insight, summary, use_gemini=use_gemini)
    st.text_area("Polished Insight (editable)", value=polished, height=220, key="polished_text_area")

# ------------------------------
# ML Demo (uses structured features)
# ------------------------------
st.subheader("🧠 Modeling Demo (automatic encoding)")

if st.checkbox("Show ML demo (encode categorical columns and run simple models)"):
    numerics = summary['numeric_columns']
    cats = summary['categorical_columns']
    all_feature_cols = st.multiselect("Select feature columns for modeling:", options=numerics+cats, default=numerics+cats[:2])

    if not all_feature_cols:
        st.info("Select at least one feature column.")
    else:
        # choose target
        target = st.selectbox("Select target column:", options=df.columns.tolist())
        X = df[all_feature_cols].copy()
        y_raw = df[target].copy()

        # Split numeric vs categorical
        sel_num = [c for c in all_feature_cols if c in numerics]
        sel_cat = [c for c in all_feature_cols if c in cats]

        X_enc, encoders = encode_features(X, sel_num, sel_cat)

        # Decide target type
        target_is_numeric = pd.api.types.is_numeric_dtype(y_raw)
        if target_is_numeric:
            y = pd.to_numeric(y_raw, errors="coerce").fillna(0)
            # regression baseline
            try:
                lr = LinearRegression()
                lr.fit(X_enc.fillna(0), y)
                preds = lr.predict(X_enc.fillna(0))
                st.write("LinearRegression R² (train):", float(r2_score(y, preds)))
            except Exception as e:
                st.error("Linear regression error: " + str(e))

            # decision tree regressor (train/test)
            try:
                X_train, X_test, y_train, y_test = train_test_split(X_enc.fillna(0), y, test_size=0.2, random_state=42)
                tree = DecisionTreeRegressor(random_state=42)
                tree.fit(X_train, y_train)
                pred_test = tree.predict(X_test)
                st.write("DecisionTreeRegressor R² (test):", float(r2_score(y_test, pred_test)))
                st.write("DecisionTreeRegressor RMSE (test):", float(mean_squared_error(y_test, pred_test, squared=False)))
            except Exception as e:
                st.error("Tree regressor error: " + str(e))

        else:
            # classification
            try:
                le_y = LabelEncoder()
                y = le_y.fit_transform(y_raw.astype(str))
                X_train, X_test, y_train, y_test = train_test_split(X_enc.fillna(0), y, test_size=0.2, random_state=42, stratify=y if len(np.unique(y))>1 else None)
                clf = DecisionTreeClassifier(random_state=42)
                clf.fit(X_train, y_train)
                preds = clf.predict(X_test)
                st.write("DecisionTreeClassifier accuracy (test):", float(accuracy_score(y_test, preds)))
                st.text(classification_report(y_test, preds, zero_division=0))
            except Exception as e:
                st.error("Classification error: " + str(e))

# ------------------------------
# Editable Approval Workflow & Publishing
# ------------------------------
st.subheader("✔ Analyst Approval & Publish")

# Prepare draft: prefer polished text if present else structured->local message
initial_draft = ""
if "polished_text_area" in st.session_state:
    initial_draft = st.session_state.get("polished_text_area", "")
if not initial_draft:
    # create a concise draft from structured_insight if AI not used
    initial_draft = "Executive summary:\n"
    for kp in structured_insight.get("key_patterns", [])[:3]:
        initial_draft += f"- {kp}\n"
    initial_draft += "\nRecommendations:\n"
    for r in structured_insight.get("recommendations", [])[:5]:
        initial_draft += f"- {r}\n"

if "approval_versions" not in st.session_state:
    st.session_state["approval_versions"] = []

show_draft_checkbox = st.checkbox("Show current draft recommendation", value=True)
if show_draft_checkbox:
    st.write("**Draft (editable below)**")
    # editable area
    draft_text = st.text_area("Draft recommendation (modify as needed)", value=initial_draft, height=200, key="draft_editor")
else:
    draft_text = initial_draft

cols = st.columns([1,1,1])
with cols[0]:
    if st.button("Approve & Edit (open editor)"):
        st.session_state["editing_approved"] = True
        st.success("Approved — you can now edit before final publish.")
with cols[1]:
    if st.button("Publish Final Version"):
        timestamp = datetime.utcnow().isoformat()
        version = {"published_at": timestamp, "tenant": tenant, "content": draft_text}
        st.session_state["approval_versions"].append(version)
        st.success("Final recommendation published.")
with cols[2]:
    if st.button("Reject"):
        st.error("Recommendation rejected (no publish).")

if st.session_state.get("editing_approved", False):
    st.info("Edit the approved draft below. When done, click 'Publish Final Version'.")
    edited = st.text_area("Edit approved draft", value=draft_text, height=220, key="approved_edit_area")
    # update the draft editor as well
    st.session_state["draft_editor"] = edited

# show history and allow download
if st.session_state.get("approval_versions"):
    st.subheader("📚 Publication History")
    for i, v in enumerate(reversed(st.session_state["approval_versions"])):
        st.write(f"**Version {len(st.session_state['approval_versions'])-i}** — published at {v['published_at']} for tenant {v['tenant']}")
        st.write(v['content'])
    # enable download of latest bundle
    if st.button("Download publication history (JSON)"):
        data = json.dumps(st.session_state["approval_versions"], indent=2)
        b = io.BytesIO(data.encode("utf-8"))
        st.download_button("Click to download", b, file_name="publication_history.json", mime="application/json")

# ------------------------------
# End of app
# ------------------------------
st.write("---")
st.write("DataSpace AI Platform — modular pipeline demo. For best results: provide domain-specific numeric and datetime columns (e.g., 'sales','date').")

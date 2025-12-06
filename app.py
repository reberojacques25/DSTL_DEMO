# app.py
"""
DataSpace AI Platform (Fixed)
- Fixes Streamlit session_state assignment error
- Avoids using ID-like, high-cardinality columns by default in ML (prevents perfect accuracy leak)
- Keeps modular analytics -> structured insight -> AI polishing pipeline
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

st.set_page_config(page_title="DataSpace AI Platform (Fixed)", layout="wide")

# ------------------------------
# Config / Gemini
# ------------------------------
GEMINI_API_KEY = None
try:
    GEMINI_API_KEY = st.secrets.get("GEMINI_API_KEY")
except Exception:
    GEMINI_API_KEY = None

def call_gemini(prompt: str, timeout=25) -> str:
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
# Helpers
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

def compute_basic_summary(df: pd.DataFrame):
    numerics, cats, dates = detect_columns(df)
    summary = {}
    summary['n_rows'] = int(df.shape[0])
    summary['n_columns'] = int(df.shape[1])
    summary['numeric_columns'] = numerics
    summary['categorical_columns'] = cats
    summary['datetime_columns'] = dates

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

    cat_summary = {}
    for c in cats:
        top = df[c].astype(str).value_counts(dropna=True).head(5).to_dict()
        cat_summary[c] = top
    summary['categorical_summary_top'] = cat_summary

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
    summary['issues'] = issues
    summary['issues']['high_cardinality'] = high_cardinality
    return summary

def generate_structured_insight(summary: dict):
    insights = {"key_patterns": [], "drivers": [], "risks": [], "opportunities": [], "recommendations": []}
    if summary['n_rows'] == 0:
        insights['risks'].append("Dataset is empty — cannot produce meaningful analysis.")
        return insights

    if summary['issues'].get('high_missing'):
        for col, frac in summary['issues']['high_missing'].items():
            insights['risks'].append(f"Column '{col}' has {frac*100:.1f}% missing values — may bias results.")

    for col in summary['issues'].get('high_cardinality', {}):
        insights['risks'].append(f"Column '{col}' has very high cardinality — encoding may inflate model complexity.")

    for c, stats in summary.get('numeric_summary', {}).items():
        if stats["count"] == 0 or stats["std"] is None:
            continue
        mean = stats["mean"]
        std = stats["std"]
        if std is not None and mean is not None and std > 0:
            cv = std / mean if mean != 0 else None
            if cv is not None and cv > 1.0:
                insights['key_patterns'].append(f"'{c}' shows high variability (CV={cv:.2f}).")
            elif cv is not None and cv < 0.1:
                insights['key_patterns'].append(f"'{c}' is very stable (low variability).")

    trend = summary.get('trend', {})
    if trend:
        gr = trend.get('growth_rate')
        num_col = trend.get('numeric_col')
        if gr is not None:
            if gr > 0.05:
                insights['key_patterns'].append(f"{num_col} is increasing over time (growth_rate={gr:.2%}).")
                insights['opportunities'].append(f"Scale up operations related to '{num_col}'.")
            elif gr < -0.05:
                insights['risks'].append(f"{num_col} is declining over time (growth_rate={gr:.2%}).")
                insights['recommendations'].append("Investigate root causes for decline in the metric.")
            else:
                insights['key_patterns'].append(f"{num_col} shows flat/no significant trend over time.")

    for cat_col, top in summary.get('categorical_summary_top', {}).items():
        if top:
            top_item, top_count = next(iter(top.items()))
            insights['drivers'].append(f"Top {cat_col}: '{top_item}' (count={top_count}).")
            if top_count / max(1, summary['n_rows']) > 0.5:
                insights['recommendations'].append(f"Consider targeted strategies for majority '{top_item}' in column '{cat_col}'.")

    if summary['n_rows'] < 50:
        insights['risks'].append("Small dataset size — results may not generalize.")
        insights['recommendations'].append("Collect more labeled data for robust modeling.")
    else:
        insights['recommendations'].append("Run targeted models on cleaned features (see ML section).")

    if not any(insights.values()):
        insights['key_patterns'].append("No strong automated patterns detected. Consider deeper analysis or domain-specific metrics.")
    return insights

def polish_with_ai(structured_insight: dict, summary: dict, use_gemini: bool = True):
    base_prompt = {
        "summary_snapshot": {
            "rows": summary.get('n_rows'),
            "cols": summary.get('n_columns'),
            "numeric_columns": summary.get('numeric_columns'),
            "categorical_columns": summary.get('categorical_columns')
        },
        "insights": structured_insight
    }
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
    lines = []
    si = structured_insight
    lines.append("Executive summary:")
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

def encode_features(df: pd.DataFrame, numeric_cols, categorical_cols, exclude_high_cardinality=True, high_card_cols=None):
    encoders = {}
    X_enc = pd.DataFrame(index=df.index)
    for c in numeric_cols:
        X_enc[c] = pd.to_numeric(df[c], errors="coerce").fillna(0)
    for c in categorical_cols:
        if exclude_high_cardinality and high_card_cols and c in high_card_cols:
            # skip high-cardinality categorical column unless user explicitly includes
            continue
        le = LabelEncoder()
        X_enc[c] = le.fit_transform(df[c].astype(str))
        encoders[c] = le
    return X_enc, encoders

# ------------------------------
# Streamlit UI
# ------------------------------
st.title("🚀 DataSpace AI Analytics (Fixed)")

st.sidebar.header("🔐 Tenant & Upload")
tenant = st.sidebar.selectbox("Select a Tenant (simulation):", ["Company A", "Company B", "Company C"])
st.sidebar.write(f"Active tenant: **{tenant}**")

uploaded_file = st.sidebar.file_uploader("Upload CSV / XLSX dataset", type=["csv", "xlsx"])
if uploaded_file is None:
    st.info("Please upload a dataset to begin.")
    st.stop()

try:
    df = safe_read_file(uploaded_file)
except Exception as e:
    st.error(str(e))
    st.stop()

df.columns = [str(c) for c in df.columns]
df_reset = df.reset_index().rename(columns={"index": "_row_index"})

st.subheader("📌 Dataset Snapshot")
st.write(f"Rows: {df.shape[0]}, Columns: {df.shape[1]}")
st.dataframe(df.head(8))
st.json({col: str(df[col].dtype) for col in df.columns})

with st.spinner("Computing analytics..."):
    summary = compute_basic_summary(df)
    structured_insight = generate_structured_insight(summary)

st.subheader("📈 Structured Analytics Summary")
col1, col2 = st.columns([2,1])
with col1:
    st.write("**Top numeric summaries (sample)**")
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

# Visualizations
st.subheader("📊 Visualizations")
viz_cols = st.multiselect("Choose up to 2 columns to visualize (first numeric, second categorical recommended):", df.columns.tolist(), default=df.columns.tolist()[:2])
if viz_cols:
    chart_df = df_reset.copy()
    chart_df["_row_index"] = pd.to_numeric(chart_df["_row_index"], errors="coerce")
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

# Show structured insights
st.subheader("🧾 Structured Insights (rule-based)")
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

# AI polishing
st.subheader("🤖 Polished Narrative (AI-enhanced)")
use_gemini = st.checkbox("Use Gemini for polishing (requires GEMINI_API_KEY in Streamlit secrets)", value=False)
if st.button("Generate Polished Narrative"):
    with st.spinner("Polishing narrative..."):
        polished = polish_with_ai(structured_insight, summary, use_gemini=use_gemini)
    # store polished in session for later retrieval
    st.session_state["polished_text"] = polished
    st.text_area("Polished Insight (editable)", value=polished, height=220, key="polished_text_area")

# ------------------------------
# ML Demo (avoid ID leakage)
# ------------------------------
st.subheader("🧠 Modeling Demo (automatic encoding)")

# identify high-cardinality columns (IDs)
high_card_cols = list(summary['issues'].get('high_cardinality', {}).keys())
if high_card_cols:
    st.info(f"High-cardinality columns detected (excluded by default in ML): {high_card_cols}")

include_high_card = st.checkbox("Include high-cardinality columns in modeling (not recommended)", value=False)

if st.checkbox("Show ML demo (encode categorical columns and run simple models)"):
    numerics = summary['numeric_columns']
    cats = summary['categorical_columns']
    # default feature set excludes high-card cols unless user opts in
    default_features = [c for c in numerics + cats if (include_high_card or c not in high_card_cols)]
    all_feature_cols = st.multiselect("Select feature columns for modeling:", options=numerics+cats, default=default_features)

    if not all_feature_cols:
        st.info("Select at least one feature column.")
    else:
        target = st.selectbox("Select target column:", options=df.columns.tolist())
        X = df[all_feature_cols].copy()
        y_raw = df[target].copy()

        sel_num = [c for c in all_feature_cols if c in numerics]
        sel_cat = [c for c in all_feature_cols if c in cats]

        X_enc, encoders = encode_features(df[all_feature_cols], sel_num, sel_cat, exclude_high_cardinality=not include_high_card, high_card_cols=high_card_cols)

        if X_enc.shape[1] == 0:
            st.error("No features available after excluding high-cardinality columns. Enable include-high-cardinality or pick other features.")
        else:
            target_is_numeric = pd.api.types.is_numeric_dtype(y_raw)
            if target_is_numeric:
                y = pd.to_numeric(y_raw, errors="coerce").fillna(0)
                try:
                    X_train, X_test, y_train, y_test = train_test_split(X_enc.fillna(0), y, test_size=0.2, random_state=42)
                    lr = LinearRegression()
                    lr.fit(X_train, y_train)
                    preds_test = lr.predict(X_test)
                    st.write("LinearRegression R² (test):", float(r2_score(y_test, preds_test)))
                except Exception as e:
                    st.error("Linear regression error: " + str(e))

                try:
                    tree = DecisionTreeRegressor(random_state=42)
                    tree.fit(X_train, y_train)
                    pred_test = tree.predict(X_test)
                    st.write("DecisionTreeRegressor R² (test):", float(r2_score(y_test, pred_test)))
                    st.write("DecisionTreeRegressor RMSE (test):", float(mean_squared_error(y_test, pred_test, squared=False)))
                except Exception as e:
                    st.error("Tree regressor error: " + str(e))
            else:
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
# Approval workflow (fixed session-state handling)
# ------------------------------
st.subheader("✔ Analyst Approval & Publish")

if "polished_text" in st.session_state and st.session_state["polished_text"]:
    initial_draft = st.session_state.get("polished_text", "")
else:
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
    # draft_editor widget stores draft in session_state automatically under key 'draft_editor'
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
        # prefer edited approved area if present, else draft_editor
        final_content = st.session_state.get("approved_edit_area") or st.session_state.get("draft_editor") or draft_text
        timestamp = datetime.utcnow().isoformat()
        version = {"published_at": timestamp, "tenant": tenant, "content": final_content}
        st.session_state["approval_versions"].append(version)
        st.success("Final recommendation published.")
with cols[2]:
    if st.button("Reject"):
        st.error("Recommendation rejected (no publish).")

if st.session_state.get("editing_approved", False):
    st.info("Edit the approved draft below. When done, click 'Publish Final Version'.")
    # approved editor uses its own key; do NOT assign to session_state manually
    approved_edited = st.text_area("Edit approved draft", value=st.session_state.get("draft_editor", initial_draft), height=220, key="approved_edit_area")

# Publication history and download
if st.session_state.get("approval_versions"):
    st.subheader("📚 Publication History")
    for i, v in enumerate(reversed(st.session_state["approval_versions"])):
        st.write(f"**Version {len(st.session_state['approval_versions'])-i}** — published at {v['published_at']} for tenant {v['tenant']}")
        st.write(v['content'])
    if st.button("Download publication history (JSON)"):
        data = json.dumps(st.session_state["approval_versions"], indent=2)
        b = io.BytesIO(data.encode("utf-8"))
        st.download_button("Click to download", b, file_name="publication_history.json", mime="application/json")

st.write("---")
st.write("Notes:")
st.write("- ID-like columns (high-cardinality) are excluded by default from ML to prevent leakage.")
st.write("- If you see perfect accuracy, check for identifiers included as features (e.g., Shipment_ID).")

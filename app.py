# app.py
"""
DataSpace Streamlit Demo (Filesystem-backed per-tenant storage)
- Tenant folders: data/<tenant_id>/
- Uploads saved to tenant folder
- Schema detection, structured insights, AI polishing (optional Gemini)
- Analyst approval workflow with persisted publication history per tenant
- Client Dashboard to view published recommendations
- Webhook simulator
"""

import streamlit as st
import pandas as pd
import numpy as np
import altair as alt
import os
import json
import io
import requests
from datetime import datetime
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error, accuracy_score, classification_report

# -------------------------
# Config & Constants
# -------------------------
st.set_page_config(page_title="DataSpace Demo (Tenant Folders)", layout="wide")
ROOT_DATA_DIR = "data"  # relative folder where tenant folders stored
os.makedirs(ROOT_DATA_DIR, exist_ok=True)

# Gemini API (optional)
GEMINI_API_KEY = None
try:
    GEMINI_API_KEY = st.secrets.get("GEMINI_API_KEY")
except Exception:
    GEMINI_API_KEY = None

# -------------------------
# Utilities: filesystem per-tenant
# -------------------------
def tenant_path(tenant_id: str) -> str:
    safe = str(tenant_id).replace(" ", "_")
    path = os.path.join(ROOT_DATA_DIR, safe)
    os.makedirs(path, exist_ok=True)
    return path

def save_uploaded_file(uploaded_file, tenant_id: str) -> str:
    """
    Save uploaded file into tenant folder with timestamped filename.
    Returns the saved filepath.
    """
    folder = tenant_path(tenant_id)
    ts = datetime.utcnow().strftime("%Y%m%dT%H%M%S")
    filename = f"{ts}__{uploaded_file.name}"
    target = os.path.join(folder, filename)
    # write bytes
    with open(target, "wb") as f:
        f.write(uploaded_file.getbuffer())
    return target

def list_tenant_files(tenant_id: str):
    folder = tenant_path(tenant_id)
    files = sorted(os.listdir(folder))
    return [os.path.join(folder, f) for f in files]

def read_dataset_from_file(filepath: str) -> pd.DataFrame:
    try:
        if filepath.lower().endswith(".csv") or filepath.lower().endswith(".txt"):
            return pd.read_csv(filepath)
        elif filepath.lower().endswith(".xlsx") or filepath.lower().endswith(".xls"):
            return pd.read_excel(filepath)
        elif filepath.lower().endswith(".json"):
            return pd.read_json(filepath)
        else:
            # try CSV fallback
            return pd.read_csv(filepath)
    except Exception as e:
        raise RuntimeError(f"Could not read dataset {filepath}: {e}")

def load_publications(tenant_id: str):
    pub_file = os.path.join(tenant_path(tenant_id), "publications.json")
    if os.path.exists(pub_file):
        try:
            with open(pub_file, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return []
    return []

def save_publication(tenant_id: str, publication: dict):
    pub_file = os.path.join(tenant_path(tenant_id), "publications.json")
    publications = load_publications(tenant_id)
    publications.append(publication)
    with open(pub_file, "w", encoding="utf-8") as f:
        json.dump(publications, f, indent=2)

# -------------------------
# Gemini call (optional)
# -------------------------
def call_gemini(prompt: str, timeout: int = 25) -> str:
    """Call Google Gemini if key present, otherwise return error string."""
    if not GEMINI_API_KEY:
        return "Gemini not configured — falling back to local polishing."
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

# -------------------------
# Detection / Analytics
# -------------------------
def detect_columns(df: pd.DataFrame):
    numerics = df.select_dtypes(include=[np.number]).columns.tolist()
    cats = df.select_dtypes(include=["object", "category", "bool"]).columns.tolist()
    dates = [c for c in df.columns if pd.api.types.is_datetime64_any_dtype(df[c])]
    # attempt detection for date-like strings
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
    summary = {
        "n_rows": int(df.shape[0]),
        "n_columns": int(df.shape[1]),
        "numeric_columns": numerics,
        "categorical_columns": cats,
        "datetime_columns": dates,
    }
    # numeric stats
    num_summary = {}
    for c in numerics:
        s = pd.to_numeric(df[c], errors="coerce")
        num_summary[c] = {
            "count": int(s.count()),
            "mean": float(s.mean()) if s.count()>0 else None,
            "median": float(s.median()) if s.count()>0 else None,
            "std": float(s.std()) if s.count()>1 else None,
            "min": float(s.min()) if s.count()>0 else None,
            "max": float(s.max()) if s.count()>0 else None
        }
    summary["numeric_summary"] = num_summary
    # categorical top values
    cat_summary = {}
    for c in cats:
        try:
            v = df[c].astype(str).value_counts(dropna=True).head(5).to_dict()
            cat_summary[c] = v
        except Exception:
            cat_summary[c] = {}
    summary["categorical_summary_top"] = cat_summary
    # issues
    missingness = {c: float(df[c].isna().mean()) for c in df.columns}
    high_missing = {k:v for k,v in missingness.items() if v > 0.4}
    high_card = {}
    for c in df.columns:
        try:
            unique_frac = df[c].nunique() / max(1, len(df))
            if unique_frac > 0.9 and df[c].nunique() > 20:
                high_card[c] = float(unique_frac)
        except Exception:
            pass
    summary["issues"] = {"high_missing": high_missing, "high_cardinality": high_card}
    # simple trend if datetime+numeric present
    trend = {}
    if summary["datetime_columns"] and summary["numeric_columns"]:
        dt = summary["datetime_columns"][0]
        num = summary["numeric_columns"][0]
        try:
            tmp = df[[dt,num]].copy()
            tmp[dt] = pd.to_datetime(tmp[dt], errors="coerce")
            tmp = tmp.dropna(subset=[dt])
            if tmp.shape[0] >= 3:
                tmp = tmp.sort_values(dt)
                first = tmp[num].iloc[0]
                last = tmp[num].iloc[-1]
                trend["numeric_col"] = num
                trend["datetime_col"] = dt
                trend["first"] = float(first) if pd.notna(first) else None
                trend["last"] = float(last) if pd.notna(last) else None
                try:
                    trend["growth_rate"] = float((last-first) / (abs(first))) if first != 0 else None
                except Exception:
                    trend["growth_rate"] = None
        except Exception:
            trend = {}
    summary["trend"] = trend
    return summary

def generate_structured_insight(summary: dict):
    insights = {"key_patterns": [], "drivers": [], "risks": [], "opportunities": [], "recommendations": []}
    if summary["n_rows"] == 0:
        insights["risks"].append("Empty dataset.")
        return insights
    # missingness
    for col, frac in summary["issues"].get("high_missing", {}).items():
        insights["risks"].append(f"Column '{col}' has {frac*100:.1f}% missing values.")
    for col in summary["issues"].get("high_cardinality", {}):
        insights["risks"].append(f"Column '{col}' has very high cardinality and may leak IDs.")
    # numeric variability
    for c, s in summary.get("numeric_summary", {}).items():
        if s["count"] == 0 or s["std"] is None or s["mean"] is None:
            continue
        cv = s["std"] / s["mean"] if s["mean"] != 0 else None
        if cv and cv > 1.0:
            insights["key_patterns"].append(f"'{c}' shows high variability (CV={cv:.2f}).")
        elif cv and cv < 0.1:
            insights["key_patterns"].append(f"'{c}' is very stable (low variability).")
    # trend
    tr = summary.get("trend", {})
    if tr:
        gr = tr.get("growth_rate")
        if gr is not None:
            if gr > 0.05:
                insights["key_patterns"].append(f"{tr['numeric_col']} is increasing over time (growth={gr:.2%}).")
                insights["opportunities"].append(f"Scale related operations for '{tr['numeric_col']}'.")
            elif gr < -0.05:
                insights["risks"].append(f"{tr['numeric_col']} is declining (growth={gr:.2%}).")
                insights["recommendations"].append("Investigate root causes.")
            else:
                insights["key_patterns"].append(f"{tr['numeric_col']} shows flat trend.")
    # categorical drivers
    for c, top in summary.get("categorical_summary_top", {}).items():
        if top:
            item, count = next(iter(top.items()))
            insights["drivers"].append(f"Top {c}: '{item}' (count={count}).")
            if count / max(1, summary["n_rows"]) > 0.5:
                insights["recommendations"].append(f"Target strategies for '{item}' in '{c}'.")
    # data size advice
    if summary["n_rows"] < 50:
        insights["risks"].append("Small dataset size — results may not generalize.")
        insights["recommendations"].append("Collect more data for robust models.")
    else:
        insights["recommendations"].append("Run targeted ML on cleaned features (see ML demo).")
    if not any(insights.values()):
        insights["key_patterns"].append("No automatic patterns detected.")
    return insights

# -------------------------
# Polishing (AI/local)
# -------------------------
def polish_with_ai(structured_insight: dict, summary: dict, use_gemini: bool=False) -> str:
    snapshot = {
        "rows": summary.get("n_rows"),
        "cols": summary.get("n_columns"),
        "numeric_columns": summary.get("numeric_columns"),
        "categorical_columns": summary.get("categorical_columns"),
        "insights": structured_insight
    }
    prompt = f"""You are a senior analyst. Convert the structured snapshot into:
1) Executive summary (3 bullets)
2) Top 3 recommendations
3) Short risk paragraph

Snapshot (JSON):
{json.dumps(snapshot, indent=2, default=str)}
"""
    if use_gemini and GEMINI_API_KEY:
        out = call_gemini(prompt)
        if out and not out.startswith("Gemini API Error"):
            return out
    # local formatting fallback
    lines = []
    lines.append("Executive summary:")
    for kp in structured_insight.get("key_patterns", [])[:3]:
        lines.append(f"- {kp}")
    if not structured_insight.get("key_patterns"):
        lines.append("- No automatic patterns found.")
    lines.append("\nRecommendations:")
    for r in structured_insight.get("recommendations", [])[:5]:
        lines.append(f"- {r}")
    lines.append("\nRisks:")
    for r in structured_insight.get("risks", [])[:5]:
        lines.append(f"- {r}")
    lines.append("\nDrivers & Opportunities:")
    for d in structured_insight.get("drivers", [])[:3]:
        lines.append(f"- {d}")
    for o in structured_insight.get("opportunities", [])[:3]:
        lines.append(f"- {o}")
    return "\n".join(lines)

# -------------------------
# ML encode + demo
# -------------------------
def encode_features(df: pd.DataFrame, numeric_cols, categorical_cols, exclude_high_cardinality=True, high_card_cols=None):
    encoders = {}
    X = pd.DataFrame(index=df.index)
    for c in numeric_cols:
        X[c] = pd.to_numeric(df[c], errors="coerce").fillna(0)
    for c in categorical_cols:
        if exclude_high_cardinality and high_card_cols and c in high_card_cols:
            continue
        le = LabelEncoder()
        X[c] = le.fit_transform(df[c].astype(str))
        encoders[c] = le
    return X, encoders

# -------------------------
# Streamlit UI
# -------------------------
st.title("🚀 DataSpace Demo — Tenant Folders (Persistent)")

# Sidebar: tenant selection and onboarding simulation
st.sidebar.header("Tenant")
tenant_choice = st.sidebar.selectbox("Select tenant (simulated):", ["Tenant_A", "Tenant_B", "Tenant_C", "Create New..."])

if tenant_choice == "Create New...":
    new_tenant = st.sidebar.text_input("Enter new tenant id (no spaces):", value="")
    if new_tenant:
        tenant_id = new_tenant.strip()
    else:
        st.sidebar.warning("Provide tenant id to create new tenant.")
        st.stop()
else:
    tenant_id = tenant_choice

# initialize tenant folder
t_folder = tenant_path(tenant_id)

# Sidebar: upload dataset
st.sidebar.header("Ingest")
uploaded = st.sidebar.file_uploader("Upload CSV / XLSX / JSON", type=["csv", "xlsx", "xls", "json"])
if uploaded:
    saved = save_uploaded_file(uploaded, tenant_id)
    st.sidebar.success(f"Saved: {os.path.basename(saved)}")

# show existing tenant files
st.sidebar.write("Tenant files:")
tenant_files = list_tenant_files(tenant_id)
if tenant_files:
    for fpath in reversed(tenant_files[-10:]):
        st.sidebar.write(os.path.basename(fpath))
else:
    st.sidebar.write("No files yet.")

# Top-level tabs: Analyst Workspace, Client Dashboard, Admin
tabs = st.tabs(["Analyst Workspace", "Client Dashboard", "Admin / Utilities"])
with tabs[0]:
    st.header(f"Analyst Workspace — {tenant_id}")

    # choose dataset file to work with
    if not tenant_files:
        st.info("Upload a dataset in the sidebar to begin.")
        st.stop()

    dataset_file = st.selectbox("Choose dataset file to analyze:", options=tenant_files, format_func=lambda p: os.path.basename(p))
    # load dataset with safety
    try:
        df = read_dataset_from_file(dataset_file)
    except Exception as e:
        st.error(str(e))
        st.stop()

    # normalize columns names
    df.columns = [str(c) for c in df.columns]
    df_reset = df.reset_index().rename(columns={"index": "_row_index"})

    st.subheader("Dataset Snapshot")
    st.write(f"Rows: {df.shape[0]}, Columns: {df.shape[1]}")
    st.dataframe(df.head(10))

    # schema detection + structured analytics
    with st.spinner("Detecting schema and computing analytics..."):
        summary = compute_basic_summary(df)
        structured_insight = generate_structured_insight(summary)

    st.subheader("Detected Schema")
    st.json({col: str(df[col].dtype) for col in df.columns})

    st.subheader("Structured Analytics Summary")
    col1, col2 = st.columns([2,1])
    with col1:
        st.write("Numeric summary (sample)")
        if summary["numeric_summary"]:
            num_df = pd.DataFrame(summary["numeric_summary"]).T
            st.dataframe(num_df)
        else:
            st.info("No numeric columns detected.")
        st.write("Top categorical values (sample)")
        st.json({k: list(v.items())[:3] for k,v in summary["categorical_summary_top"].items()})
    with col2:
        st.write("Detected issues")
        st.json(summary["issues"])

    # Visualizations (safe)
    st.subheader("Auto Visualizations")
    viz_cols = st.multiselect("Choose up to 2 columns (first numeric recommended):", options=list(df.columns), default=list(df.columns)[:2])
    if viz_cols:
        chart_df = df_reset.copy()
        chart_df["_row_index"] = pd.to_numeric(chart_df["_row_index"], errors="coerce")
        if viz_cols[0] in summary["numeric_columns"]:
            chart_df[viz_cols[0]] = pd.to_numeric(chart_df[viz_cols[0]], errors="coerce")
            try:
                line = alt.Chart(chart_df).mark_line().encode(
                    x=alt.X("_row_index:Q", title="Row"),
                    y=alt.Y(f"{viz_cols[0]}:Q", title=viz_cols[0]),
                    tooltip=[alt.Tooltip("_row_index:Q"), alt.Tooltip(f"{viz_cols[0]}:Q")]
                ).properties(height=300)
                st.altair_chart(line, use_container_width=True)
            except Exception as e:
                st.error("Numeric chart error: " + str(e))
        if len(viz_cols) > 1 and viz_cols[1] in summary["categorical_columns"]:
            try:
                bar = alt.Chart(chart_df).mark_bar().encode(
                    x=alt.X(f"{viz_cols[1]}:N", sort='-y', title=viz_cols[1]),
                    y=alt.Y("count()", title="count"),
                    tooltip=[alt.Tooltip(f"{viz_cols[1]}:N"), alt.Tooltip("count():Q")]
                ).properties(height=300)
                st.altair_chart(bar, use_container_width=True)
            except Exception as e:
                st.error("Categorical chart error: " + str(e))

    # show structured insights
    st.subheader("Structured Insights (rule-based)")
    for kp in structured_insight.get("key_patterns", []):
        st.write(f"- {kp}")
    if structured_insight.get("drivers"):
        st.write("Drivers:")
        for d in structured_insight.get("drivers", []):
            st.write(f"- {d}")
    if structured_insight.get("risks"):
        st.write("Risks:")
        for r in structured_insight.get("risks", []):
            st.write(f"- {r}")
    if structured_insight.get("recommendations"):
        st.write("Recommendations:")
        for rec in structured_insight.get("recommendations", []):
            st.write(f"- {rec}")

    # Polished narrative
    st.subheader("AI Polishing")
    use_gemini = st.checkbox("Use Gemini for polishing (requires GEMINI_API_KEY in Streamlit secrets)", value=False)
    if st.button("Generate Polished Narrative"):
        with st.spinner("Polishing..."):
            polished = polish_with_ai(structured_insight, summary, use_gemini)
        # stash polished into session_state
        st.session_state[f"polished__{tenant_id}__{os.path.basename(dataset_file)}"] = polished
        st.success("Polished narrative stored for this session.")

    polished_key = f"polished__{tenant_id}__{os.path.basename(dataset_file)}"
    default_draft = st.session_state.get(polished_key) or ""
    if not default_draft:
        # build default draft if no polished present
        default_draft = "Executive summary:\n"
        for kp in structured_insight.get("key_patterns", [])[:3]:
            default_draft += f"- {kp}\n"
        default_draft += "\nRecommendations:\n"
        for r in structured_insight.get("recommendations", [])[:5]:
            default_draft += f"- {r}\n"

    st.subheader("Analyst Draft Recommendation")
    # editable draft widget (stores into session_state automatically)
    draft_editor_val = st.text_area("Draft recommendation (edit before approval):", value=default_draft, height=220, key=f"draft_editor__{tenant_id}")

    # Approval controls
    st.subheader("Approval Controls")
    colA, colB, colC = st.columns(3)
    with colA:
        if st.button("Approve & Open Edit", key=f"approve_edit__{tenant_id}"):
            st.session_state[f"editing_approved__{tenant_id}"] = True
            st.success("Approved for editing.")
    with colB:
        if st.button("Publish Final Version", key=f"publish__{tenant_id}"):
            # prefer approved edit area if present
            final_text = st.session_state.get(f"approved_edit_area__{tenant_id}") or st.session_state.get(f"draft_editor__{tenant_id}") or draft_editor_val
            publication = {
                "tenant_id": tenant_id,
                "dataset_file": os.path.basename(dataset_file),
                "published_at": datetime.utcnow().isoformat(),
                "content": final_text
            }
            save_publication(tenant_id, publication)
            st.success("Published and saved to tenant folder.")
    with colC:
        if st.button("Reject", key=f"reject__{tenant_id}"):
            st.error("Recommendation rejected.")

    if st.session_state.get(f"editing_approved__{tenant_id}", False):
        st.info("Edit the approved draft below; then click Publish Final Version to persist.")
        approved_edit = st.text_area("Edit approved draft", value=st.session_state.get(f"draft_editor__{tenant_id}", default_draft), height=220, key=f"approved_edit_area__{tenant_id}")

    # ML demo
    st.subheader("ML Demo (encode + run simple models)")
    st.write("High-cardinality ID-like columns are excluded by default to avoid leakage.")
    high_card = list(summary["issues"].get("high_cardinality", {}).keys())
    if high_card:
        st.warning(f"High-cardinality columns detected: {high_card}")
    include_high_card = st.checkbox("Include high-cardinality columns in modeling (not recommended)", value=False)
    if st.checkbox("Show ML demo (run simple models)"):
        numerics = summary["numeric_columns"]
        cats = summary["categorical_columns"]
        default_feats = [c for c in numerics+cats if (include_high_card or c not in high_card)]
        feat_cols = st.multiselect("Feature columns:", options=numerics+cats, default=default_feats)
        if not feat_cols:
            st.info("Select features to run models.")
        else:
            target_col = st.selectbox("Select target column:", options=df.columns.tolist())
            X_enc, encs = encode_features(df[feat_cols], [c for c in feat_cols if c in numerics], [c for c in feat_cols if c in cats], exclude_high_cardinality=not include_high_card, high_card_cols=high_card)
            if X_enc.shape[1] == 0:
                st.error("No features available after excluding high-cardinality columns.")
            else:
                if pd.api.types.is_numeric_dtype(df[target_col]):
                    y = pd.to_numeric(df[target_col], errors="coerce").fillna(0)
                    try:
                        X_train, X_test, y_train, y_test = train_test_split(X_enc.fillna(0), y, test_size=0.2, random_state=42)
                        lr = LinearRegression()
                        lr.fit(X_train, y_train)
                        preds = lr.predict(X_test)
                        st.write("LinearRegression R² (test):", float(r2_score(y_test, preds)))
                    except Exception as e:
                        st.error("Regression error: " + str(e))
                else:
                    try:
                        le_y = LabelEncoder()
                        y = le_y.fit_transform(df[target_col].astype(str))
                        X_train, X_test, y_train, y_test = train_test_split(X_enc.fillna(0), y, test_size=0.2, random_state=42, stratify=y if len(np.unique(y))>1 else None)
                        clf = DecisionTreeClassifier(random_state=42)
                        clf.fit(X_train, y_train)
                        preds = clf.predict(X_test)
                        st.write("DecisionTreeClassifier accuracy (test):", float(accuracy_score(y_test, preds)))
                        st.text(classification_report(y_test, preds, zero_division=0))
                    except Exception as e:
                        st.error("Classification error: " + str(e))

with tabs[1]:
    st.header(f"Client Dashboard — {tenant_id}")
    st.write("Published recommendations for this tenant (persisted in tenant folder).")
    pubs = load_publications(tenant_id)
    if pubs:
        for i, p in enumerate(reversed(pubs)):
            st.markdown(f"**Published #{len(pubs)-i} — {p['published_at']}**")
            st.write(f"Dataset file: {p.get('dataset_file')}")
            st.write(p.get("content"))
            # webhook quick-send
            st.write("---")
            with st.expander("Webhook / notification"):
                webhook_url = st.text_input(f"Webhook URL to send this recommendation (leave blank to simulate):", key=f"webhook_url_{i}").strip()
                col1, col2 = st.columns([1,1])
                with col1:
                    if st.button(f"Send to webhook (publish #{len(pubs)-i})", key=f"send_webhook_{i}"):
                        payload = {
                            "tenant_id": tenant_id,
                            "published_at": p["published_at"],
                            "content": p["content"]
                        }
                        if webhook_url:
                            try:
                                resp = requests.post(webhook_url, json=payload, timeout=10)
                                st.success(f"Webhook returned status {resp.status_code}")
                                st.write(resp.text[:1000])
                            except Exception as e:
                                st.error(f"Webhook error: {e}")
                        else:
                            st.info("Simulated send — no webhook provided. Payload:")
                            st.json(payload)
                with col2:
                    if st.button(f"Download recommendation (#{len(pubs)-i})", key=f"dl_pub_{i}"):
                        b = io.BytesIO(json.dumps(p, indent=2).encode("utf-8"))
                        st.download_button("Download JSON", b, file_name=f"recommendation_{i}.json", mime="application/json")

    else:
        st.info("No published recommendations yet for this tenant.")

with tabs[2]:
    st.header("Admin / Utilities")
    st.write("Tenant folders are stored at `./data/` relative to this app.")
    st.write("Tenants found:")
    tenants = sorted([d for d in os.listdir(ROOT_DATA_DIR) if os.path.isdir(os.path.join(ROOT_DATA_DIR, d))])
    st.write(tenants)
    st.write("Create a tenant by selecting 'Create New...' in the Tenant selector and providing an ID.")
    st.write("---")
    st.write("Quick actions")
    if st.button("Clear all demo data (delete data/ folder)"):
        # careful: require confirm
        if st.checkbox("I confirm I want to delete all demo tenant data"):
            import shutil
            try:
                shutil.rmtree(ROOT_DATA_DIR)
                os.makedirs(ROOT_DATA_DIR, exist_ok=True)
                st.success("Deleted and recreated data/ folder.")
            except Exception as e:
                st.error(f"Failed to clear data: {e}")
        else:
            st.info("Please check confirmation box to delete demo data.")
    st.write("---")
    st.write("Developer notes:")
    st.write("- For production, replace filesystem persistence with S3 / MinIO and a proper metadata DB.")
    st.write("- Implement RBAC, OAuth, and background workers for real-time pipelines.")

st.write("---")
st.write("Demo notes: This Streamlit demo simulates many DSTL platform flows. For a production-grade platform, you will need backend services, orchestration, and secure infrastructure.")

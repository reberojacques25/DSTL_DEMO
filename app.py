# app.py
"""
DataSpace Streamlit Demo — Tenant Folders with Roles, Agent Simulation, and Vector Store Mock
Features:
- Per-tenant filesystem persistence: data/<tenant>/
- Role simulation: Admin / Analyst / Client with UI gating
- Background-run simulation: auto-agent that runs inline if interval elapsed + Run Now button
- Embeddings pipeline (TF-IDF) and vector-store mock (cosine similarity search)
- Schema detection, structured insights, AI polishing (Gemini optional)
- Analyst approval workflow persisted to publications.json
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
from datetime import datetime, timezone
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error, accuracy_score, classification_report

# -------------------------
# Config
# -------------------------
st.set_page_config(page_title="DataSpace Demo (Roles + Agent + Vector Store)", layout="wide")
ROOT_DATA_DIR = "data"
os.makedirs(ROOT_DATA_DIR, exist_ok=True)

# Optional Gemini key in streamlit secrets
GEMINI_API_KEY = None
try:
    GEMINI_API_KEY = st.secrets.get("GEMINI_API_KEY")
except Exception:
    GEMINI_API_KEY = None

# -------------------------
# Filesystem helpers
# -------------------------
def tenant_path(tenant_id: str) -> str:
    safe = str(tenant_id).replace(" ", "_")
    path = os.path.join(ROOT_DATA_DIR, safe)
    os.makedirs(path, exist_ok=True)
    return path

def save_uploaded_file(uploaded_file, tenant_id: str) -> str:
    folder = tenant_path(tenant_id)
    ts = datetime.utcnow().strftime("%Y%m%dT%H%M%S")
    filename = f"{ts}__{uploaded_file.name}"
    target = os.path.join(folder, filename)
    with open(target, "wb") as f:
        f.write(uploaded_file.getbuffer())
    return target

def list_tenant_files(tenant_id: str):
    folder = tenant_path(tenant_id)
    files = sorted(os.listdir(folder))
    return [os.path.join(folder, f) for f in files if os.path.isfile(os.path.join(folder, f))]

def read_dataset_from_file(filepath: str) -> pd.DataFrame:
    try:
        if filepath.lower().endswith(".csv") or filepath.lower().endswith(".txt"):
            return pd.read_csv(filepath)
        elif filepath.lower().endswith(".xlsx") or filepath.lower().endswith(".xls"):
            return pd.read_excel(filepath)
        elif filepath.lower().endswith(".json"):
            return pd.read_json(filepath)
        else:
            return pd.read_csv(filepath)
    except Exception as e:
        raise RuntimeError(f"Could not read dataset {filepath}: {e}")

def load_json_file(path: str):
    if os.path.exists(path):
        try:
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return []
    return []

def save_json_file(path: str, data):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, default=str)

# Publications and drafts
def load_publications(tenant_id: str):
    path = os.path.join(tenant_path(tenant_id), "publications.json")
    return load_json_file(path)

def save_publication(tenant_id: str, publication: dict):
    path = os.path.join(tenant_path(tenant_id), "publications.json")
    pubs = load_json_file(path) or []
    pubs.append(publication)
    save_json_file(path, pubs)

def load_agent_runs(tenant_id: str):
    path = os.path.join(tenant_path(tenant_id), "agent_runs.json")
    return load_json_file(path)

def save_agent_run(tenant_id: str, run_record: dict):
    path = os.path.join(tenant_path(tenant_id), "agent_runs.json")
    runs = load_json_file(path) or []
    runs.append(run_record)
    save_json_file(path, runs)

def get_last_agent_run_time(tenant_id: str):
    runs = load_agent_runs(tenant_id)
    if not runs:
        return None
    # last item has last_run timestamp
    last = runs[-1].get("ran_at")
    if last:
        try:
            return datetime.fromisoformat(last)
        except Exception:
            return None
    return None

# Vector store paths
def vector_store_dir(tenant_id: str) -> str:
    d = os.path.join(tenant_path(tenant_id), "vector_store")
    os.makedirs(d, exist_ok=True)
    return d

def save_vector_store(tenant_id: str, embeddings: np.ndarray, ids: list, meta: list):
    d = vector_store_dir(tenant_id)
    np.save(os.path.join(d, "embeddings.npy"), embeddings)
    save_json_file(os.path.join(d, "ids.json"), ids)
    save_json_file(os.path.join(d, "meta.json"), meta)

def load_vector_store(tenant_id: str):
    d = vector_store_dir(tenant_id)
    emb_path = os.path.join(d, "embeddings.npy")
    ids_path = os.path.join(d, "ids.json")
    meta_path = os.path.join(d, "meta.json")
    if os.path.exists(emb_path) and os.path.exists(ids_path) and os.path.exists(meta_path):
        embs = np.load(emb_path)
        ids = load_json_file(ids_path)
        meta = load_json_file(meta_path)
        return embs, ids, meta
    return None, None, None

# -------------------------
# Agent / Gemini (optional)
# -------------------------
def call_gemini(prompt: str, timeout: int = 25) -> str:
    if not GEMINI_API_KEY:
        return "Gemini not configured — local polishing used."
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
# Detection, Analytics, Insight functions
# -------------------------
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
    summary = {
        "n_rows": int(df.shape[0]),
        "n_columns": int(df.shape[1]),
        "numeric_columns": numerics,
        "categorical_columns": cats,
        "datetime_columns": dates,
    }
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
    cat_summary = {}
    for c in cats:
        try:
            cat_summary[c] = df[c].astype(str).value_counts(dropna=True).head(5).to_dict()
        except Exception:
            cat_summary[c] = {}
    summary["categorical_summary_top"] = cat_summary
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
    # simple trend
    trend = {}
    if summary["datetime_columns"] and summary["numeric_columns"]:
        dt = summary["datetime_columns"][0]
        num = summary["numeric_columns"][0]
        try:
            tmp = df[[dt, num]].copy()
            tmp[dt] = pd.to_datetime(tmp[dt], errors="coerce")
            tmp = tmp.dropna(subset=[dt])
            if tmp.shape[0] >= 3:
                tmp = tmp.sort_values(dt)
                first = tmp[num].iloc[0]
                last = tmp[num].iloc[-1]
                trend = {"numeric_col": num, "datetime_col": dt, "first": float(first) if pd.notna(first) else None, "last": float(last) if pd.notna(last) else None}
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
    for col, frac in summary["issues"].get("high_missing", {}).items():
        insights["risks"].append(f"Column '{col}' has {frac*100:.1f}% missing values.")
    for col in summary["issues"].get("high_cardinality", {}):
        insights["risks"].append(f"Column '{col}' has very high cardinality and may leak IDs.")
    for c, s in summary.get("numeric_summary", {}).items():
        if s["count"] == 0 or s["std"] is None or s["mean"] is None:
            continue
        cv = s["std"] / s["mean"] if s["mean"] != 0 else None
        if cv and cv > 1.0:
            insights["key_patterns"].append(f"'{c}' shows high variability (CV={cv:.2f}).")
        elif cv and cv < 0.1:
            insights["key_patterns"].append(f"'{c}' is very stable (low variability).")
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
    for c, top in summary.get("categorical_summary_top", {}).items():
        if top:
            item, count = next(iter(top.items()))
            insights["drivers"].append(f"Top {c}: '{item}' (count={count}).")
            if count / max(1, summary["n_rows"]) > 0.5:
                insights["recommendations"].append(f"Target strategies for '{item}' in '{c}'.")
    if summary["n_rows"] < 50:
        insights["risks"].append("Small dataset size — results may not generalize.")
        insights["recommendations"].append("Collect more data for robust models.")
    else:
        insights["recommendations"].append("Run targeted ML on cleaned features (see ML demo).")
    if not any(insights.values()):
        insights["key_patterns"].append("No automatic patterns detected.")
    return insights

def polish_with_ai(structured_insight: dict, summary: dict, use_gemini: bool=False) -> str:
    snapshot = {"rows": summary.get("n_rows"), "cols": summary.get("n_columns"), "numeric_columns": summary.get("numeric_columns"), "categorical_columns": summary.get("categorical_columns"), "insights": structured_insight}
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
    # local fallback
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
# Vector store (TF-IDF) helpers
# -------------------------
def build_embeddings_for_df(df: pd.DataFrame, tenant_id: str, text_cols: list):
    """
    Build TF-IDF embeddings for each row by concatenating specified text columns.
    Saves embeddings (.npy), ids.json and meta.json in tenant vector_store folder.
    """
    # build documents
    docs = []
    meta = []
    ids = []
    for idx, row in df.iterrows():
        parts = []
        for c in text_cols:
            v = row.get(c, "")
            parts.append(str(v))
        doc = " ".join(parts)
        docs.append(doc)
        meta.append({ "row_index": int(idx), "preview": doc[:200] })
        ids.append(f"r{idx}")
    if len(docs) == 0:
        return False, "No docs to embed"
    vectorizer = TfidfVectorizer(max_features=512)
    embeddings = vectorizer.fit_transform(docs).toarray()  # shape (n_rows, n_features)
    save_vector_store(tenant_id, embeddings, ids, meta)
    # store vectorizer vocab for potential future use
    vs_dir = vector_store_dir(tenant_id)
    save_json_file(os.path.join(vs_dir, "vocab.json"), vectorizer.vocabulary_)
    return True, f"Built {len(docs)} embeddings with dim {embeddings.shape[1]}"

def vector_search(tenant_id: str, query: str, top_k: int = 5):
    embeddings, ids, meta = load_vector_store(tenant_id)
    if embeddings is None:
        return []
    # embed query using stored vocab (approximate using TF-IDF with vocab)
    vocab = load_json_file(os.path.join(vector_store_dir(tenant_id), "vocab.json"))
    if vocab:
        vectorizer = TfidfVectorizer(vocabulary=vocab)
    else:
        vectorizer = TfidfVectorizer()
    q_vec = vectorizer.fit_transform([query]).toarray()
    # pad/vocab mismatch handling: if q_vec dim different from embeddings, try simple fallback using cosine with same dims
    try:
        sims = cosine_similarity(q_vec, embeddings)[0]  # length n_rows
    except Exception:
        # reduce dims or compute naive dot if shapes differ
        sims = np.random.rand(embeddings.shape[0]) * 0.001  # fallback tiny random
    # rank
    top_idx = np.argsort(-sims)[:top_k]
    results = []
    for i in top_idx:
        results.append({"id": ids[i], "score": float(sims[i]), "meta": meta[i]})
    return results

# -------------------------
# ML helpers
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
# UI: Role selection + Tenant selection + Ingest
# -------------------------
st.title("🚀 DataSpace Demo — Roles, Agent Simulation & Vector Store")

# Sidebar: Role simulation
st.sidebar.header("User Role (simulation)")
role = st.sidebar.selectbox("Select role:", ["Admin", "Analyst", "Client"], index=1)
st.sidebar.write(f"Active role: **{role}**")

# Tenant selection / create
st.sidebar.header("Tenant")
tenants = sorted([d for d in os.listdir(ROOT_DATA_DIR) if os.path.isdir(os.path.join(ROOT_DATA_DIR, d))])
default_tenants = tenants if tenants else ["Tenant_A", "Tenant_B", "Tenant_C"]
tenant_choice = st.sidebar.selectbox("Choose tenant or create new:", default_tenants + ["Create New..."])
if tenant_choice == "Create New...":
    new_tenant = st.sidebar.text_input("New tenant id:", value="My_Tenant")
    if st.sidebar.button("Create tenant"):
        tenant_id = new_tenant.strip()
        tenant_path(tenant_id)
        st.sidebar.success(f"Created tenant {tenant_id}")
        st.experimental_rerun()
    else:
        st.stop()
else:
    tenant_id = tenant_choice

# make sure folder exists
t_folder = tenant_path(tenant_id)

# Ingest UI (any role can upload but Admin/Analyst usually do)
st.sidebar.header("Ingest / Files")
uploaded = st.sidebar.file_uploader(f"Upload dataset to {tenant_id} (CSV/XLSX/JSON)", type=["csv","xlsx","xls","json"])
if uploaded:
    saved = save_uploaded_file(uploaded, tenant_id)
    st.sidebar.success(f"Saved: {os.path.basename(saved)}")

# list tenant files
files = list_tenant_files(tenant_id)
st.sidebar.write("Existing files (latest 10):")
for f in reversed(files[-10:]):
    st.sidebar.write(os.path.basename(f))

# Agent background-run config (Admin only)
st.sidebar.header("AI Agent (automation)")
agent_enabled_key = f"agent_enabled__{tenant_id}"
agent_interval_key = f"agent_interval__{tenant_id}"  # seconds
agent_last_run_key = f"agent_last_run__{tenant_id}"

if role == "Admin":
    agent_enabled = st.sidebar.checkbox("Enable auto-agent (runs inline when interval elapsed)", value=st.session_state.get(agent_enabled_key, False))
    st.session_state[agent_enabled_key] = agent_enabled
    interval = st.sidebar.number_input("Agent run interval (seconds)", min_value=10, max_value=86400, value=int(st.session_state.get(agent_interval_key, 60)), step=10)
    st.session_state[agent_interval_key] = int(interval)
    st.sidebar.write("Agent history is persisted per tenant (agent_runs.json).")
else:
    agent_enabled = st.session_state.get(agent_enabled_key, False)
    interval = st.session_state.get(agent_interval_key, 60)

# -------------------------
# Tabs: Workspace, Client, Admin
# -------------------------
tabs = st.tabs(["Workspace", "Client Dashboard", "Admin / Tools"])
with tabs[0]:
    st.header(f"Workspace — Tenant: {tenant_id}  (Role: {role})")

    # dataset selection
    if not files:
        st.info("No files uploaded yet. Upload a dataset from the sidebar.")
        st.stop()

    data_file = st.selectbox("Select dataset file to analyze:", options=files, format_func=lambda p: os.path.basename(p))
    try:
        df = read_dataset_from_file(data_file)
    except Exception as e:
        st.error(str(e))
        st.stop()

    df.columns = [str(c) for c in df.columns]
    df_reset = df.reset_index().rename(columns={"index": "_row_index"})
    st.subheader("Dataset Snapshot")
    st.write(f"Rows: {df.shape[0]}  Columns: {df.shape[1]}")
    st.dataframe(df.head(8))

    # analytics
    with st.spinner("Computing schema + analytics..."):
        summary = compute_basic_summary(df)
        structured_insight = generate_structured_insight(summary)

    st.subheader("Detected Schema")
    st.json({col: str(df[col].dtype) for col in df.columns})

    st.subheader("Structured Analytics Summary")
    c1, c2 = st.columns([2,1])
    with c1:
        st.write("Numeric summary (sample)")
        if summary["numeric_summary"]:
            st.dataframe(pd.DataFrame(summary["numeric_summary"]).T)
        else:
            st.info("No numeric columns.")
        st.write("Categorical top values (sample)")
        st.json({k: list(v.items())[:3] for k,v in summary["categorical_summary_top"].items()})
    with c2:
        st.write("Issues")
        st.json(summary["issues"])

    # visualizations
    st.subheader("Visualizations")
    viz_cols = st.multiselect("Select up to 2 columns (first numeric recommended):", options=list(df.columns), default=list(df.columns)[:2], key=f"viz_cols__{tenant_id}")
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
    st.subheader("Structured Insights")
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

    # Polishing and draft
    st.subheader("AI Polishing & Draft Recommendation")
    use_gemini = st.checkbox("Use Gemini (if configured) to polish", value=False, key=f"use_gemini__{tenant_id}")
    if st.button("Generate Polished Draft", key=f"gen_polished__{tenant_id}"):
        with st.spinner("Generating polished draft..."):
            polished = polish_with_ai(structured_insight, summary, use_gemini=use_gemini)
        st.session_state[f"polished__{tenant_id}__{os.path.basename(data_file)}"] = polished
        st.success("Polished draft saved to session.")
    polished_key = f"polished__{tenant_id}__{os.path.basename(data_file)}"
    initial_draft = st.session_state.get(polished_key) or ""
    if not initial_draft:
        initial_draft = "Executive summary:\n"
        for kp in structured_insight.get("key_patterns", [])[:3]:
            initial_draft += f"- {kp}\n"
        initial_draft += "\nRecommendations:\n"
        for r in structured_insight.get("recommendations", [])[:5]:
            initial_draft += f"- {r}\n"

    draft_editor = st.text_area("Draft recommendation (editable)", value=initial_draft, height=220, key=f"draft_editor__{tenant_id}")

    # Approval gating based on role
    st.subheader("Approval Workflow")
    if role != "Analyst":
        st.info("Only an Analyst role can Approve. Use the sidebar to switch role.")
    # Approve & publish controls (Analyst only)
    col1, col2, col3 = st.columns(3)
    with col1:
        if role == "Analyst" and st.button("Approve & Edit", key=f"approve_edit__{tenant_id}"):
            st.session_state[f"editing_approved__{tenant_id}"] = True
            st.success("Approved — you can edit before final publishing.")
    with col2:
        if role == "Analyst" and st.button("Publish Final Version", key=f"publish__{tenant_id}"):
            final_content = st.session_state.get(f"approved_edit_area__{tenant_id}") or st.session_state.get(f"draft_editor__{tenant_id}") or draft_editor
            publication = {"tenant_id": tenant_id, "dataset_file": os.path.basename(data_file), "published_at": datetime.utcnow().isoformat(), "content": final_content}
            save_publication(tenant_id, publication)
            st.success("Published and saved.")
    with col3:
        if role == "Analyst" and st.button("Reject", key=f"reject__{tenant_id}"):
            st.error("Recommendation rejected.")

    if st.session_state.get(f"editing_approved__{tenant_id}", False) and role == "Analyst":
        st.info("Edit the approved draft; click Publish when ready.")
        approved_edit = st.text_area("Edit approved draft", value=st.session_state.get(f"draft_editor__{tenant_id}", initial_draft), height=220, key=f"approved_edit_area__{tenant_id}")

    # ML demo section (optional)
    st.subheader("ML Demo (encode + simple models)")
    high_card = list(summary["issues"].get("high_cardinality", {}).keys())
    if high_card:
        st.warning(f"High-cardinality columns detected: {high_card}")
    include_high_card = st.checkbox("Include high-cardinality columns in modeling (not recommended)", value=False, key=f"incl_high_card__{tenant_id}")
    if st.checkbox("Run ML demo", key=f"run_ml_demo__{tenant_id}"):
        numerics = summary["numeric_columns"]
        cats = summary["categorical_columns"]
        default_feats = [c for c in numerics+cats if (include_high_card or c not in high_card)]
        feat_cols = st.multiselect("Feature columns:", options=numerics+cats, default=default_feats, key=f"ml_feats__{tenant_id}")
        if feat_cols:
            target_col = st.selectbox("Target column:", options=df.columns.tolist(), key=f"ml_target__{tenant_id}")
            X_enc, encs = encode_features(df[feat_cols], [c for c in feat_cols if c in numerics], [c for c in feat_cols if c in cats], exclude_high_cardinality=not include_high_card, high_card_cols=high_card)
            if X_enc.shape[1] == 0:
                st.error("No features available after excluding high-card columns.")
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

    # Embeddings pipeline & vector store
    st.subheader("Embeddings & Vector Store (mock)")
    text_columns = st.multiselect("Select text/categorical columns to build embeddings from:", options=summary["categorical_columns"] + summary["datetime_columns"] + list(df.select_dtypes(include=["object"]).columns), default=summary["categorical_columns"][:2], key=f"text_cols__{tenant_id}")
    if st.button("Build embeddings for selected columns", key=f"build_emb__{tenant_id}"):
        with st.spinner("Building embeddings..."):
            ok, msg = build_embeddings_for_df(df, tenant_id, text_columns)
        if ok:
            st.success(msg)
        else:
            st.error(msg)

    if st.checkbox("Show vector store status", key=f"show_vs__{tenant_id}"):
        embs, ids, meta = load_vector_store(tenant_id)
        if embs is None:
            st.info("No vector store found for this tenant.")
        else:
            st.write(f"Embeddings shape: {embs.shape}")
            st.write(f"Rows indexed: {len(ids)}")
            st.write("Sample meta:")
            st.json(meta[:5])

    st.write("---")
    # Vector search UI
    st.subheader("Vector Search (semantic) — query the vector store")
    q = st.text_input("Enter query text to search:", key=f"vs_query__{tenant_id}")
    top_k = st.slider("Top K", min_value=1, max_value=20, value=5, key=f"vs_topk__{tenant_id}")
    if st.button("Search vector store", key=f"vs_search__{tenant_id}"):
        results = vector_search(tenant_id, q, top_k)
        if not results:
            st.info("No results — vector store empty or not built.")
        else:
            st.write("Top matches:")
            for r in results:
                st.write(f"- id: {r['id']}  score: {r['score']:.4f}")
                st.write(r['meta'])

    # Agent run controls (Run now button visible to Admin and Analyst)
    st.subheader("AI Agent — Run & Simulation")
    st.write("Agent inspects schema, generates a structured draft recommendation, polishes it, and saves a run record.")
    if role in ["Admin", "Analyst"]:
        if st.button("Run Agent Now (inspect current dataset)"):
            with st.spinner("Running agent..."):
                sr = compute_basic_summary(df)
                si = generate_structured_insight(sr)
                polished = polish_with_ai(si, sr, use_gemini=False)
                run_record = {"ran_at": datetime.utcnow().isoformat(), "dataset_file": os.path.basename(data_file), "structured_insight": si, "polished": polished}
                save_agent_run(tenant_id, run_record)
                # also persist as a draft publication with status 'draft'
                draft = {"tenant_id": tenant_id, "dataset_file": os.path.basename(data_file), "created_at": datetime.utcnow().isoformat(), "status": "draft", "content": polished}
                drafts_path = os.path.join(tenant_path(tenant_id), "drafts.json")
                drafts = load_json_file(drafts_path) or []
                drafts.append(draft)
                save_json_file(drafts_path, drafts)
                st.success("Agent run complete and draft saved.")
        # Show agent history
        runs = load_agent_runs(tenant_id)
        if runs:
            st.write("Recent agent runs:")
            for r in reversed(runs[-5:]):
                st.write(f"- {r.get('ran_at')} (dataset: {r.get('dataset_file')})")
    else:
        st.info("Switch to Admin or Analyst to run the agent.")

    # Background-run simulation: check last run time and auto-run if enabled and interval elapsed
    if agent_enabled:
        last_run_iso = load_agent_runs(tenant_id)[-1]["ran_at"] if load_agent_runs(tenant_id) else None
        last_run = None
        try:
            last_run = datetime.fromisoformat(last_run_iso) if last_run_iso else None
        except Exception:
            last_run = None
        now = datetime.utcnow()
        interval_sec = int(interval)
        do_run = False
        if last_run is None:
            do_run = True
        else:
            elapsed = (now - last_run).total_seconds()
            if elapsed >= interval_sec:
                do_run = True
        # We run agent inline (not background) to simulate scheduled automation
        if do_run:
            # Only run for Admin/Analyst context to avoid surprising Clients, but still persist runs
            with st.spinner("Auto-agent: triggering scheduled agent run..."):
                sr = compute_basic_summary(df)
                si = generate_structured_insight(sr)
                polished = polish_with_ai(si, sr, use_gemini=False)
                run_record = {"ran_at": datetime.utcnow().isoformat(), "dataset_file": os.path.basename(data_file), "structured_insight": si, "polished": polished, "auto": True}
                save_agent_run(tenant_id, run_record)
                # save draft as auto-generated draft
                drafts_path = os.path.join(tenant_path(tenant_id), "drafts.json")
                drafts = load_json_file(drafts_path) or []
                drafts.append({"tenant_id": tenant_id, "dataset_file": os.path.basename(data_file), "created_at": datetime.utcnow().isoformat(), "status": "draft", "content": polished, "auto": True})
                save_json_file(drafts_path, drafts)
            st.success("Auto-agent run completed (draft saved).")

with tabs[1]:
    st.header(f"Client Dashboard — {tenant_id}")
    st.write("Published recommendations (persisted).")
    pubs = load_publications(tenant_id)
    if pubs:
        for i, p in enumerate(reversed(pubs)):
            st.markdown(f"**Published #{len(pubs)-i} — {p.get('published_at')}**")
            st.write(f"Dataset file: {p.get('dataset_file')}")
            st.write(p.get("content"))
            with st.expander("Webhook / Download"):
                wurl = st.text_input(f"Webhook URL to send this recommendation (leave blank to simulate):", key=f"cli_webhook_{i}")
                col1, col2 = st.columns(2)
                with col1:
                    if st.button(f"Send to webhook (pub #{len(pubs)-i})", key=f"cli_send_{i}"):
                        payload = {"tenant_id": tenant_id, "published_at": p["published_at"], "content": p["content"]}
                        if wurl:
                            try:
                                resp = requests.post(wurl, json=payload, timeout=10)
                                st.success(f"Webhook returned {resp.status_code}")
                                st.write(resp.text[:1000])
                            except Exception as e:
                                st.error(f"Webhook error: {e}")
                        else:
                            st.info("Simulated send. Payload:")
                            st.json(payload)
                with col2:
                    if st.button(f"Download pub #{len(pubs)-i}", key=f"cli_dl_{i}"):
                        b = io.BytesIO(json.dumps(p, indent=2).encode("utf-8"))
                        st.download_button("Download JSON", b, file_name=f"pub_{i}.json", mime="application/json")
    else:
        st.info("No published recommendations yet.")

    # show drafts if client is allowed to view (usually not)
    if role == "Client":
        drafts_path = os.path.join(tenant_path(tenant_id), "drafts.json")
        drafts = load_json_file(drafts_path)
        if drafts:
            st.subheader("Drafts (client view)")
            for d in reversed(drafts[-5:]):
                st.write(d.get("created_at"))
                st.write(d.get("content"))

with tabs[2]:
    st.header("Admin / Tools")
    st.write("Tenant folder location:", tenant_path(tenant_id))
    st.write("Agent runs history:")
    runs = load_agent_runs(tenant_id)
    if runs:
        st.dataframe(pd.DataFrame(runs)[["ran_at", "dataset_file"]].tail(20))
    else:
        st.info("No agent runs yet.")
    st.write("---")
    st.write("Vector store files (if built):")
    vs_dir = vector_store_dir(tenant_id)
    st.write(os.path.exists(vs_dir) and os.listdir(vs_dir) or "No vector store files.")
    st.write("---")
    st.write("Quick admin actions:")
    if st.button("Clear tenant publications & drafts"):
        pub_file = os.path.join(tenant_path(tenant_id), "publications.json")
        drafts_file = os.path.join(tenant_path(tenant_id), "drafts.json")
        try:
            if os.path.exists(pub_file):
                os.remove(pub_file)
            if os.path.exists(drafts_file):
                os.remove(drafts_file)
            st.success("Cleared publications & drafts for tenant.")
        except Exception as e:
            st.error(f"Error clearing: {e}")
    if st.button("Force-run agent now (admin)"):
        df_files = list_tenant_files(tenant_id)
        if not df_files:
            st.error("No dataset to run agent on.")
        else:
            # run agent using latest file
            lf = df_files[-1]
            try:
                df_admin = read_dataset_from_file(lf)
                sr = compute_basic_summary(df_admin)
                si = generate_structured_insight(sr)
                polished = polish_with_ai(si, sr, use_gemini=False)
                run_record = {"ran_at": datetime.utcnow().isoformat(), "dataset_file": os.path.basename(lf), "structured_insight": si, "polished": polished, "forced_by_admin": True}
                save_agent_run(tenant_id, run_record)
                drafts_path = os.path.join(tenant_path(tenant_id), "drafts.json")
                drafts = load_json_file(drafts_path) or []
                drafts.append({"tenant_id": tenant_id, "dataset_file": os.path.basename(lf), "created_at": datetime.utcnow().isoformat(), "status": "draft", "content": polished, "forced_by_admin": True})
                save_json_file(drafts_path, drafts)
                st.success("Agent run forced and draft saved.")
            except Exception as e:
                st.error(f"Agent run failed: {e}")

st.write("---")
st.write("Demo notes: This is a Streamlit-level demo to simulate many DSTL platform flows (roles, agent automation, embeddings). For production, implement backend services, secure storage, RBAC, and background workers.")

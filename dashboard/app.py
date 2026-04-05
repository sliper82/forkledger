"""
ForkLedger Dashboard — Streamlit web UI
========================================
Visualize decision history, policies, win rates, and regret.

Install:
    pip install forkledger[dashboard]

Run:
    forkledger dashboard --store store.db --backend sqlite
    # or directly:
    streamlit run dashboard/app.py -- --store store.db --backend sqlite
"""

import sys
import argparse
import json
from pathlib import Path

try:
    import streamlit as st
    import pandas as pd
    import altair as alt
    _STREAMLIT_OK = True
except ImportError:
    _STREAMLIT_OK = False
    print("Install dashboard deps: pip install forkledger[dashboard]")
    sys.exit(1)

# Parse CLI args passed after `--`
parser = argparse.ArgumentParser()
parser.add_argument("--store", default=".forkledger/store.db")
parser.add_argument("--backend", default="sqlite")
args, _ = parser.parse_known_args()

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from forkledger import ForkLedgerEngine  # noqa: E402

@st.cache_resource
def get_engine():
    return ForkLedgerEngine(args.store, backend=args.backend)

engine = get_engine()

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="ForkLedger",
    page_icon="⑂",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
.metric-card { background: #0e1117; border: 1px solid #21283a;
               border-radius: 8px; padding: 1rem; }
</style>
""", unsafe_allow_html=True)

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("⑂ ForkLedger")
    st.caption("Branch-based decision memory")
    st.divider()

    page = st.radio("Navigate", [
        "📊 Overview",
        "🌿 Decisions",
        "🎯 Recommendations",
        "📋 Policies",
        "📈 Analytics",
        "🔎 Audit Trail",
    ])

    st.divider()
    tags_filter = st.multiselect("Filter by tags", options=[
        t for r in engine.load() for t in r.tags
    ])
    min_conf = st.slider("Min confidence", 0.0, 1.0, 0.0, 0.05)

load_kwargs = dict(tags=tags_filter or None, min_confidence=min_conf)
records = engine.load(**load_kwargs)

# ── Overview ─────────────────────────────────────────────────────────────────
if page == "📊 Overview":
    st.title("Overview")
    stats = engine.stats()

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Forks", stats.get("total_forks", len(records)))
    c2.metric("Avg Confidence", f"{stats.get('avg_confidence', 0):.2f}")
    wr = engine.win_rates(**load_kwargs)
    if wr:
        top = max(wr, key=lambda b: wr[b]["win_rate"])
        c3.metric("Top Branch", top)
        c4.metric("Win Rate", f"{wr[top]['win_rate']:.1%}")

    if records:
        st.subheader("Decisions over time")
        df = pd.DataFrame([{
            "date": r.created_at[:10],
            "outcome": r.realized_value,
            "branch": r.chosen_branch,
            "confidence": r.confidence,
        } for r in records])
        chart = alt.Chart(df).mark_circle(size=60).encode(
            x="date:T",
            y="outcome:Q",
            color="branch:N",
            size="confidence:Q",
            tooltip=["date", "branch", "outcome", "confidence"],
        ).interactive()
        st.altair_chart(chart, use_container_width=True)

# ── Decisions ─────────────────────────────────────────────────────────────────
elif page == "🌿 Decisions":
    st.title("Decision History")
    if not records:
        st.info("No decisions recorded yet.")
    else:
        df = pd.DataFrame([{
            "fork_id":      r.fork_id,
            "date":         r.created_at[:10],
            "trigger":      r.trigger[:60],
            "chosen":       r.chosen_branch,
            "outcome":      r.realized_value,
            "confidence":   r.confidence,
            "tags":         ", ".join(r.tags),
        } for r in records])
        st.dataframe(df, use_container_width=True)

        selected_id = st.selectbox("Inspect fork", [r.fork_id for r in records])
        record = engine.get(selected_id)
        if record:
            col1, col2 = st.columns(2)
            with col1:
                st.json({"pre_state": record.pre_state, "trigger": record.trigger})
            with col2:
                st.json({"regret_vector": record.regret_vector,
                         "chosen": record.chosen_branch,
                         "outcome": record.realized_value})

# ── Recommendations ───────────────────────────────────────────────────────────
elif page == "🎯 Recommendations":
    st.title("Get Recommendation")
    state_input = st.text_area("Current state (JSON)", value='{"task": "research", "signal": "mixed"}')
    top_k = st.slider("Top K", 1, 10, 5)

    if st.button("Get recommendations"):
        try:
            state = json.loads(state_input)
            recs = engine.recommend(state, top_k=top_k, **load_kwargs)
            if not recs:
                st.warning("No relevant historical decisions found.")
            else:
                df = pd.DataFrame([{
                    "branch": r["branch"],
                    "score":  r["score"],
                    "evidence_count": len(r["support"]),
                } for r in recs])
                st.bar_chart(df.set_index("branch")["score"])
                st.dataframe(df, use_container_width=True)
        except json.JSONDecodeError:
            st.error("Invalid JSON in state field.")

# ── Policies ──────────────────────────────────────────────────────────────────
elif page == "📋 Policies":
    st.title("Distilled Policies")
    min_support = st.slider("Min support", 1, 10, 2)
    threshold = st.slider("Fuzzy threshold", 0.0, 1.0, 0.6, 0.05)

    policies = engine.policies(min_support=min_support,
                               similarity_threshold=threshold, **load_kwargs)
    if not policies:
        st.info(f"Need {min_support}+ similar decisions per state pattern.")
    else:
        for p in policies:
            with st.expander(f"State: {p['state']} (support={p['support']})"):
                st.markdown(f"**Recommended branch:** `{p['recommended_branch']}`")
                st.markdown(f"**Policy confidence:** {p.get('policy_confidence', 'N/A')}")
                st.markdown(f"**Win rates:** {p.get('win_rate', {})}")
                df_r = pd.DataFrame([
                    {"branch": k, "avg_regret": v}
                    for k, v in p["average_regret"].items()
                ])
                st.bar_chart(df_r.set_index("branch"))

# ── Analytics ─────────────────────────────────────────────────────────────────
elif page == "📈 Analytics":
    st.title("Analytics")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Win Rates")
        wr = engine.win_rates(**load_kwargs)
        if wr:
            df_wr = pd.DataFrame([
                {"branch": b, "win_rate": d["win_rate"],
                 "wins": d["wins"], "appearances": d["appearances"]}
                for b, d in wr.items()
            ]).sort_values("win_rate", ascending=False)
            st.bar_chart(df_wr.set_index("branch")["win_rate"])
            st.dataframe(df_wr, use_container_width=True)

    with col2:
        st.subheader("Accumulated Regret (CFR-weighted)")
        ar = engine.accumulated_regret(**load_kwargs)
        if ar:
            df_ar = pd.DataFrame([
                {"branch": b, "accumulated_regret": v}
                for b, v in sorted(ar.items(), key=lambda x: x[1])
            ])
            st.bar_chart(df_ar.set_index("branch")["accumulated_regret"])

# ── Audit Trail ───────────────────────────────────────────────────────────────
elif page == "🔎 Audit Trail":
    st.title("Audit Trail")
    trail = engine.audit_trail(tags=tags_filter or None, limit=100)
    if not trail:
        st.info("No decisions recorded yet.")
    else:
        df = pd.DataFrame(trail)
        st.dataframe(df[[
            "created_at", "fork_id", "trigger", "chosen_branch",
            "realized_value", "effective_weight", "tags"
        ]], use_container_width=True)

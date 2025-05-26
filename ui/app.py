import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src', 'inference')))

from predict_downtime import predict_downtime
import streamlit as st
import pandas as pd
import joblib
from pathlib import Path
import altair as alt

# ─── Page Config ─────────────────────────────────────────────
st.set_page_config(page_title="Smart Yeast Control", layout="wide")

# ─── Load Recommender Model ─────────────────────────────────
@st.cache_resource
def load_recommender():
    models_dir = Path("models/recommender")
    scaler = joblib.load(models_dir / "scaler.joblib")
    model = joblib.load(models_dir / "recommender_model.joblib")
    sp_cols = joblib.load(models_dir / "sp_cols.joblib")
    return scaler, model, sp_cols

# ─── Prediction Function for Recommender ─────────────────────
def predict_sp(df):
    scaler, model, sp_cols = load_recommender()
    drop_cols = sp_cols + ['Quality', 'quality', 'Set Time', 'VYP batch', 'Part']
    X = df.drop(columns=[c for c in drop_cols if c in df.columns], errors='ignore')
    expected = list(scaler.feature_names_in_)
    X = X.reindex(columns=expected, fill_value=0)
    X_scaled = scaler.transform(X)
    preds = model.predict(X_scaled)
    return pd.DataFrame(preds, columns=sp_cols, index=df.index)

# ─── UI ───────────────────────────────────────────────────────
st.title("📊 Smart Yeast Control Dashboard")
st.caption("AI-driven set-point recommendations and anomaly detection during Vegemite production.")

st.sidebar.header("📂 Upload your feature files")
rec_file = st.sidebar.file_uploader("🧠 Recommender features CSV", type=["csv"])
dt_file  = st.sidebar.file_uploader("⚠️ Downtime features CSV",    type=["csv"])

# ─── Set-Point Recommendation ────────────────────────────────
st.markdown("## 🔧 Set-Point Recommendation")
if rec_file is not None:
    rec_df = pd.read_csv(rec_file)
    with st.spinner("Predicting set-points..."):
        rec_out = predict_sp(rec_df)
    st.dataframe(rec_out, use_container_width=True)

    st.markdown("### 📈 SP Trends")
    rec_plot = rec_out.reset_index().rename(columns={'index': 'Sample'})
    for col in rec_out.columns:
        chart = alt.Chart(rec_plot).mark_line(point=True).encode(
            x='Sample:Q',
            y=alt.Y(f'{col}:Q', title='Recommended Value'),
            tooltip=['Sample', f'{col}']
        ).properties(title=f'{col} over Samples', width=700, height=300).interactive()
        st.altair_chart(chart, use_container_width=True)

    st.download_button(
        "⬇️ Download Recommendations",
        rec_out.to_csv(index=False).encode('utf-8'),
        "recommendations.csv",
        "text/csv"
    )

# ─── Downtime Detection ───────────────────────────────────────
st.markdown("---")
st.markdown("## ⏱️ Downtime Detection")

if dt_file is not None:
    dt_df = pd.read_csv(dt_file)
    with st.spinner("Predicting downtime events..."):
        dt_out = predict_downtime(dt_df)

    st.markdown("### 📋 Predicted Downtime Events")

    # Highlight predicted values with colors
    def highlight_prediction(val):
        if val == 1:
            return 'background-color: #ffe0e0'  # Light red
        elif val == 0:
            return 'background-color: #e0ffe0'  # Light green
        return ''

    styled_df = dt_out.style.applymap(highlight_prediction, subset=["downtime_prediction"])
    st.dataframe(styled_df, use_container_width=True)

    # Visualize chart classification
    if "downtime_prediction" in dt_out.columns:
        st.markdown("### 📈 Downtime Prediction Summary")
        chart_data = dt_out["downtime_prediction"].value_counts().rename_axis("Predicted Class").reset_index(name="Count")
        chart = alt.Chart(chart_data).mark_bar().encode(
            x="Predicted Class:O",
            y="Count:Q",
            tooltip=["Predicted Class", "Count"]
        ).properties(title="Distribution of Predicted Downtime", width=500, height=300)
        st.altair_chart(chart, use_container_width=True)

    st.download_button(
        "⬇️ Download Downtime Predictions",
        dt_out.to_csv(index=False).encode('utf-8'),
        "downtime_predictions.csv",
        "text/csv"
    )
else:
    st.markdown("⚠️ Please upload a Downtime features CSV file to proceed.")

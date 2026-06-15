import streamlit as st
import pickle
import pandas as pd
import streamlit.components.v1 as components

st.set_page_config(page_title="EqualCare", page_icon="image_93ea2b12.png", layout="wide")

# ── Responsive CSS ─────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');
* { font-family:'Inter',sans-serif; box-sizing:border-box; }
.block-container { padding-top:0 !important; max-width:1100px; padding-left:1rem !important; padding-right:1rem !important; }
header, footer { display:none !important; }

/* ── Hero ── */
.ec-hero {
  background: linear-gradient(135deg,#1a5276 0%,#1e8449 100%);
  padding: 48px 40px 44px;
  text-align: center;
  border-radius: 0 0 24px 24px;
  margin-bottom: 28px;
}
.ec-hero h1 { color:white; font-size:2.8rem; font-weight:800; margin:0 0 10px; }
.ec-hero p  { color:rgba(255,255,255,0.88); font-size:1.05rem; margin:0 0 20px; }
.ec-badges  { display:flex; gap:12px; justify-content:center; flex-wrap:wrap; }
.ec-badge   {
  background:rgba(255,255,255,0.15);
  border:1px solid rgba(255,255,255,0.3);
  color:white; padding:8px 18px;
  border-radius:999px; font-size:0.82rem; font-weight:500;
}

/* ── Stat cards ── */
.ec-stats {
  display: grid;
  grid-template-columns: repeat(4, 1fr);
  gap: 14px;
  margin-bottom: 24px;
}
.ec-stat {
  background:white; border:1px solid #e8edf2; border-radius:16px;
  padding:24px 18px; text-align:center;
  box-shadow:0 2px 6px rgba(0,0,0,0.04);
}
.ec-stat-num { font-size:2.2rem; font-weight:800; color:#1a2e44; line-height:1; margin-bottom:6px; }
.ec-stat-lbl { font-size:0.82rem; color:#6b7a8d; font-weight:500; }

/* ── Search panel ── */
.ec-search {
  background:white; border:1px solid #e8edf2; border-radius:18px;
  padding:28px 30px 20px;
  box-shadow:0 2px 10px rgba(0,0,0,0.05); margin-bottom:24px;
}
.ec-search-title { font-size:1.1rem; font-weight:700; color:#1a2e44; margin-bottom:18px; }

/* ── TABLET (≤768px) ── */
@media (max-width: 768px) {
  .ec-hero { padding:32px 20px 28px; border-radius:0 0 16px 16px; margin-bottom:18px; }
  .ec-hero h1 { font-size:2rem; }
  .ec-hero p  { font-size:0.92rem; }
  .ec-badge   { font-size:0.73rem; padding:6px 12px; }

  .ec-stats { grid-template-columns: repeat(2, 1fr); gap:10px; margin-bottom:16px; }
  .ec-stat-num { font-size:1.7rem; }

  .ec-search { padding:20px 16px 14px; border-radius:14px; }
  .block-container { padding-left:0.5rem !important; padding-right:0.5rem !important; }
}

/* ── MOBILE (≤480px) ── */
@media (max-width: 480px) {
  .ec-hero { padding:24px 14px 22px; }
  .ec-hero h1 { font-size:1.6rem; }
  .ec-hero p  { font-size:0.85rem; margin-bottom:14px; }
  .ec-badge   { font-size:0.68rem; padding:5px 10px; }

  .ec-stats { grid-template-columns: repeat(2, 1fr); gap:8px; }
  .ec-stat  { padding:16px 10px; border-radius:12px; }
  .ec-stat-num { font-size:1.4rem; }
  .ec-stat-lbl { font-size:0.72rem; }

  .ec-search { padding:14px 12px 10px; border-radius:12px; }
  .ec-search-title { font-size:0.95rem; }
}
</style>
""", unsafe_allow_html=True)

# ── Load pickle files ──────────────────────────────────────────────
@st.cache_data
def load_pickles():
    try:
        model    = pickle.load(open('model.pkl',         'rb'))
        encoder  = pickle.load(open('encoder.pkl',       'rb'))
        df       = pickle.load(open('hospital_data.pkl', 'rb'))
        meta     = pickle.load(open('meta.pkl',          'rb'))
        return model, encoder, df, meta
    except FileNotFoundError as e:
        st.error(f"Missing file: {e}. Run the backend notebook first to generate pickle files.")
        st.stop()

model, encoder, df, meta = load_pickles()

states    = meta['states']
diseases  = meta['diseases']
accuracy  = meta['accuracy']
n_hosp    = meta['total_hospitals']
n_states  = meta['total_states']
n_dist    = meta['total_districts']

FEATURES  = ['Hospital_Rating', 'District_Avg_Rating', 'Hospital_Score', 'Affordability_Score']
SCORE_MAP = {'Highly Recommended':4, 'Recommended':3, 'Moderate':2, 'Not Recommended':1}

# ── Hero banner ────────────────────────────────────────────────────
st.markdown(f"""
<div class="ec-hero">
  <h1>EqualCare</h1>
  <p>Find the most <b style="color:white;">affordable</b> and
     <b style="color:white;">quality</b> hospital district for your health needs</p>
  <div class="ec-badges">
    <div class="ec-badge"> SDG 1 — No Poverty &nbsp;|&nbsp; PES University — Subhash L</div>
    <div class="ec-badge"> Model: GradientBoostingClassifier &nbsp;|&nbsp; Accuracy: {accuracy}%</div>
  </div>
</div>
""", unsafe_allow_html=True)

# ── Stat cards ─────────────────────────────────────────────────────
st.markdown(f"""
<div class="ec-stats">
  <div class="ec-stat">
    <div class="ec-stat-num">{n_hosp:,}</div>
    <div class="ec-stat-lbl">🏥 Total Hospitals</div>
  </div>
  <div class="ec-stat">
    <div class="ec-stat-num">{n_states}</div>
    <div class="ec-stat-lbl">🗺️ States Covered</div>
  </div>
  <div class="ec-stat">
    <div class="ec-stat-num">{n_dist}</div>
    <div class="ec-stat-lbl">📍 Districts</div>
  </div>
  <div class="ec-stat">
    <div class="ec-stat-num">{accuracy}%</div>
    <div class="ec-stat-lbl">🎯 Model Accuracy</div>
  </div>
</div>
""", unsafe_allow_html=True)

# ── Search panel ───────────────────────────────────────────────────
st.markdown("""
<div class="ec-search">
  <div class="ec-search-title">🔍 Find Your Best Hospital District</div>
</div>
""", unsafe_allow_html=True)

col1, col2, col3 = st.columns([2, 2, 1])
with col1:
    st.markdown("<p style='font-size:13px;font-weight:600;color:#1a2e44;margin-bottom:4px;'>📍 Select Your State</p>", unsafe_allow_html=True)
    state = st.selectbox("state", states, label_visibility="collapsed")
with col2:
    st.markdown("<p style='font-size:13px;font-weight:600;color:#1a2e44;margin-bottom:4px;'>🩺 Select Disease / Health Need</p>", unsafe_allow_html=True)
    disease = st.selectbox("disease", diseases, label_visibility="collapsed")
with col3:
    st.markdown("<p style='font-size:13px;color:transparent;margin-bottom:4px;background-color: lightgreen'>.</p>", unsafe_allow_html=True)
    clicked = st.button("🔍  Find Districts", use_container_width=True)

# ── Prediction logic ───────────────────────────────────────────────
def get_top5(state, disease):
    f = df[df['State'] == state].copy()
    f = f[f['Suitable_For_Disease'].str.contains(disease, case=False, na=False)]
    if f.empty:
        return None

    preds = model.predict(f[FEATURES])
    f['Pred_Label']  = encoder.inverse_transform(preds)
    f['Label_Score'] = f['Pred_Label'].map(SCORE_MAP)

    grp = f.groupby('District').agg(
        Avg_Rating      = ('Hospital_Rating',     'mean'),
        Avg_Dist_Rating = ('District_Avg_Rating', 'mean'),
        Avg_Score       = ('Hospital_Score',      'mean'),
        Avg_Afford      = ('Affordability_Score', 'mean'),
        Avg_Label_Score = ('Label_Score',         'mean'),
        Total_Hospitals = ('Hospital_Rating',     'count')
    ).reset_index()

    grp['Combined_Score'] = (
        grp['Avg_Rating'] + grp['Avg_Dist_Rating'] +
        grp['Avg_Score']  + grp['Avg_Afford'] +
        grp['Avg_Label_Score'] * 2
    ) / 6

    best = f.groupby('District')['Pred_Label'].agg(
        lambda x: x.value_counts().idxmax()
    ).reset_index().rename(columns={'Pred_Label':'Top_Label'})

    grp = grp.merge(best, on='District')
    return grp.sort_values('Combined_Score', ascending=False).head(5).reset_index(drop=True).round(2)

# ── Results ────────────────────────────────────────────────────────
if clicked:
    with st.spinner("Finding top districts..."):
        result = get_top5(state, disease)

    if result is None:
        st.warning(f"No hospitals found in **{state}** for **{disease}**.")
    else:
        medals      = ['1','2','3','4','5']
        badge_bg    = ['#fef3c7','#f1f5f9','#fce7f3','#ede9fe','#dcfce7']
        badge_color = ['#b45309','#475569','#be185d','#6d28d9','#15803d']
        pill_style  = {
            'Highly Recommended': 'background:#dcfce7;color:#15803d;',
            'Recommended':        'background:#dbeafe;color:#1d4ed8;',
            'Moderate':           'background:#fef3c7;color:#b45309;',
            'Not Recommended':    'background:#fee2e2;color:#b91c1c;'
        }

        rows_html = ""
        for i, row in result.iterrows():
            rc = row['Top_Label']
            ps = pill_style.get(rc, 'background:#dbeafe;color:#1d4ed8;')
            rows_html += f"""
            <div class="ec-card">
              <div class="ec-medal" style="background:{badge_bg[i]};color:{badge_color[i]};">{medals[i]}</div>
              <div class="ec-district">{row['District']}</div>
              <div class="ec-stats-row">
                <span>⭐ Rating <b>{row['Avg_Rating']}</b></span>
                <span>📊 Score <b>{row['Avg_Score']}</b></span>
                <span>💰 Afford. <b>{row['Avg_Afford']}</b></span>
                <span>🏥 <b>{int(row['Total_Hospitals'])}</b> hospitals</span>
              </div>
              <div class="ec-pill" style="{ps}">{rc}</div>
              <div class="ec-score">⚡ {row['Combined_Score']}</div>
            </div>"""

        full_html = f"""
        <!DOCTYPE html>
        <html>
        <head>
        <meta name="viewport" content="width=device-width, initial-scale=1">
        <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap" rel="stylesheet">
        <style>
          * {{ box-sizing:border-box; margin:0; padding:0; font-family:'Inter',sans-serif; }}
          body {{ background:transparent; }}

          .ec-results {{
            background:white; border:1px solid #e8edf2; border-radius:18px;
            padding:24px 26px; box-shadow:0 2px 10px rgba(0,0,0,0.05);
          }}
          .ec-results-title {{
            font-size:1.1rem; font-weight:700; color:#1a2e44; margin-bottom:16px;
          }}

          /* Desktop card: single row */
          .ec-card {{
            display:flex; align-items:center; gap:12px;
            padding:14px 16px; border-radius:12px;
            border:1px solid #eef1f5; margin-bottom:10px;
            background:white; flex-wrap:wrap;
          }}
          .ec-medal {{
            width:36px; height:36px; border-radius:50%;
            font-size:0.9rem; font-weight:700; flex-shrink:0;
            display:flex; align-items:center; justify-content:center;
          }}
          .ec-district {{
            font-weight:700; font-size:1rem; color:#1a2e44; flex:1; min-width:120px;
          }}
          .ec-stats-row {{
            display:flex; gap:14px; flex-wrap:wrap;
          }}
          .ec-stats-row span {{ font-size:0.76rem; color:#6b7a8d; white-space:nowrap; }}
          .ec-stats-row b   {{ color:#1a2e44; }}
          .ec-pill {{
            padding:4px 12px; border-radius:999px;
            font-size:0.76rem; font-weight:600; white-space:nowrap;
          }}
          .ec-score {{
            background:#1a5276; color:white; padding:4px 14px;
            border-radius:999px; font-size:0.79rem; font-weight:600;
            white-space:nowrap;
          }}

          /* Tablet */
          @media (max-width: 768px) {{
            .ec-results {{ padding:18px 16px; border-radius:14px; }}
            .ec-results-title {{ font-size:0.95rem; }}
            .ec-card {{ gap:10px; padding:12px 12px; }}
            .ec-district {{ font-size:0.9rem; min-width:100px; }}
            .ec-stats-row {{ gap:10px; }}
            .ec-stats-row span {{ font-size:0.72rem; }}
            .ec-pill  {{ font-size:0.7rem; padding:3px 10px; }}
            .ec-score {{ font-size:0.72rem; padding:3px 10px; }}
          }}

          /* Mobile — stack each card vertically */
          @media (max-width: 480px) {{
            .ec-results {{ padding:14px 12px; border-radius:12px; }}
            .ec-results-title {{ font-size:0.88rem; margin-bottom:12px; }}
            .ec-card {{
              flex-direction:column; align-items:flex-start;
              gap:8px; padding:12px 12px;
            }}
            .ec-medal {{ width:30px; height:30px; font-size:0.8rem; }}
            .ec-district {{ font-size:0.95rem; }}
            .ec-stats-row {{ gap:8px; }}
            .ec-stats-row span {{ font-size:0.74rem; }}
            .ec-pill  {{ font-size:0.72rem; }}
            .ec-score {{ font-size:0.74rem; align-self:flex-end; }}
          }}
        </style>
        </head>
        <body>
        <div class="ec-results">
          <div class="ec-results-title">🏆 Top Districts in <b>{state}</b> for <b>{disease}</b></div>
          {rows_html}
        </div>
        </body>
        </html>
        """
        components.html(full_html, height=len(result) * 100 + 120, scrolling=False)

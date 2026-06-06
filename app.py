import streamlit as st
import pickle
import pandas as pd

st.set_page_config(page_title="EqualCare", page_icon="🏥", layout="wide")

# ── Inject CSS ─────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');
* { font-family:'Inter',sans-serif; }
.block-container { padding-top:0 !important; max-width:1100px; }
header, footer { display:none !important; }
</style>
""", unsafe_allow_html=True)

# ── Load pickle files ──────────────────────────────────────────────
# Uses st.cache_data so files are loaded ONCE and reused — safe for deployment
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
<div style="background:linear-gradient(135deg,#1a5276 0%,#1e8449 100%);
            padding:48px 40px 44px;text-align:center;
            border-radius:0 0 24px 24px;margin-bottom:28px;">
  <h1 style="color:white;font-size:2.8rem;font-weight:800;margin:0 0 10px;">EqualCare</h1>
  <p style="color:rgba(255,255,255,0.88);font-size:1.05rem;margin:0 0 20px;">
    Find the most <b style="color:white;">affordable</b> and
    <b style="color:white;">quality</b> hospital district for your health needs
  </p>
  <div style="display:flex;gap:12px;justify-content:center;flex-wrap:wrap;">
    <div style="background:rgba(255,255,255,0.15);border:1px solid rgba(255,255,255,0.3);
                color:white;padding:8px 18px;border-radius:999px;font-size:0.82rem;font-weight:500;">
      🎯 SDG 1 — No Poverty &nbsp;|&nbsp; PES University — Section F
    </div>
    <div style="background:rgba(255,255,255,0.15);border:1px solid rgba(255,255,255,0.3);
                color:white;padding:8px 18px;border-radius:999px;font-size:0.82rem;font-weight:500;">
      🤖 Model: GradientBoostingClassifier &nbsp;|&nbsp; Accuracy: {accuracy}%
    </div>
  </div>
</div>
""", unsafe_allow_html=True)

# ── Stat cards ─────────────────────────────────────────────────────
st.markdown(f"""
<div style="display:grid;grid-template-columns:repeat(4,1fr);gap:14px;margin-bottom:24px;">
  <div style="background:white;border:1px solid #e8edf2;border-radius:16px;
              padding:24px 18px;text-align:center;box-shadow:0 2px 6px rgba(0,0,0,0.04);">
    <div style="font-size:2.2rem;font-weight:800;color:#1a2e44;line-height:1;margin-bottom:6px;">{n_hosp:,}</div>
    <div style="font-size:0.82rem;color:#6b7a8d;font-weight:500;">🏥 Total Hospitals</div>
  </div>
  <div style="background:white;border:1px solid #e8edf2;border-radius:16px;
              padding:24px 18px;text-align:center;box-shadow:0 2px 6px rgba(0,0,0,0.04);">
    <div style="font-size:2.2rem;font-weight:800;color:#1a2e44;line-height:1;margin-bottom:6px;">{n_states}</div>
    <div style="font-size:0.82rem;color:#6b7a8d;font-weight:500;">🗺️ States Covered</div>
  </div>
  <div style="background:white;border:1px solid #e8edf2;border-radius:16px;
              padding:24px 18px;text-align:center;box-shadow:0 2px 6px rgba(0,0,0,0.04);">
    <div style="font-size:2.2rem;font-weight:800;color:#1a2e44;line-height:1;margin-bottom:6px;">{n_dist}</div>
    <div style="font-size:0.82rem;color:#6b7a8d;font-weight:500;">📍 Districts</div>
  </div>
  <div style="background:white;border:1px solid #e8edf2;border-radius:16px;
              padding:24px 18px;text-align:center;box-shadow:0 2px 6px rgba(0,0,0,0.04);">
    <div style="font-size:2.2rem;font-weight:800;color:#1a2e44;line-height:1;margin-bottom:6px;">{accuracy}%</div>
    <div style="font-size:0.82rem;color:#6b7a8d;font-weight:500;">🎯 Model Accuracy</div>
  </div>
</div>
""", unsafe_allow_html=True)

# ── Search panel ───────────────────────────────────────────────────
st.markdown("""
<div style="background:white;border:1px solid #e8edf2;border-radius:18px;
            padding:28px 30px 20px;box-shadow:0 2px 10px rgba(0,0,0,0.05);margin-bottom:24px;">
  <div style="font-size:1.1rem;font-weight:700;color:#1a2e44;margin-bottom:18px;">
    🔍 Find Your Best Hospital District
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
    st.markdown("<p style='font-size:13px;color:transparent;margin-bottom:4px;'>.</p>", unsafe_allow_html=True)
    clicked = st.button("🔍  Find Districts", use_container_width=True)

st.markdown("</div>", unsafe_allow_html=True)

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
        medals      = ['🥇','🥈','🥉','4','5']
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
            <div style="display:flex;align-items:center;gap:14px;padding:14px 16px;
                        border-radius:12px;border:1px solid #eef1f5;
                        margin-bottom:10px;background:white;">
              <div style="width:36px;height:36px;border-radius:50%;
                          background:{badge_bg[i]};color:{badge_color[i]};
                          font-size:0.9rem;font-weight:700;flex-shrink:0;
                          display:flex;align-items:center;justify-content:center;">
                {medals[i]}
              </div>
              <div style="font-weight:700;font-size:1rem;color:#1a2e44;flex:1;">
                {row['District']}
              </div>
              <div style="display:flex;gap:16px;flex-wrap:wrap;">
                <span style="font-size:0.76rem;color:#6b7a8d;">⭐ Rating <b style="color:#1a2e44;">{row['Avg_Rating']}</b></span>
                <span style="font-size:0.76rem;color:#6b7a8d;">📊 Score <b style="color:#1a2e44;">{row['Avg_Score']}</b></span>
                <span style="font-size:0.76rem;color:#6b7a8d;">💰 Afford. <b style="color:#1a2e44;">{row['Avg_Afford']}</b></span>
                <span style="font-size:0.76rem;color:#6b7a8d;">🏥 <b style="color:#1a2e44;">{int(row['Total_Hospitals'])}</b> hospitals</span>
              </div>
              <span style="padding:4px 12px;border-radius:999px;font-size:0.76rem;
                           font-weight:600;white-space:nowrap;{ps}">{rc}</span>
              <div style="background:#1a5276;color:white;padding:4px 14px;
                          border-radius:999px;font-size:0.79rem;font-weight:600;
                          white-space:nowrap;">⚡ {row['Combined_Score']}</div>
            </div>"""

        st.markdown(f"""
        <div style="background:white;border:1px solid #e8edf2;border-radius:18px;
                    padding:28px 30px;box-shadow:0 2px 10px rgba(0,0,0,0.05);">
          <div style="font-size:1.1rem;font-weight:700;color:#1a2e44;margin-bottom:18px;">
            🏆 Top 5 Districts in <b>{state}</b> for <b>{disease}</b>
          </div>
          {rows_html}
        </div>
        """, unsafe_allow_html=True)

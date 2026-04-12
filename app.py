import streamlit as st
import pandas as pd
import numpy as np
from catboost import CatBoostRegressor
import os

# ─── CONFIGURATION ───
st.set_page_config(
    page_title="Punching Shear Prediction Web",
    page_icon="💎",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ─── PREMIUM CUSTOM CSS ───
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600;800&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Outfit', sans-serif;
        background: linear-gradient(135deg, #f8fafc 0%, #f1f5f9 100%);
        color: #1e293b;
    }
    
    /* Center the app container */
    .block-container {
        padding-top: 2rem;
        max-width: 1200px;
    }
    
    /* Header Styling */
    .app-header {
        text-align: center;
        margin-bottom: 3rem;
    }
    
    .app-title {
        font-size: 42px;
        font-weight: 800;
        background: linear-gradient(90deg, #0f172a, #004c6d);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 10px;
    }
    
    .app-subtitle {
        font-size: 16px;
        color: #64748b;
        letter-spacing: 1px;
        text-transform: uppercase;
    }

    /* Professional Cards */
    .glass-card {
        background: rgba(255, 255, 255, 0.95);
        backdrop-filter: blur(10px);
        border: 1px solid rgba(255, 255, 255, 0.2);
        padding: 40px;
        border-radius: 32px;
        box-shadow: 0 20px 50px rgba(0,0,0,0.05);
        height: 100%;
        transition: transform 0.3s ease;
    }
    
    .glass-card:hover {
        transform: translateY(-5px);
    }
    
    .section-title {
        font-size: 20px;
        font-weight: 700;
        color: #0f172a;
        margin-bottom: 25px;
        border-left: 5px solid #008df9;
        padding-left: 15px;
    }

    /* Input Styling */
    .stNumberInput label {
        font-size: 14px !important;
        font-weight: 600 !important;
        color: #475569 !important;
    }
    
    /* Button Premium */
    .stButton > button {
        background: linear-gradient(135deg, #004c6d, #008df9);
        color: white;
        font-weight: 700;
        font-size: 18px;
        padding: 0.8rem 2rem;
        border-radius: 16px;
        border: none;
        width: 100%;
        margin-top: 20px;
        box-shadow: 0 10px 20px rgba(0,76,109,0.2);
        transition: all 0.3s cubic-bezier(0.175, 0.885, 0.32, 1.275);
    }
    
    .stButton > button:hover {
        transform: scale(1.02);
        box-shadow: 0 15px 30px rgba(0,76,109,0.3);
        color: white;
    }
    
    /* Results Styling */
    .result-display {
        background: #f8fafc;
        border-radius: 20px;
        padding: 30px;
        text-align: center;
        border: 1px solid #e2e8f0;
    }
    
    .vu-text {
        font-size: 72px;
        font-weight: 800;
        color: #0f172a;
        margin: 0;
        line-height:1;
    }
    
    .unit-text {
        font-size: 24px;
        color: #64748b;
        font-weight: 600;
    }

    /* Feature Detail List */
    .feature-list-box {
        background: #ffffff;
        border: 1.5px dashed #008df9;
        border-radius: 20px;
        padding: 25px;
        margin-top: 30px;
    }
    
    .feature-item {
        display: flex;
        justify-content: space-between;
        padding: 8px 0;
        border-bottom: 1px solid #f1f5f9;
        font-family: 'Courier New', monospace;
        font-size: 14px;
    }
    
    .feature-item:last-child {
        border-bottom: none;
    }
    </style>
""", unsafe_allow_html=True)

# ─── HEADER ───
st.markdown("""
<div class="app-header">
    <p class="app-subtitle">Advanced Machine Learning Predictor</p>
    <h1 class="app-title">Punching Shear Strength Web</h1>
</div>
""", unsafe_allow_html=True)

# ─── MAIN LAYOUT ───
col_input, col_result = st.columns([1.1, 0.9], gap="large")

with col_input:
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">Geometry & Materials</div>', unsafe_allow_html=True)
    
    g1, g2 = st.columns(2)
    with g1:
        h_slab = st.number_input("📏 h - Slab thickness (mm)", value=200.0, step=1.0)
        c_cov = st.number_input("🛡️ c_cov - Concrete cover (mm)", value=30.0, step=1.0)
        L_span = st.number_input("↔️ L - Span length (mm)", value=2000.0, step=10.0)
        c = st.number_input("⬛ c - Column width (mm)", value=300.0, step=1.0)

    with g2:
        fc = st.number_input("🧪 f'c - Concrete strength (MPa)", value=30.0, step=0.1)
        rho = st.number_input("⛓️ ρ - Reinf. ratio (%)", value=1.0, step=0.01)
        Dop = st.number_input("🔲 Dop - Opening size (mm)", value=0.0, step=1.0)
        Sop = st.number_input("📍 Sop - Opening distance (mm)", value=1000.0, step=1.0)

    # Internal Calculations
    d_eff = h_slab - c_cov
    a_span = L_span / 2
    ad_ratio = a_span / d_eff if d_eff > 0 else 0

    st.markdown("<br>", unsafe_allow_html=True)
    calculate = st.button("Calculate Prediction (Vu)")
    st.markdown('</div>', unsafe_allow_html=True)

with col_result:
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">Prediction Results</div>', unsafe_allow_html=True)
    
    if calculate:
        try:
            # Load Model
            model = CatBoostRegressor()
            model.load_model('best_catboost_model.cbm')
            
            # Prepare Input (7 features model)
            input_df = pd.DataFrame([[d_eff, c, fc, rho, ad_ratio, Dop, Sop]], 
                                    columns=['d_mm', 'C_mm', 'fc_prime_MPa', 'rho_percent', 'a_over_d', 'Opening_Size_mm', 'Opening_Dist_mm'])
            prediction = model.predict(input_df)[0]
            
            # Display Result
            st.markdown(f"""
                <div class="result-display">
                    <p style="color: #008df9; font-weight: 700; margin-bottom: 10px;">PREDICTED PUNCHING STRENGTH</p>
                    <h1 class="vu-text">{prediction:.2f}</h1>
                    <p class="unit-text">kN</p>
                </div>
            """, unsafe_allow_html=True)
            
            # Detailed Variable List (Derived values)
            st.markdown(f"""
                <div class="feature-list-box">
                    <p style="font-weight: 800; color: #0f172a; margin-bottom: 15px;">• Detailed Calculated Values</p>
                    <div class="feature-item"><span>d (h - c_cov)</span> <span>{d_eff:.2f} mm</span></div>
                    <div class="feature-item"><span>a/d ((L/2)/d)</span> <span>{ad_ratio:.3f}</span></div>
                </div>
            """, unsafe_allow_html=True)
            
            st.markdown("<br>", unsafe_allow_html=True)
            st.info("💡 This result is calculated based on the optimized CatBoost model with R² = 0.988.")
            
        except Exception as e:
            st.error(f"Error loading model or predicting: {e}")
    else:
        st.markdown("""
            <div class="result-display" style="opacity: 0.5;">
                <p style="font-weight: 700; margin-bottom: 10px;">STATUS: WAITING</p>
                <h1 class="vu-text">---.--</h1>
                <p class="unit-text">kN</p>
            </div>
            <p style="text-align: center; color: #94a3b8; font-size: 14px; margin-top: 20px;">
                Please input parameters on the left and click 'Calculate Prediction'
            </p>
        """, unsafe_allow_html=True)
        
    st.markdown('</div>', unsafe_allow_html=True)

# ─── FOOTER ───
st.markdown("""
    <div style="text-align: center; margin-top: 4rem; padding-bottom: 2rem;">
        <p style="color: #94a3b8; font-size: 12px; font-weight: 500;">
            © 2026 • AI-POWERED PUNCHING SHEAR PREDICTOR • RESEARCH BY OWNER & UTC TEAM
        </p>
    </div>
""", unsafe_allow_html=True)

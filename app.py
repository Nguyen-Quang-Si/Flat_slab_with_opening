import os
import joblib
import numpy as np
import pandas as pd
import streamlit as st

# ─── CONFIGURATION ───
st.set_page_config(
    page_title="Punching Shear Prediction Tool",
    page_icon="💎",
    layout="wide",
    initial_sidebar_state="collapsed",
)

MODEL_FILE = "best_random_forest_model.joblib"

# Training-domain limits reported in Table 2 of the manuscript
TRAINING_RANGES = {
    "d": (30.0, 500.0),                 # mm
    "c": (40.06, 1000.0),              # mm
    "sqrt_fc": (2.943, 10.895),         # sqrt(MPa)
    "rho": (0.22, 3.73),                # %
    "a_over_d": (0.62, 13.55),          # -
    "Dop": (0.0, 700.0),                # mm
    "Sop": (0.0, 450.0),                # mm
}

# ─── CUSTOM CSS ───
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600;800&display=swap');

    html, body, [class*="css"] {
        font-family: 'Outfit', sans-serif;
        background: linear-gradient(135deg, #f8fafc 0%, #f1f5f9 100%);
        color: #1e293b;
    }

    .block-container {
        padding-top: 2rem;
        max-width: 1200px;
    }

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

    .glass-card {
        background: rgba(255, 255, 255, 0.95);
        backdrop-filter: blur(10px);
        border: 1px solid rgba(255, 255, 255, 0.2);
        padding: 40px;
        border-radius: 32px;
        box-shadow: 0 20px 50px rgba(0,0,0,0.05);
        height: 100%;
    }

    .section-title {
        font-size: 20px;
        font-weight: 700;
        color: #0f172a;
        margin-bottom: 25px;
        border-left: 5px solid #008df9;
        padding-left: 15px;
    }

    .stNumberInput label {
        font-size: 14px !important;
        font-weight: 600 !important;
        color: #475569 !important;
    }

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
    }

    .stButton > button:hover {
        transform: scale(1.02);
        color: white;
    }

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
        line-height: 1;
    }

    .unit-text {
        font-size: 24px;
        color: #64748b;
        font-weight: 600;
    }

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
    """,
    unsafe_allow_html=True,
)


# ─── LOAD RANDOM FOREST MODEL ───
@st.cache_resource
def load_model():
    if not os.path.exists(MODEL_FILE):
        raise FileNotFoundError(
            f"Model file '{MODEL_FILE}' was not found. "
            "Export the final optimized Random Forest model with joblib "
            "and place it in the same GitHub repository as app.py."
        )
    return joblib.load(MODEL_FILE)


def _normalize_feature_name(name):
    """Normalize a feature name for robust matching."""
    return (
        str(name)
        .strip()
        .lower()
        .replace("'", "")
        .replace("√", "sqrt")
        .replace("/", "_over_")
        .replace("-", "_")
        .replace(" ", "_")
        .replace("(", "")
        .replace(")", "")
    )


def build_model_input(model, d_eff, c, sqrt_fc, rho, ad_ratio, Dop, Sop):
    """
    The RF model used in the paper has seven predictors, in this physical order:
    d, c, sqrt(f'c), rho, a/d, Dop, Sop.

    If the fitted sklearn model stores feature_names_in_, this function uses those
    exact names. Otherwise it passes a NumPy array in the seven-feature order.
    """
    values_by_key = {
        "d": d_eff,
        "c": c,
        "sqrt_fc": sqrt_fc,
        "rho": rho,
        "a_over_d": ad_ratio,
        "dop": Dop,
        "sop": Sop,
    }

    aliases = {
        # effective depth
        "d": "d",
        "d_mm": "d",
        "effective_depth": "d",
        "effective_depth_mm": "d",

        # column width
        "c": "c",
        "c_mm": "c",
        "column_width": "c",
        "column_width_mm": "c",

        # transformed concrete strength sqrt(f'c)
        "sqrt_fc": "sqrt_fc",
        "sqrt_fc_mpa": "sqrt_fc",
        "sqrt_fc_prime": "sqrt_fc",
        "sqrt_fc_prime_mpa": "sqrt_fc",
        "sqrt_f_c": "sqrt_fc",
        "sqrt_f_c_prime": "sqrt_fc",
        "fc_sqrt": "sqrt_fc",
        "fc_prime_sqrt": "sqrt_fc",

        # reinforcement ratio
        "rho": "rho",
        "rho_percent": "rho",
        "reinforcement_ratio": "rho",
        "flexural_reinforcement_ratio": "rho",

        # shear span-to-depth ratio
        "a_over_d": "a_over_d",
        "a_d": "a_over_d",
        "shear_span_to_depth_ratio": "a_over_d",

        # opening size
        "dop": "dop",
        "dop_mm": "dop",
        "opening_size": "dop",
        "opening_size_mm": "dop",

        # opening distance
        "sop": "sop",
        "sop_mm": "sop",
        "opening_dist": "sop",
        "opening_dist_mm": "sop",
        "opening_distance": "sop",
        "opening_distance_mm": "sop",
        "opening_distance_to_column_face": "sop",
    }

    if hasattr(model, "feature_names_in_"):
        model_columns = list(model.feature_names_in_)
        row = {}

        for original_name in model_columns:
            normalized = _normalize_feature_name(original_name)

            # Direct/alias match
            canonical_key = aliases.get(normalized)

            # A few flexible pattern matches
            if canonical_key is None:
                if normalized.startswith("sqrt") and ("fc" in normalized or "concrete" in normalized):
                    canonical_key = "sqrt_fc"
                elif "opening" in normalized and ("size" in normalized or "dop" in normalized):
                    canonical_key = "dop"
                elif "opening" in normalized and ("dist" in normalized or "distance" in normalized or "sop" in normalized):
                    canonical_key = "sop"
                elif "rho" in normalized or "reinforcement_ratio" in normalized:
                    canonical_key = "rho"
                elif "a_over_d" in normalized or "shear_span" in normalized:
                    canonical_key = "a_over_d"
                elif normalized in {"column", "column_dimension", "column_width"}:
                    canonical_key = "c"
                elif normalized in {"depth", "effective_depth"}:
                    canonical_key = "d"

            if canonical_key is None:
                raise ValueError(
                    "The saved RF model contains an unrecognized feature name: "
                    f"'{original_name}'. Model features are: {model_columns}. "
                    "Please make the exported model use the seven predictors "
                    "[d, c, sqrt(f'c), rho, a/d, Dop, Sop]."
                )

            row[original_name] = values_by_key[canonical_key]

        return pd.DataFrame([row], columns=model_columns)

    # Fallback if the model was fitted from a NumPy array
    return np.array(
        [[d_eff, c, sqrt_fc, rho, ad_ratio, Dop, Sop]],
        dtype=float,
    )


def outside_range(value, bounds):
    return value < bounds[0] or value > bounds[1]


# ─── HEADER ───
st.markdown(
    """
    <div class="app-header">
        <p class="app-subtitle">Advanced Machine Learning Predictor</p>
        <h1 class="app-title">Punching Shear Strength Prediction Tool</h1>
    </div>
    """,
    unsafe_allow_html=True,
)


# ─── MAIN LAYOUT ───
col_input, col_result = st.columns([1.1, 0.9], gap="large")

with col_input:
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.markdown(
        '<div class="section-title">Geometry & Materials</div>',
        unsafe_allow_html=True,
    )

    g1, g2 = st.columns(2)

    with g1:
        h_slab = st.number_input(
            "📏 h — Slab thickness (mm)",
            min_value=1.0,
            value=200.0,
            step=1.0,
        )
        c_cov = st.number_input(
            "🛡️ cover — Concrete cover (mm)",
            min_value=0.0,
            value=30.0,
            step=1.0,
        )
        a_span = st.number_input(
            "↔️ a — Shear span (mm)",
            min_value=1.0,
            value=1000.0,
            step=10.0,
            help="Enter the shear span a directly. The application automatically calculates a/d.",
        )
        c = st.number_input(
            "⬛ c — Column width (mm)",
            min_value=1.0,
            value=300.0,
            step=1.0,
        )

    with g2:
        fc = st.number_input(
            "🧪 f'c — Concrete compressive strength (MPa)",
            min_value=0.1,
            value=30.0,
            step=0.1,
            help="The RF model uses sqrt(f'c); the transformation is calculated automatically.",
        )
        rho = st.number_input(
            "⛓️ ρ — Flexural reinforcement ratio (%)",
            min_value=0.01,
            value=1.0,
            step=0.01,
        )
        Dop = st.number_input(
            "🔲 Dop — Opening size (mm)",
            min_value=0.0,
            value=0.0,
            step=1.0,
        )
        Sop_input = st.number_input(
            "📍 Sop — Opening distance to column face (mm)",
            min_value=0.0,
            value=0.0,
            step=1.0,
        )

    # ── Derived model predictors ──
    d_eff = h_slab - c_cov
    ad_ratio = a_span / d_eff if d_eff > 0 else np.nan
    sqrt_fc = np.sqrt(fc)

    # For solid slabs, use the database convention Dop = Sop = 0
    Sop_model = 0.0 if Dop == 0.0 else Sop_input

    # ── Input validation ──
    if d_eff <= 0:
        st.error("⚠️ Effective depth d = h − cover must be greater than 0.")

    if Dop == 0.0 and Sop_input != 0.0:
        st.info(
            "ℹ️ Dop = 0 indicates a solid slab. Sop is therefore set to 0 for prediction "
            "to remain consistent with the database encoding."
        )

    # Applicability-domain warnings
    domain_warnings = []
    if d_eff > 0 and outside_range(d_eff, TRAINING_RANGES["d"]):
        domain_warnings.append(
            f"d = {d_eff:.1f} mm is outside the training range "
            f"{TRAINING_RANGES['d'][0]:.0f}–{TRAINING_RANGES['d'][1]:.0f} mm."
        )
    if outside_range(c, TRAINING_RANGES["c"]):
        domain_warnings.append(
            f"c = {c:.1f} mm is outside the training range "
            f"{TRAINING_RANGES['c'][0]:.2f}–{TRAINING_RANGES['c'][1]:.0f} mm."
        )
    if outside_range(sqrt_fc, TRAINING_RANGES["sqrt_fc"]):
        domain_warnings.append(
            f"sqrt(f'c) = {sqrt_fc:.3f} is outside the training range "
            f"{TRAINING_RANGES['sqrt_fc'][0]:.3f}–{TRAINING_RANGES['sqrt_fc'][1]:.3f}."
        )
    if outside_range(rho, TRAINING_RANGES["rho"]):
        domain_warnings.append(
            f"ρ = {rho:.2f}% is outside the training range "
            f"{TRAINING_RANGES['rho'][0]:.2f}–{TRAINING_RANGES['rho'][1]:.2f}%."
        )
    if d_eff > 0 and outside_range(ad_ratio, TRAINING_RANGES["a_over_d"]):
        domain_warnings.append(
            f"a/d = {ad_ratio:.3f} is outside the training range "
            f"{TRAINING_RANGES['a_over_d'][0]:.2f}–{TRAINING_RANGES['a_over_d'][1]:.2f}."
        )
    if outside_range(Dop, TRAINING_RANGES["Dop"]):
        domain_warnings.append(
            f"Dop = {Dop:.1f} mm is outside the training range "
            f"{TRAINING_RANGES['Dop'][0]:.0f}–{TRAINING_RANGES['Dop'][1]:.0f} mm."
        )
    if outside_range(Sop_model, TRAINING_RANGES["Sop"]):
        domain_warnings.append(
            f"Sop = {Sop_model:.1f} mm is outside the training range "
            f"{TRAINING_RANGES['Sop'][0]:.0f}–{TRAINING_RANGES['Sop'][1]:.0f} mm."
        )

    if domain_warnings:
        st.warning(
            "⚠️ Applicability-domain warning:\n\n- "
            + "\n- ".join(domain_warnings)
            + "\n\nPredictions outside the training domain should be interpreted with caution."
        )

    st.markdown("<br>", unsafe_allow_html=True)
    calculate = st.button("⚡ Calculate Prediction (Vu)")

    st.caption(
        "Derived automatically: d = h − cover; a/d is calculated from the entered shear span a and derived d; and sqrt(f'c) is calculated from f'c. "
        "If 'cover' is clear concrete cover rather than distance to the reinforcement centroid, "
        "d is an approximation; for greater precision, account for half the bar diameter."
    )

    st.markdown("</div>", unsafe_allow_html=True)


with col_result:
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.markdown(
        '<div class="section-title">Prediction Results</div>',
        unsafe_allow_html=True,
    )

    if calculate:
        if d_eff <= 0:
            st.error("Cannot predict because effective depth d ≤ 0.")
        else:
            try:
                model = load_model()

                X = build_model_input(
                    model=model,
                    d_eff=d_eff,
                    c=c,
                    sqrt_fc=sqrt_fc,
                    rho=rho,
                    ad_ratio=ad_ratio,
                    Dop=Dop,
                    Sop=Sop_model,
                )

                prediction = float(model.predict(X)[0])
                prediction = max(0.0, prediction)

                st.markdown(
                    f"""
                    <div class="result-display">
                        <p style="color: #008df9; font-weight: 700; margin-bottom: 10px;">
                            PREDICTED PUNCHING SHEAR STRENGTH
                        </p>
                        <h1 class="vu-text">{prediction:.2f}</h1>
                        <p class="unit-text">kN</p>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

                st.markdown(
                    f"""
                    <div class="feature-list-box">
                        <p style="font-weight: 800; color: #0f172a; margin-bottom: 15px;">
                            • Derived model inputs
                        </p>
                        <div class="feature-item"><span>d = h − cover</span><span>{d_eff:.1f} mm</span></div>
                        <div class="feature-item"><span>a</span><span>{a_span:.1f} mm</span></div>
                        <div class="feature-item"><span>a/d</span><span>{ad_ratio:.3f}</span></div>
                        <div class="feature-item"><span>sqrt(f'c)</span><span>{sqrt_fc:.3f}</span></div>
                        <div class="feature-item"><span>Dop</span><span>{Dop:.1f} mm</span></div>
                        <div class="feature-item"><span>Sop used by model</span><span>{Sop_model:.1f} mm</span></div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

                st.markdown("<br>", unsafe_allow_html=True)
                st.info(
                    "💡 Optimized Random Forest model · "
                    "Testing R² = 0.9720 · RMSE = 33.07 kN · "
                    "MAE = 22.84 kN · MAPE = 11.86%."
                )

                if domain_warnings:
                    st.caption(
                        "The prediction contains one or more inputs outside the experimental "
                        "training domain and should therefore be interpreted with caution."
                    )

            except Exception as e:
                st.error(f"Error: {e}")
    else:
        st.markdown(
            """
            <div class="result-display" style="opacity: 0.5;">
                <p style="font-weight: 700; margin-bottom: 10px;">STATUS: WAITING</p>
                <h1 class="vu-text">---.--</h1>
                <p class="unit-text">kN</p>
            </div>
            <p style="text-align: center; color: #94a3b8; font-size: 14px; margin-top: 20px;">
                Enter parameters on the left and click 'Calculate Prediction'
            </p>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("</div>", unsafe_allow_html=True)


# ─── FOOTER ───
st.markdown(
    """
    <div style="text-align: center; margin-top: 4rem; padding-bottom: 2rem;">
        <p style="color: #94a3b8; font-size: 12px; font-weight: 500;">
            © 2026 · AI-Powered Punching Shear Prediction Tool ·
            University of Transport and Communications (UTC Team)
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)

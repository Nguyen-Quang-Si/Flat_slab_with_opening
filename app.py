from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(
    page_title="OpenPunch-RF",
    page_icon="◼",
    layout="wide",
    initial_sidebar_state="expanded",
)

BASE_DIR = Path(__file__).resolve().parent
MODEL_FILES = [
    BASE_DIR / "best_rf_model.joblib",
    BASE_DIR / "best_random_forest_model.joblib",
]

RANGES = {
    "d": (30.0, 500.0),
    "c": (40.06, 1000.0),
    "sqrt_fc": (2.943, 10.895),
    "rho": (0.22, 3.73),
    "ad": (0.62, 13.55),
    "Dop": (0.0, 700.0),
    "Sop": (0.0, 450.0),
}

st.markdown("""
<style>
:root{
  --bg:#0b0f15; --side:#111827; --line:#273142; --text:#f4f7fb;
  --muted:#8e9bab; --blue:#78c8f4; --yellow:#4d4a11; --yellowb:#6d681d;
  --red:#4c1f25; --redb:#78343c; --green:#123623; --greenb:#255d3d;
}
html,body,[data-testid="stAppViewContainer"]{background:var(--bg);color:var(--text);}
[data-testid="stHeader"]{background:rgba(11,15,21,.92);}
[data-testid="stSidebar"]{background:var(--side);border-right:1px solid #222c38;}
.block-container{padding-top:1.6rem;max-width:1450px;}
#MainMenu, footer{visibility:hidden;}

.kicker{font-size:.72rem;color:#9ba8b8;font-weight:800;text-transform:uppercase;letter-spacing:.1em}
.title{font-size:2.35rem;font-weight:850;color:#fff;line-height:1.05;margin:.25rem 0 .35rem}
.subtitle{font-size:.92rem;color:#aab4c2;line-height:1.55;max-width:1050px}
.notice{background:var(--yellow);border:1px solid var(--yellowb);color:#eee9a8;
padding:.72rem .9rem;border-radius:4px;font-size:.8rem;line-height:1.4;margin-top:1rem}
.sec{font-size:1.02rem;font-weight:850;color:#fff;margin:1.2rem 0 .55rem}
.metric-caption{font-size:.68rem;color:#99a5b5;font-weight:800;text-transform:uppercase;letter-spacing:.05em}
.metric-value{font-size:2rem;font-weight:850;color:#fff;line-height:1.05}
.metric-unit{font-size:.84rem;color:#8895a5;margin-left:.2rem}
.mini{height:54px;border-top:1px solid #202936;border-bottom:1px solid #202936;
display:flex;align-items:flex-end;padding:.32rem 0;margin-top:.35rem}
.bar{width:100%;background:var(--blue)}
.good{background:var(--green);border:1px solid var(--greenb);color:#baf0cc;
padding:.72rem .9rem;border-radius:4px;font-size:.82rem;line-height:1.45}
.bad{background:var(--red);border:1px solid var(--redb);color:#ffb9bf;
padding:.72rem .9rem;border-radius:4px;font-size:.82rem;line-height:1.45}
.sidehead{font-size:.95rem;font-weight:850;color:#fff;margin:.1rem 0 .65rem}
.sidesub{font-size:.68rem;font-weight:850;color:#aab4c2;text-transform:uppercase;
letter-spacing:.06em;margin:.8rem 0 .3rem}
.derived{background:#0d1420;border:1px solid #273142;border-radius:4px;padding:.7rem .75rem;
font-size:.77rem;line-height:1.6;color:#c3cbd6}
[data-testid="stNumberInput"] input,
[data-testid="stSelectbox"] div[data-baseweb="select"]>div{
background:#192231!important;color:#fff!important;border-color:#2e3948!important}
[data-testid="stNumberInput"] button{
background:#151e2a!important;color:#dce4ee!important;border-color:#2e3948!important}
.stButton>button{width:100%;border-radius:4px;border:1px solid #2c8dbd;
background:#1177aa;color:#fff;font-weight:850;min-height:2.7rem}
.stButton>button:hover{background:#168bc5;border-color:#45b9ef;color:#fff}
[data-testid="stDataFrame"]{border:1px solid #273142;border-radius:4px;overflow:hidden}
[data-testid="stExpander"]{background:#0f1623;border:1px solid #273142;border-radius:4px}
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_model():
    for p in MODEL_FILES:
        if p.exists():
            return joblib.load(p), p.name
    raise FileNotFoundError(
        "Random Forest model not found. Put best_rf_model.joblib "
        "or best_random_forest_model.joblib beside app.py."
    )

def norm(s):
    return (str(s).strip().lower().replace("'", "").replace("√","sqrt")
            .replace("/","_over_").replace("-","_").replace(" ","_")
            .replace("(","").replace(")",""))

def make_X(model, d, c, sqrt_fc, rho, ad, Dop, Sop):
    vals = {
        "d":d, "c":c, "sqrt_fc":sqrt_fc, "rho":rho,
        "a_over_d":ad, "dop":Dop, "sop":Sop
    }
    aliases = {
        "d":"d","d_mm":"d","effective_depth":"d","effective_depth_mm":"d",
        "c":"c","c_mm":"c","column_width":"c","column_dimension":"c",
        "sqrt_fc":"sqrt_fc","sqrt_f_c":"sqrt_fc","sqrt_fc_prime":"sqrt_fc",
        "sqrt_f_c_prime":"sqrt_fc","sqrt_concrete_strength":"sqrt_fc",
        "rho":"rho","rho_percent":"rho","reinforcement_ratio":"rho",
        "flexural_reinforcement_ratio":"rho",
        "a_over_d":"a_over_d","a_d":"a_over_d","shear_span_to_depth_ratio":"a_over_d",
        "dop":"dop","dop_mm":"dop","opening_size":"dop","opening_size_mm":"dop",
        "sop":"sop","sop_mm":"sop","opening_distance":"sop",
        "opening_distance_mm":"sop","opening_distance_to_column_face":"sop",
    }
    if hasattr(model, "feature_names_in_"):
        cols = list(model.feature_names_in_)
        row = {}
        for col in cols:
            n = norm(col)
            key = aliases.get(n)
            if key is None and n.startswith("sqrt") and ("fc" in n or "concrete" in n):
                key = "sqrt_fc"
            if key is None and ("opening" in n and ("size" in n or "dop" in n)):
                key = "dop"
            if key is None and ("opening" in n and ("distance" in n or "sop" in n or "dist" in n)):
                key = "sop"
            if key is None and ("rho" in n or "reinforcement_ratio" in n):
                key = "rho"
            if key is None and ("a_over_d" in n or "shear_span" in n):
                key = "a_over_d"
            if key is None:
                raise ValueError(f"Unrecognized feature name in saved model: {col}")
            row[col] = vals[key]
        return pd.DataFrame([row], columns=cols)
    return np.array([[d,c,sqrt_fc,rho,ad,Dop,Sop]], dtype=float)

def in_range(v, lo, hi):
    return lo <= v <= hi

def metric_card(caption, value, unit="", ratio=.5):
    h = max(4, min(46, 46*float(ratio)))
    st.markdown(f"""
    <div>
      <div class="metric-caption">{caption}</div>
      <span class="metric-value">{value}</span><span class="metric-unit">{unit}</span>
      <div class="mini"><div class="bar" style="height:{h:.1f}px"></div></div>
    </div>
    """, unsafe_allow_html=True)

# ---------------- SIDEBAR ----------------
with st.sidebar:
    st.markdown('<div class="sidehead">Input case</div>', unsafe_allow_html=True)

    a1, a2 = st.columns(2)
    with a1:
        h = st.number_input("h — slab thickness (mm)", min_value=1.0, value=200.0, step=1.0)
        cover = st.number_input("cover (mm)", min_value=0.0, value=30.0, step=1.0)
        a = st.number_input("a — shear span (mm)", min_value=1.0, value=1000.0, step=10.0)
        c = st.number_input("c — column width (mm)", min_value=1.0, value=300.0, step=1.0)

    with a2:
        fc = st.number_input("f'c (MPa)", min_value=0.1, value=30.0, step=0.1)
        rho = st.number_input("ρ (%)", min_value=0.01, value=1.00, step=0.01)
        Dop_ui = st.number_input("Dop (mm)", min_value=0.0, value=0.0, step=1.0)
        Sop_ui = st.number_input("Sop (mm)", min_value=0.0, value=0.0, step=1.0)

    st.markdown('<div class="sidesub">Opening configuration</div>', unsafe_allow_html=True)
    case = st.selectbox("Case", ["Solid slab / no opening", "Slab with opening"])

    d = h - cover
    sqrt_fc = np.sqrt(fc)
    ad = a/d if d > 0 else np.nan

    if case == "Solid slab / no opening":
        Dop, Sop = 0.0, 0.0
    else:
        Dop, Sop = Dop_ui, Sop_ui

    st.markdown('<div class="sidesub">Derived model variables</div>', unsafe_allow_html=True)
    if d > 0:
        st.markdown(f"""
        <div class="derived">
          <b>d</b> = {d:.1f} mm<br>
          <b>a/d</b> = {ad:.3f}<br>
          <b>√f'c</b> = {sqrt_fc:.3f}<br>
          <b>Dop used</b> = {Dop:.1f} mm<br>
          <b>Sop used</b> = {Sop:.1f} mm
        </div>
        """, unsafe_allow_html=True)
    else:
        st.error("d = h − cover must be greater than zero.")

    run = st.button("Run RF diagnostic")

# ---------------- HEADER ----------------
st.markdown('<div class="kicker">Random Forest diagnostic predictor</div>', unsafe_allow_html=True)
st.markdown('<div class="title">OpenPunch-RF</div>', unsafe_allow_html=True)
st.markdown("""
<div class="subtitle">
A diagnostic benchmark tool for punching shear prediction of RC flat slabs with openings.
The interface keeps practical inputs <b>h</b>, <b>cover</b>, and shear span <b>a</b>, while
internally deriving the seven predictors used by the optimized Random Forest model.
</div>
""", unsafe_allow_html=True)

st.markdown("""
<div class="notice">
<b>Research diagnostic.</b> Predictions are data-driven and should be interpreted within
the experimental domain used to train the model. The application is intended for research
and preliminary assessment, not as a replacement for code-based design verification.
</div>
""", unsafe_allow_html=True)

if "pred" not in st.session_state:
    st.session_state.pred = None
    st.session_state.model_name = None
    st.session_state.err = None

if run:
    if d <= 0:
        st.session_state.err = "Effective depth d must be greater than zero."
        st.session_state.pred = None
    else:
        try:
            model, name = load_model()
            X = make_X(model, d, c, sqrt_fc, rho, ad, Dop, Sop)
            st.session_state.pred = max(float(model.predict(X)[0]), 0.0)
            st.session_state.model_name = name
            st.session_state.err = None
        except Exception as e:
            st.session_state.err = str(e)
            st.session_state.pred = None

if st.session_state.err:
    st.error(st.session_state.err)

# ---------------- HEADLINE ----------------
st.markdown('<div class="sec">Headline diagnostic — single case</div>', unsafe_allow_html=True)
c1,c2,c3 = st.columns(3, gap="large")

with c1:
    if st.session_state.pred is None:
        metric_card("Optimized Random Forest", "—", "kN", .35)
    else:
        metric_card("Optimized Random Forest", f"{st.session_state.pred:.1f}", "kN",
                    min(st.session_state.pred/700,1))

with c2:
    metric_card("Derived effective depth, d", f"{d:.1f}" if d>0 else "—", "mm",
                min(max(d,0)/500,1))

with c3:
    metric_card("Derived shear-span ratio, a/d", f"{ad:.3f}" if d>0 else "—", "",
                min(max(ad if d>0 else 0,0)/13.55,1))

# ---------------- DIAGNOSTIC TABLE ----------------
st.markdown('<div class="sec">Applicability bracket — model-input diagnostics</div>', unsafe_allow_html=True)

table = pd.DataFrame([
    ["Effective depth","d",d,"mm",*RANGES["d"]],
    ["Column width","c",c,"mm",*RANGES["c"]],
    ["Concrete-strength transform","√f'c",sqrt_fc,"√MPa",*RANGES["sqrt_fc"]],
    ["Reinforcement ratio","ρ",rho,"%",*RANGES["rho"]],
    ["Shear-span ratio","a/d",ad,"-",*RANGES["ad"]],
    ["Opening size","Dop",Dop,"mm",*RANGES["Dop"]],
    ["Opening distance","Sop",Sop,"mm",*RANGES["Sop"]],
], columns=["Variable","Symbol","Current value","Unit","Training min","Training max"])

table["Status"] = [
    "Within range" if in_range(v,lo,hi) else "Outside range"
    for v,lo,hi in zip(table["Current value"],table["Training min"],table["Training max"])
]

st.dataframe(
    table,
    use_container_width=True,
    hide_index=True,
    column_config={
        "Current value": st.column_config.NumberColumn(format="%.3f"),
        "Training min": st.column_config.NumberColumn(format="%.3f"),
        "Training max": st.column_config.NumberColumn(format="%.3f"),
    },
)

# ---------------- INTERPRETATION ----------------
st.markdown('<div class="sec">Geometry-region interpretation</div>', unsafe_allow_html=True)
outside = table[table["Status"]=="Outside range"]

if outside.empty:
    st.markdown("""
    <div class="good">
    <b>Within the reported experimental domain.</b>
    All seven model predictors for the current case fall within the ranges reported in the study database.
    This does not guarantee specimen-level accuracy, but the case is not an obvious range extrapolation.
    </div>
    """, unsafe_allow_html=True)
else:
    syms = ", ".join(outside["Symbol"].astype(str))
    st.markdown(f"""
    <div class="bad">
    <b>Applicability warning.</b>
    The following predictor(s) fall outside the reported training ranges: <b>{syms}</b>.
    The Random Forest prediction should therefore be interpreted with additional caution.
    </div>
    """, unsafe_allow_html=True)

with st.expander("▸ Random Forest model information"):
    st.write(
        "The app loads `best_rf_model.joblib` or `best_random_forest_model.joblib` "
        "from the same repository folder as app.py."
    )
    if st.session_state.model_name:
        st.write(f"Loaded model: `{st.session_state.model_name}`")
    st.write(
        "Reported independent-test performance used in the manuscript: "
        "R² = 0.9720, RMSE = 33.07 kN, MAE = 22.84 kN, MAPE = 11.86%."
    )

with st.expander("▸ Predictor transformation used by the app"):
    st.markdown(r"""
- \(d = h - cover\)
- \(a/d = a / d\)
- \(\sqrt{f'_c}\) is calculated internally from the entered concrete strength.
- For a solid slab, the model receives \(D_{op}=0\) and \(S_{op}=0\).

The final RF input vector is:
\(d,\;c,\;\sqrt{f'_c},\;\rho,\;a/d,\;D_{op},\;S_{op}\).
""")

with st.expander("▸ Important modelling note"):
    st.write(
        "If 'cover' means clear concrete cover, d = h − cover is an approximation. "
        "For strict section geometry, effective depth should account for the reinforcement-bar centroid."
    )

st.caption("OpenPunch-RF · Research prototype · University of Transport and Communications")

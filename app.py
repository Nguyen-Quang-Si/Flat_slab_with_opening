import streamlit as st
import pandas as pd
from catboost import CatBoostRegressor

st.set_page_config(page_title="Punching Opening Web", layout="wide")

@st.cache_resource
def load_model():
    model = CatBoostRegressor()
    model.load_model("best_catboost_model.cbm")
    return model

model = load_model()

st.markdown("""
<style>
body { background-color: #eef2f7; }
.card {
    background: white;
    padding: 25px;
    border-radius: 20px;
    box-shadow: 0 10px 25px rgba(0,0,0,0.05);
}
.big-result {
    font-size: 52px;
    font-weight: 800;
    color: #1f4e79;
}
</style>
""", unsafe_allow_html=True)

st.title("Beam Open Web")
st.caption("Ứng dụng tính sức kháng sàn có lỗ mở (AI CatBoost)")

col1, col2 = st.columns([1.2, 0.8])

with col1:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("Thông số đầu vào")

    d = st.number_input("d (mm)", 100.0, 600.0, 300.0)
    c = st.number_input("c (mm)", 100.0, 800.0, 300.0)
    fc = st.number_input("f'c (MPa)", 20.0, 100.0, 30.0)
    rho = st.number_input("ρ (%)", 0.1, 5.0, 1.0)
    a = st.number_input("a (mm)", 100.0, 2000.0, 900.0)
    ax = st.number_input("ax (mm)", 0.0, 1000.0, 200.0)
    ay = st.number_input("ay (mm)", 0.0, 1000.0, 200.0)
    Xo = st.number_input("Xo (mm)", 0.0, 1000.0, 300.0)
    Yo = st.number_input("Yo (mm)", 0.0, 1000.0, 300.0)

    run = st.button("Tính Vu")
    st.markdown('</div>', unsafe_allow_html=True)

with col2:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("Kết quả")

    if run:
        Dop = (ax + ay) / 2
        Sop = (Xo + Yo) / 2
        ad = a / d

        df = pd.DataFrame([[d, c, fc, rho, ad, Dop, Sop]],
            columns=[
                'd_mm', 'C_mm', 'fc_prime_MPa',
                'rho_percent', 'a_over_d',
                'Opening_Size_mm', 'Opening_Dist_mm'
            ])

        Vu = model.predict(df)[0]

        st.markdown(f'<div class="big-result">Vu ≈ {Vu:.2f} kN</div>', unsafe_allow_html=True)

        if d > 500 or fc > 100:
            st.error("⚠️ Ngoài miền dữ liệu")
        else:
            st.success("Trạng thái: OK")

        st.write("a/d =", round(ad,2))
        st.write("Dop =", round(Dop,2))
        st.write("Sop =", round(Sop,2))

    else:
        st.write("Nhập dữ liệu và bấm tính")

    st.markdown('</div>', unsafe_allow_html=True)

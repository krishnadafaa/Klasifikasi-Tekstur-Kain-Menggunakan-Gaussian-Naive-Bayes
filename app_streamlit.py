import streamlit as st
import cv2
import numpy as np
import joblib
from skimage.feature import graycomatrix, graycoprops, local_binary_pattern

# Page Configuration
st.set_page_config(page_title="Klasifikasi Kain AI", layout="centered")

# Custom Styles
st.markdown("""
    <style>
        .block-container { padding-top: 2rem; padding-bottom: 2rem; }
        h1 { font-weight: 800; color: #2C3E50; }
        .stButton>button { width: 100%; font-weight: bold; }
    </style>
""", unsafe_allow_html=True)

st.title("Klasifikasi Jenis Kain")
st.caption("Aplikasi berbasis AI dengan Ekstraksi Fitur GLCM + LBP dan Klasifikasi Naive Bayes")

with st.expander("Panduan Penggunaan Aplikasi (Dokumentasi)"):
    st.markdown("""
    1. Siapkan foto sampel kain (Katun, Wol, atau Sintetis).
    2. Pastikan foto tegak lurus, dekat, dan pencahayaan jelas agar serat tekstur terlihat.
    3. Unggah gambar pada kotak di bawah.
    4. Klik tombol Mulai Analisis Kain untuk melihat hasil prediksi AI.
    """)

st.write("")

# Load Models
@st.cache_resource
def load_models():
    scaler = joblib.load('scaler.pkl')
    pca_prep = joblib.load('pca_prep.pkl')
    model = joblib.load('model.pkl')
    return scaler, pca_prep, model

try:
    scaler, pca_prep, model = load_models()
except Exception as e:
    st.error(f"Gagal memuat file model: {e}")

# Feature Extraction
def ekstract_fitur_kain(img_resized, n_levels=16):
    img_quantized = (img_resized // (256 // n_levels)).astype(np.uint8)
    
    # GLCM
    glcm = graycomatrix(img_quantized, distances=[1], angles=[0, np.pi/4, np.pi/2, 3*np.pi/4], levels=n_levels, symmetric=True, normed=True)
    features = []
    for prop in ['contrast', 'dissimilarity', 'homogeneity', 'energy', 'correlation']:
        values = graycoprops(glcm, prop).flatten()
        features.extend(values)
    array_glcm = np.array(features)

    # LBP
    radius = 1 
    n_points = 8 * radius
    lbp = local_binary_pattern(img_resized, n_points, radius, method='uniform')
    n_bins = n_points + 2
    array_lbp, _ = np.histogram(lbp.ravel(), bins=n_bins, range=(0, n_bins), density=True)

    return np.hstack((array_glcm, array_lbp)).reshape(1, -1)

# UI Layout
uploaded_file = st.file_uploader("Unggah gambar serat kain:", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    st.write("")
    
    with st.container(border=True):
        col1, col2 = st.columns([5, 4], gap="large")
        
        with col1:
            st.subheader("Pratinjau Gambar")
            file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
            img_bgr = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
            img_gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
            st.image(img_bgr, channels="BGR", use_container_width=True)
            
        with col2:
            st.subheader("Analisis Sistem")
            st.write("Gambar berhasil dimuat dan siap diekstraksi.")
            
            if st.button("Mulai Analisis Kain", type="primary"):
                with st.spinner("Mengekstrak fitur tekstur..."):
                    img_resized = cv2.resize(img_gray, (128, 128))
                    fitur = ekstract_fitur_kain(img_resized)
                    fitur_scaled = scaler.transform(fitur)
                    fitur_pca = pca_prep.transform(fitur_scaled)
                    prediksi = model.predict(fitur_pca)[0]
                
                st.write("---")
                st.markdown("### Hasil Prediksi:")
                st.info(f"Jenis Kain: {prediksi.upper()}")
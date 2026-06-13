import joblib
import os
import glob
import cv2
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from skimage.feature import graycomatrix, graycoprops, local_binary_pattern
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.decomposition import PCA


def ekstract_fitur_kain(image_path, size=(128, 128), n_levels=16):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    if img is None:
        return None

    img_resized = cv2.resize(img, size)
    img_quantized = (img_resized // (256 // n_levels)).astype(np.uint8)

    distances = [1]
    angles = [0, np.pi / 4, np.pi / 2, 3 * np.pi / 4]
    glcm = graycomatrix(
        img_quantized,
        distances=distances,
        angles=angles,
        levels=n_levels,
        symmetric=True,
        normed=True
    )

    features = []
    for prop in ['contrast', 'dissimilarity', 'homogeneity', 'energy', 'correlation']:
        values = graycoprops(glcm, prop).flatten()
        features.extend(values)
    array_glcm = np.array(features)

    radius = 1 
    n_points = 8 * radius
    lbp = local_binary_pattern(img_resized, n_points, radius, method='uniform')
    n_bins = n_points + 2
    array_lbp, _ = np.histogram(lbp.ravel(), bins=n_bins, range=(0, n_bins), density=True)

    fitur_gabungan = np.hstack ((array_glcm, array_lbp))

    return np.array(fitur_gabungan)


# --- Konfigurasi Path ---
base_path = os.path.join('images')

kategori_kain = {
    'katun': ['crosshatched', 'woven', 'lined'],
    'wol': ['knitted', 'matted', 'flecked'],
    'sintesis': ['interlaced', 'frilly','pleated']
}

X = []
y = []

EXTENSIONS = ['*.jpg', '*.JPG', '*.jpeg', '*.JPEG', '*.png', '*.PNG']

print("Memulai ekstraksi fitur dari gambar...")

for label_kain, daftar_folder in kategori_kain.items():
    for nama_folder in daftar_folder:
        folder_path = os.path.join(base_path, nama_folder) 

        if not os.path.exists(folder_path):
            print(f"  [PERINGATAN] Folder tidak ditemukan: {folder_path}")
            continue

        gambar_ditemukan = 0
        for ext in EXTENSIONS:
            for image_path in glob.glob(os.path.join(folder_path, ext)):
                fitur_gabungan = ekstract_fitur_kain(image_path)
                if fitur_gabungan is not None:
                    X.append(fitur_gabungan)
                    y.append(label_kain)
                    gambar_ditemukan += 1

        print(f"  [{label_kain}] {nama_folder}: {gambar_ditemukan} gambar diproses")

X = np.array(X)
y = np.array(y)

print(f"\nSelesai! Total gambar yang berhasil diproses: {len(X)}")

# --- Training & Evaluasi ---
if len(X) > 0:

    # Normalisasi fitur
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    from sklearn.decomposition import PCA
    pca_prep = PCA(n_components=0.95)
    X_independent = pca_prep.fit_transform(X_scaled)

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X_independent, y, test_size=0.2, random_state=42, shuffle=True
    )

    # Training model
    model = GaussianNB()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    # --- Hasil Evaluasi ---
    print("\n" + "=" * 40)
    print("       HASIL EVALUASI MODEL GLCM")
    print("=" * 40)

    akurasi = accuracy_score(y_test, y_pred)
    print(f"Akurasi Keseluruhan : {akurasi * 100:.2f}%")
    print("-" * 40)
    print("Classification Report:")
    print(classification_report(y_test, y_pred))
    print("Confusion Matrix:")
    cm = confusion_matrix(y_test, y_pred)
    print(cm)

    # --- Visualisasi 1: Scatter Plot PCA (semua fitur) ---
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_test)

    plt.figure(figsize=(8, 6))
    sns.scatterplot(
        x=X_pca[:, 0],
        y=X_pca[:, 1],
        hue=y_test,
        palette="Set1",
        s=100,
        alpha=0.7
    )
    plt.title('Sebaran Fitur GLCM via PCA (2 Komponen Utama)')
    plt.xlabel(f'PC1 ({pca.explained_variance_ratio_[0]*100:.1f}% variansi)')
    plt.ylabel(f'PC2 ({pca.explained_variance_ratio_[1]*100:.1f}% variansi)')
    plt.legend(title='Jenis Kain')
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    # --- Visualisasi 2: Confusion Matrix Heatmap ---
    labels_unik = sorted(set(y_test))
    plt.figure(figsize=(6, 5))
    sns.heatmap(
        cm,
        annot=True,
        fmt='d',
        cmap='Blues',
        xticklabels=labels_unik,
        yticklabels=labels_unik
    )
    plt.title('Confusion Matrix')
    plt.xlabel('Prediksi')
    plt.ylabel('Aktual')
    plt.tight_layout()
    plt.show()
    
    # Menyimpan model agar bisa dipakai oleh web
    joblib.dump(scaler, 'scaler.pkl')
    joblib.dump(pca_prep, 'pca_prep.pkl') 
    joblib.dump(model, 'model.pkl')
    print("Model berhasil disimpan untuk Web!")

else:
    print("\n[ERROR] Tidak ada gambar yang berhasil diproses.")
    print("Periksa kembali path dataset dan struktur folder.")
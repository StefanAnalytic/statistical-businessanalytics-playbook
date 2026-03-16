"""
Griff 61: Image Basics (Feature Extraction & Pretrained Embeddings)

Was ist das und was macht das Skript?
Klassische Machine-Learning-Modelle (wie Random Forests) brauchen Tabellen mit Zahlen. 
Aber was machen wir mit unstrukturierten Daten wie Produktfotos? 
Wir können Bilder nicht einfach roh in ein Tabellen-Modell werfen.

Die Lösung: Pretrained Embeddings (Transfer Learning).
Wir nutzen ein massives Deep-Learning-Modell (hier MobileNetV2 von Google), das bereits 
auf Millionen von Bildern (ImageNet) trainiert wurde. Wir schneiden den "Klassifikations-Kopf" 
des Modells ab und nutzen nur den Rumpf. Wenn wir jetzt ein Produktfoto durch das Modell 
schicken, spuckt es keinen Namen aus (z.B. "Das ist ein Stuhl"), sondern einen dichten 
Vektor aus Zahlen (ein "Embedding" oder "Feature Vector"). 

Diese Zahlen fassen Formen, Kanten und Texturen des Bildes zusammen. Wir können diese 
Embeddings dann nutzen, um ähnliche Produkte zu finden (Visual Search) oder sie als 
neue Features in unser Tabellen-Modell (z.B. zur Preisvorhersage) einzubauen.

Business Case in diesem Skript:
Wir extrahieren die visuellen Features von Produktbildern. Da wir für den Olist-Datensatz 
keine echten Bilder haben, generieren wir simulierte "Bild-Tensoren" für 3 Kategorien 
(Möbel, Elektronik, Kleidung). Wir schicken sie durch MobileNetV2, reduzieren die 
hochdimensionalen Embeddings mit PCA auf 2 Dimensionen und visualisieren, wie das 
Modell die Produkte rein anhand ihrer visuellen Muster gruppiert.

WICHTIG: Benötigt TensorFlow -> pip install tensorflow
"""

import os
import ssl
ssl._create_default_https_context = ssl._create_unverified_context
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.decomposition import PCA

try:
    import tensorflow as tf
    from tensorflow.keras.applications import MobileNetV2
    from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
except ImportError:
    print("FEHLER: Die Bibliothek 'tensorflow' fehlt. Bitte ausführen: pip install tensorflow")
    exit()

# -------------------------------------------------------------------
# 1. Setup & Pfade
# -------------------------------------------------------------------
GRAPHIC_DIR = "/Users/stefanbechthold/statistik_business_analytics/Grafiken/14_Computer_Vision"
GRAPHIC_PATH = os.path.join(GRAPHIC_DIR, "54_image_embeddings_pca.png")

os.makedirs(GRAPHIC_DIR, exist_ok=True)

def generate_dummy_images(num_images_per_class=50):
    """
    Simuliert Bild-Tensoren im Format (Batch, Height, Width, Channels).
    MobileNetV2 erwartet (224, 224, 3). Wir fügen künstliche Muster ein, 
    damit das Modell etwas zu "sehen" hat.
    """
    images = []
    labels = []
    
    # 1. Kategorie: "Möbel" (Eher dunkle, erdige Töne / Rauschen)
    for _ in range(num_images_per_class):
        img = np.random.normal(loc=100, scale=30, size=(224, 224, 3))
        images.append(img)
        labels.append('Möbel')
        
    # 2. Kategorie: "Elektronik" (Eher helle, metallische Töne / Rauschen)
    for _ in range(num_images_per_class):
        img = np.random.normal(loc=180, scale=20, size=(224, 224, 3))
        images.append(img)
        labels.append('Elektronik')
        
    # 3. Kategorie: "Kleidung" (Bunt, hohes Rauschen)
    for _ in range(num_images_per_class):
        img = np.random.normal(loc=150, scale=60, size=(224, 224, 3))
        images.append(img)
        labels.append('Kleidung')
        
    # Clipping auf valide Pixelwerte [0, 255]
    images = np.clip(images, 0, 255).astype(np.float32)
    return np.array(images), np.array(labels)

def main():
    print("🚀 Starte Griff 54: Image Feature Extraction (Pretrained Embeddings)...")
    
    # -------------------------------------------------------------------
    # 2. Modell laden & Bilder vorbereiten
    # -------------------------------------------------------------------
    # Wir laden MobileNetV2. 'include_top=False' ist der wichtigste Befehl!
    # Er entfernt den Klassifikator. Das Modell gibt nun rohe Embeddings zurück.
    # 'pooling=avg' wandelt die 2D-Feature-Maps am Ende in einen flachen 1D-Vektor um.
    print("Lade vortrainiertes MobileNetV2 (ohne Classification-Top)...")
    base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(224, 224, 3), pooling='avg')
    
    # Simuliere 150 Produktbilder (50 pro Kategorie)
    print("Generiere/Lade Produktbilder...")
    X_images, y_labels = generate_dummy_images(num_images_per_class=50)
    
    # Bilder für das spezifische Netz vorverarbeiten (Skalierung, Normalisierung)
    X_preprocessed = preprocess_input(X_images)

    # -------------------------------------------------------------------
    # 3. Feature Extraction (Embeddings erzeugen)
    # -------------------------------------------------------------------
    print("Extrahiere visuelle Features (Forward Pass durch das Deep Learning Modell)...")
    # Das Modell spuckt pro Bild einen Vektor mit 1280 Zahlen aus.
    embeddings = base_model.predict(X_preprocessed, batch_size=32)
    
    print(f"Erfolgreich extrahiert. Embedding-Matrix Shape: {embeddings.shape} (Bilder x Features)")

    # -------------------------------------------------------------------
    # 4. Dimensionalitätsreduktion (PCA) für Visualisierung
    # -------------------------------------------------------------------
    # Wir können keine 1280 Dimensionen plotten. PCA komprimiert die wichtigsten 
    # Informationen in 2 Dimensionen (X und Y Achse).
    pca = PCA(n_components=2, random_state=42)
    embeddings_2d = pca.fit_transform(embeddings)
    
    # In einen DataFrame packen
    df_plot = pd.DataFrame({
        'PCA_1': embeddings_2d[:, 0],
        'PCA_2': embeddings_2d[:, 1],
        'Kategorie': y_labels
    })

    # -------------------------------------------------------------------
    # 5. Visualisierung im Dark Mode Design
    # -------------------------------------------------------------------
    plt.style.use('dark_background')
    sns.set_theme(style="darkgrid", rc={
        "axes.facecolor": "#121212", 
        "figure.facecolor": "#121212", 
        "text.color": "white", 
        "axes.labelcolor": "white", 
        "xtick.color": "white", 
        "ytick.color": "white",
        "grid.color": "#2c3e50"
    })
    
    fig, ax1 = plt.subplots(figsize=(12, 8), facecolor='#121212')

    # Scatterplot der komprimierten Bild-Embeddings
    sns.scatterplot(
        x='PCA_1', y='PCA_2', 
        hue='Kategorie', 
        data=df_plot, 
        palette=['#e74c3c', '#3498db', '#2ecc71'], 
        s=100, alpha=0.8, edgecolor='white', ax=ax1
    )
    
    ax1.set_title("Computer Vision: Visualisierung der Bild-Embeddings (PCA-komprimiert)", 
                  fontsize=14, fontweight='bold', color='white', pad=15)
    ax1.set_xlabel(f"Hauptkomponente 1 (PCA 1) - Erklärt {pca.explained_variance_ratio_[0]*100:.1f}% Varianz")
    ax1.set_ylabel(f"Hauptkomponente 2 (PCA 2) - Erklärt {pca.explained_variance_ratio_[1]*100:.1f}% Varianz")
    
    legend = ax1.legend(title='Produkt-Kategorie', loc='upper right', facecolor='#1e1e1e', edgecolor='#333333')
    plt.setp(legend.get_title(), color='white')
    plt.setp(legend.get_texts(), color='white')

    # Info-Box
    info_text = (
        "Die Magie des Transfer Learnings:\n"
        "---------------------------------\n"
        "Obwohl das Modell diese spezifischen Dummy-Bilder noch nie\n"
        "gesehen hat und wir es nicht neu trainiert haben, erkennt es\n"
        "visuelle Muster! Die 1280 Zahlen (Embeddings) gruppieren sich\n"
        "sofort in logische Cluster (Möbel, Elektronik, Kleidung).\n\n"
        "Business Action:\n"
        "Diese Vektoren können wir nun nutzen für:\n"
        "1. Ähnliche Produkte empfehlen ('Kunden kauften auch...').\n"
        "2. An ein Tabellen-Modell (z.B. XGBoost) anflanschen,\n"
        "   um Conversion-Rates anhand der Foto-Qualität vorherzusagen."
    )
    props = dict(boxstyle='round,pad=1', facecolor='#1e1e1e', alpha=0.9, edgecolor='#f1c40f', linewidth=1.5)
    ax1.text(0.03, 0.03, info_text, transform=ax1.transAxes, fontsize=11,
             verticalalignment='bottom', bbox=props, color='white')

    plt.suptitle("Image Basics: Feature Extraction mit Deep Learning (MobileNetV2)", 
                 fontsize=18, fontweight='bold', color='white', y=1.02)
    plt.tight_layout()
    
    # Speichern der Grafik
    plt.savefig(GRAPHIC_PATH, dpi=300, bbox_inches='tight', facecolor=fig.get_facecolor())
    plt.close()
    
    print(f"✅ Grafik erfolgreich gespeichert unter:\n{GRAPHIC_PATH}")

if __name__ == "__main__":
    main()
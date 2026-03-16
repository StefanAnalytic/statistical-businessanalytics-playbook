"""
Griff 26: Feature-Reduktion bei hoher Dimensionalität (PCA & t-SNE)

Was ist das und was macht das Skript?
Oft haben wir im Business Datensätze mit hunderten Spalten (Features). Zum Beispiel Kunden 
mit 50 verschiedenen Verhaltensmetriken oder Produkte mit 70 verschiedenen Kategorien. 
Problem 1: Wir können 70 Dimensionen nicht visualisieren (Menschen sehen nur 3D!).
Problem 2: Machine-Learning-Modelle werden extrem langsam und neigen zum Overfitting 
(der sogenannte "Fluch der Dimensionalität").

Die Lösung ist "Dimensionality Reduction" (Dimensionsreduktion). Wir komprimieren die 
Daten auf z.B. 2 Dimensionen, behalten aber so viel Information wie möglich bei!
1. PCA (Principal Component Analysis): Sucht mathematisch nach den "Hauptachsen", entlang 
   derer die Daten am stärksten streuen. Linear, extrem schnell, super um Rauschen zu filtern.
2. t-SNE (t-Distributed Stochastic Neighbor Embedding): Ein komplexer, nicht-linearer 
   Algorithmus. Er versucht, Datenpunkte, die im 70D-Raum Nachbarn waren, auch im 2D-Raum 
   als Nachbarn darzustellen. Perfekt, um versteckte Cluster (Gruppen) visuell zu finden!

Business Case in diesem Skript:
Wir nehmen unsere Produkte und bauen einen hochdimensionalen Datensatz: Gewicht, Volumen, 
Preis, Versandkosten UND 70 Dummy-Variablen für jede einzelne Produktkategorie. 
Das sind fast 80 Dimensionen! Wir schicken diese durch PCA und t-SNE, komprimieren sie 
auf 2 Dimensionen (X und Y) und färben sie nach "Preisklasse" ein, um zu sehen, 
ob die Algorithmen die Business-Logik (Teure vs. Billige Produkte) automatisch erkennen.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

# -------------------------------------------------------------------
# 1. Setup & Pfade
# -------------------------------------------------------------------
# Wir wechseln in den Ordner "09_Feature_Engineering_und_Reduktion"
ITEMS_PATH = "/Users/stefanbechthold/statistik_business_analytics/data/olist_order_items_dataset.csv"
PRODUCTS_PATH = "/Users/stefanbechthold/statistik_business_analytics/data/olist_products_dataset.csv"

GRAPHIC_DIR = "/Users/stefanbechthold/statistik_business_analytics/Grafiken/09_Feature_Engineering_und_Reduktion"
GRAPHIC_PATH = os.path.join(GRAPHIC_DIR, "26_feature_reduktion_pca_tsne.png")

os.makedirs(GRAPHIC_DIR, exist_ok=True)

def main():
    print("🚀 Starte Griff 26: Dimensionsreduktion (PCA & t-SNE)...")
    
    # -------------------------------------------------------------------
    # 2. Daten laden & High-Dimensional Feature Engineering
    # -------------------------------------------------------------------
    print("Lade Artikel- und Produktdaten...")
    df_items = pd.read_csv(ITEMS_PATH)
    df_products = pd.read_csv(PRODUCTS_PATH)
    
    # Aggregation: Da ein Produkt mehrfach verkauft werden kann, nehmen wir die Durchschnittswerte
    prod_stats = df_items.groupby('product_id').agg({
        'price': 'mean',
        'freight_value': 'mean'
    }).reset_index()
    
    # Mergen mit physischen Produktmerkmalen
    df_merged = pd.merge(prod_stats, df_products, on='product_id', how='inner')
    
    # Feature: Volumen in cm³
    df_merged['volume_cm3'] = df_merged['product_length_cm'] * df_merged['product_width_cm'] * df_merged['product_height_cm']
    
    # Bereinigung: Wir droppen NaNs und begrenzen Ausreißer
    df_clean = df_merged.dropna(subset=['price', 'freight_value', 'product_weight_g', 'volume_cm3', 'product_category_name']).copy()
    df_clean = df_clean[(df_clean['price'] < 2000) & (df_clean['product_weight_g'] < 30000)]
    
    # Um t-SNE rechentechnisch in Sekunden (statt Stunden) auszuführen, 
    # ziehen wir ein repräsentatives Sample von 3.000 Produkten.
    df_sample = df_clean.sample(n=3000, random_state=42)
    
    # 70+ Dimensionen erzeugen (One-Hot-Encoding der Kategorien)
    df_dummies = pd.get_dummies(df_sample['product_category_name'], prefix='cat', drop_first=True)
    
    # Das ist unser hochdimensionaler X-Datensatz (ca. 75 Spalten!)
    X = pd.concat([df_sample[['price', 'freight_value', 'product_weight_g', 'volume_cm3']], df_dummies], axis=1)
    
    # Für die Visualisierung bauen wir ein "Business-Label" (Target), 
    # damit wir die grauen Punkte nachher sinnvoll einfärben können.
    # Wir teilen die Produkte in 3 Preisklassen (Günstig, Mittel, Teuer) ein.
    df_sample['Preisklasse'] = pd.qcut(df_sample['price'], q=3, labels=['1. Günstig', '2. Mittel', '3. Teuer'])
    y_label = df_sample['Preisklasse']
    
    print(f"Datensatz bereit: {X.shape[0]} Produkte mit {X.shape[1]} Dimensionen (Features).")

    # -------------------------------------------------------------------
    # 3. Daten Skalieren (Kritisch für PCA & t-SNE!)
    # -------------------------------------------------------------------
    print("\nSkaliere die Daten (StandardScaler)...")
    # Ohne Skalierung würde das Merkmal "Volumen" (Zahlen in den Zehntausenden) 
    # die PCA komplett dominieren, während "Preis" (Hunderte) ignoriert wird.
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # -------------------------------------------------------------------
    # 4. Dimensionality Reduction durchführen
    # -------------------------------------------------------------------
    print("Führe PCA (Lineare Kompression) durch...")
    # Wir sagen der PCA: Gib uns exakt 2 Dimensionen zurück
    pca = PCA(n_components=2, random_state=42)
    X_pca = pca.fit_transform(X_scaled)
    
    # Wieviel Prozent der ursprünglichen "Wahrheit" (Varianz) steckt in diesen 2 Dimensionen?
    var_explained = pca.explained_variance_ratio_.sum() * 100
    print(f"-> PCA 2D erklärt {var_explained:.1f}% der gesamten Varianz des {X.shape[1]}D-Raums.")

    print("Führe t-SNE (Nicht-Lineare Kompression) durch...")
    # t-SNE braucht etwas Länger. perplexity ist ein wichtiger Hyperparameter 
    # (wie viele Nachbarn beachtet werden sollen, 30-50 ist Standard).
    tsne = TSNE(n_components=2, perplexity=40, random_state=42, n_iter=1000)
    X_tsne = tsne.fit_transform(X_scaled)
    print("-> t-SNE abgeschlossen.")

    # -------------------------------------------------------------------
    # 5. Wunderschöne Visualisierung (2-teiliges Dashboard)
    # -------------------------------------------------------------------
    print("\nGeneriere Visualisierungs-Grid...")
    sns.set_theme(style="whitegrid")
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))

    # Wir packen die komprimierten Koordinaten für Seaborn in einen DataFrame
    plot_df = pd.DataFrame({
        'PCA_1': X_pca[:, 0],
        'PCA_2': X_pca[:, 1],
        'tSNE_1': X_tsne[:, 0],
        'tSNE_2': X_tsne[:, 1],
        'Preisklasse': y_label
    })

    # Farbpalette für die Preisklassen
    palette = {'1. Günstig': '#3498db', '2. Mittel': '#f1c40f', '3. Teuer': '#e74c3c'}

    # --- Plot 1: PCA ---
    sns.scatterplot(data=plot_df, x='PCA_1', y='PCA_2', hue='Preisklasse', 
                    palette=palette, alpha=0.7, s=30, ax=axes[0], edgecolor='w', linewidth=0.5)
    
    axes[0].set_title(f"1. PCA (Principal Component Analysis)\nErklärte Varianz: {var_explained:.1f}%", fontsize=14, fontweight="bold")
    axes[0].set_xlabel("Hauptkomponente 1 (Viel Gewicht/Volumen)")
    axes[0].set_ylabel("Hauptkomponente 2 (Viel Preis/Versand)")
    
    # Info-Box für PCA
    pca_info = (
        "PCA komprimiert linear.\n"
        "Man sieht einen klaren Gradienten:\n"
        "Günstige Produkte (Blau) links,\n"
        "Teure (Rot) streuen nach rechts.\n"
        "Sehr schnell, aber Cluster\n"
        "überlappen oft stark."
    )
    props = dict(boxstyle='round,pad=0.8', facecolor='white', alpha=0.9, edgecolor='gray', linewidth=1.5)
    axes[0].text(0.03, 0.97, pca_info, transform=axes[0].transAxes, fontsize=11,
                 verticalalignment='top', horizontalalignment='left', bbox=props, color='#2c3e50')

    # --- Plot 2: t-SNE ---
    sns.scatterplot(data=plot_df, x='tSNE_1', y='tSNE_2', hue='Preisklasse', 
                    palette=palette, alpha=0.7, s=30, ax=axes[1], edgecolor='w', linewidth=0.5)
    
    axes[1].set_title("2. t-SNE (t-Distributed Stochastic Neighbor Embedding)", fontsize=14, fontweight="bold")
    axes[1].set_xlabel("t-SNE Dimension 1")
    axes[1].set_ylabel("t-SNE Dimension 2")
    
    # Info-Box für t-SNE
    tsne_info = (
        "t-SNE entwirrt komplexe Daten.\n"
        "Es bilden sich isolierte 'Inseln'\n"
        "basierend auf Produktkategorien\n"
        "und Preiseigenschaften.\n"
        "Hervorragend, um verborgene\n"
        "Business-Cluster aufzudecken!"
    )
    axes[1].text(0.03, 0.97, tsne_info, transform=axes[1].transAxes, fontsize=11,
                 verticalalignment='top', horizontalalignment='left', bbox=props, color='#2c3e50')

    # Layout optimieren und speichern
    plt.suptitle("Von 75 Dimensionen auf 2: High-Dimensional Data Visualisierung", 
                 fontsize=18, fontweight="bold", y=1.05)
    plt.tight_layout()
    sns.despine()
    
    plt.savefig(GRAPHIC_PATH, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✅ Grafik erfolgreich gespeichert unter:\n{GRAPHIC_PATH}")

if __name__ == "__main__":
    main()
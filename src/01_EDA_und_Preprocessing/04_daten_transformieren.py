"""
Griff 4: Datentransformation für stabile Modelle (Log, Box-Cox, Yeo-Johnson, Robust Scaling)

Was macht dieses Skript?
Lädt reale E-Commerce-Preise und wendet Transformationen an, um extrem schiefe 
Datenverteilungen (rechtsschief) in eine modellfreundlichere (symmetrische/normalverteilte) 
Form zu zwingen. Es visualisiert den Vorher-Nachher-Effekt in einem professionellen 2x2 Grid.

Warum an dieser Stelle?
Machine-Learning-Modelle (wie lineare Regression) performen katastrophal, wenn 99% der 
Kunden 50€ ausgeben und 1% der Kunden 5000€. Transformationen entschärfen diese Ausreißer 
mathematisch, ohne die Datenpunkte löschen zu müssen.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.preprocessing import RobustScaler

# -------------------------------------------------------------------
# 1. Setup & Pfade (Deiner Struktur folgend)
# -------------------------------------------------------------------
DATA_PATH = "/Users/stefanbechthold/statistik_business_analytics/data/olist_order_items_dataset.csv"
GRAPHIC_DIR = "/Users/stefanbechthold/statistik_business_analytics/Grafiken/01_EDA_und_Preprocessing"
GRAPHIC_PATH = os.path.join(GRAPHIC_DIR, "04_transformationen_vergleich.png")

# Sicherstellen, dass der Zielordner existiert
os.makedirs(GRAPHIC_DIR, exist_ok=True)

def main():
    print("🚀 Starte Griff 4: Datentransformation...")
    
    # -------------------------------------------------------------------
    # 2. Daten laden & bereinigen
    # -------------------------------------------------------------------
    print(f"Lade Daten von: {DATA_PATH}")
    df = pd.read_csv(DATA_PATH)
    
    # Für Transformationen wie Box-Cox müssen die Werte zwingend echt größer 0 sein.
    # Wir isolieren die Spalte 'price' und filtern fehlerhafte 0€-Einträge.
    prices = df.loc[df['price'] > 0, 'price'].copy()
    print(f"Analysiere {len(prices)} gültige Preis-Datenpunkte.")

    # -------------------------------------------------------------------
    # 3. Mathematische Transformationen anwenden
    # -------------------------------------------------------------------
    print("Berechne Transformationen...")
    
    # A. Log-Transformation (np.log1p macht log(1+x), robuster bei kleinen Werten)
    # Zieht extrem hohe Werte stark nach unten und staucht die Skala.
    prices_log = np.log1p(prices)
    
    # B. Box-Cox-Transformation
    # Sucht algorithmisch den perfekten Exponenten (Lambda), um die Daten 
    # so nah wie möglich an eine Normalverteilung (Glockenkurve) zu bringen.
    prices_boxcox, best_lambda = stats.boxcox(prices)
    print(f"➔ Box-Cox hat optimales Lambda = {best_lambda:.4f} gefunden.")
    
    # C. Robust Scaler (Skalierung, keine Formänderung!)
    # Verändert NICHT die schiefe Form, sondern zentriert den Median auf 0 
    # und skaliert anhand des Interquartilsabstands (Ignoriert Ausreißer).
    # Sklearn erwartet ein 2D-Array, daher das .values.reshape(-1, 1).
    scaler = RobustScaler()
    prices_robust = scaler.fit_transform(prices.values.reshape(-1, 1)).flatten()

    # -------------------------------------------------------------------
    # 4. Wunderschöne Visualisierung erstellen
    # -------------------------------------------------------------------
    print("Generiere 2x2 Visualisierungs-Grid...")
    
    # Stil-Einstellungen für einen professionellen, cleanen Look
    sns.set_theme(style="whitegrid")
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    
    # Plot 1: Originale Daten (Oben Links)
    sns.histplot(prices, bins=60, kde=True, color="#34495e", ax=axes[0, 0])
    axes[0, 0].set_title("1. Originale Verteilung (Stark Rechtsschief)", fontsize=14, fontweight="bold")
    axes[0, 0].set_xlabel("Preis in BRL (begrenzt auf 0-1000 für Sichtbarkeit)")
    axes[0, 0].set_xlim(0, 1000) # Extremwerte abschneiden, sonst sieht man nur einen Balken
    
    # Plot 2: Log-Transformation (Oben Rechts)
    sns.histplot(prices_log, bins=60, kde=True, color="#3498db", ax=axes[0, 1])
    axes[0, 1].set_title("2. Logarithmische Transformation (log1p)", fontsize=14, fontweight="bold")
    axes[0, 1].set_xlabel("Log(Preis)")
    
    # Plot 3: Box-Cox Transformation (Unten Links)
    sns.histplot(prices_boxcox, bins=60, kde=True, color="#2ecc71", ax=axes[1, 0])
    axes[1, 0].set_title(f"3. Box-Cox Transformation (λ = {best_lambda:.2f})", fontsize=14, fontweight="bold")
    axes[1, 0].set_xlabel("Transformierter Preis")
    
    # Plot 4: Robust Scaler (Unten Rechts)
    sns.histplot(prices_robust, bins=60, kde=True, color="#9b59b6", ax=axes[1, 1])
    axes[1, 1].set_title("4. Robust Skalierung (Median=0)", fontsize=14, fontweight="bold")
    axes[1, 1].set_xlabel("Skalierter Preis (begrenzt auf -2 bis 6)")
    axes[1, 1].set_xlim(-2, 6) # Auch hier für die Sichtbarkeit clippen

    # Layout optimieren und speichern
    plt.suptitle("Effekt von Datentransformationen auf schiefe Business-Metriken", fontsize=18, fontweight="bold", y=1.03)
    plt.tight_layout()
    sns.despine()
    
    plt.savefig(GRAPHIC_PATH, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✅ Grafik erfolgreich gespeichert unter:\n{GRAPHIC_PATH}")

if __name__ == "__main__":
    main()
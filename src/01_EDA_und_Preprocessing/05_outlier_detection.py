"""
Griff 5: Gehört ein Wert zu einem Ausreißer? (Outlier Detection)

Was ist das und was macht das Skript?
Dieses Skript analysiert E-Commerce-Daten (Preis und Versandkosten) und identifiziert 
Ausreißer (Outlier). Ein Ausreißer ist ein Datenpunkt, der extrem vom Rest der Daten abweicht 
(z. B. ein Produkt für 10.000 €, während der Rest 50 € kostet). 

Wir wenden 4 verschiedene statistische und Machine-Learning-Methoden an:
1. IQR (Interquartile Range): Der Klassiker. Berechnet die Boxplot-Grenzen. Robust gegen Schiefe.
2. Z-Score: Misst die Standardabweichungen vom Mittelwert. (Achtung: Anfällig für extreme Werte!).
3. Robust Z-Score (MAD): Wie Z-Score, nutzt aber Median statt Mittelwert. Sehr robust!
4. Isolation Forest: Ein Machine-Learning-Algorithmus, der Anomalien in mehreren Dimensionen 
   (hier: Preis UND Versandkosten gleichzeitig) isoliert.

Wofür ist das im Business gut?
Ausreißer zerstören Durchschnittswerte, ruinieren Machine-Learning-Modelle und verfälschen 
Forecasts. Gleichzeitig können Ausreißer aber auch hochinteressant sein: Betrugsversuche 
(Fraud Detection), Systemfehler oder extrem wertvolle VIP-Kunden. Du musst sie erkennen, 
um bewusst entscheiden zu können, ob du sie löschst, transformierst oder genauer untersuchst.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.ensemble import IsolationForest

# -------------------------------------------------------------------
# 1. Setup & Pfade
# -------------------------------------------------------------------
# Ab hier wechseln wir in den Ordner "02_Verteilungen_und_Sampling", 
# da Outlier eng mit der Verteilung der Daten zusammenhängen.
DATA_PATH = "/Users/stefanbechthold/statistik_business_analytics/data/olist_order_items_dataset.csv"
GRAPHIC_DIR = "/Users/stefanbechthold/statistik_business_analytics/Grafiken/02_Verteilungen_und_Sampling"
GRAPHIC_PATH = os.path.join(GRAPHIC_DIR, "05_ausreisser_erkennung.png")

os.makedirs(GRAPHIC_DIR, exist_ok=True)

def main():
    print("🚀 Starte Griff 5: Ausreißer-Erkennung (Outlier Detection)...")
    
    # -------------------------------------------------------------------
    # 2. Daten laden & vorbereiten
    # -------------------------------------------------------------------
    print(f"Lade Daten von: {DATA_PATH}")
    df = pd.read_csv(DATA_PATH)
    
    # Wir nehmen uns ein Subset von 10.000 zufälligen Zeilen, damit der Isolation Forest 
    # blitzschnell rechnet und der Scatterplot nicht zu einem einzigen schwarzen Fleck mutiert (Overplotting).
    df_sample = df[['price', 'freight_value']].dropna().sample(n=10000, random_state=42).copy()
    
    # Wir untersuchen primär den 'price' für die univariaten (eindimensionalen) Methoden.
    prices = df_sample['price'].values
    
    print(f"Analysiere {len(prices)} Datenpunkte auf Ausreißer...")

    # -------------------------------------------------------------------
    # 3. Methode 1: IQR (Interquartile Range)
    # -------------------------------------------------------------------
    # Berechne das 25% Quantil (Q1) und das 75% Quantil (Q3)
    q1 = np.percentile(prices, 25)
    q3 = np.percentile(prices, 75)
    iqr = q3 - q1
    
    # Definition der Grenzen: Alles was kleiner als Q1 - 1.5*IQR oder größer als Q3 + 1.5*IQR ist, gilt als Ausreißer.
    lower_bound_iqr = q1 - 1.5 * iqr
    upper_bound_iqr = q3 + 1.5 * iqr
    
    # Wir erstellen eine True/False Maske: True, wenn es ein Ausreißer ist.
    outliers_iqr = (prices < lower_bound_iqr) | (prices > upper_bound_iqr)

    # -------------------------------------------------------------------
    # 4. Methode 2: Klassischer Z-Score
    # -------------------------------------------------------------------
    # Der Z-Score sagt: "Wie viele Standardabweichungen ist der Wert vom Durchschnitt entfernt?"
    # Ein Z-Score > 3 oder < -3 gilt klassischerweise als Ausreißer (deckt 99.7% der Normalverteilung ab).
    z_scores = stats.zscore(prices)
    outliers_z = np.abs(z_scores) > 3

    # -------------------------------------------------------------------
    # 5. Methode 3: Robust Z-Score (Median Absolute Deviation - MAD)
    # -------------------------------------------------------------------
    # Mittelwert und Standardabweichung werden selbst stark von Ausreißern verzerrt.
    # Daher nutzen wir den Median (robust) und die absolute Abweichung vom Median (MAD).
    median_price = np.median(prices)
    mad = np.median(np.abs(prices - median_price))
    
    # Konstante 0.6745 bringt den MAD auf die gleiche Skala wie die Standardabweichung bei Normalverteilungen.
    modified_z_scores = 0.6745 * (prices - median_price) / mad
    outliers_robust_z = np.abs(modified_z_scores) > 3.5 # Threshold oft bei 3.5 gesetzt

    # -------------------------------------------------------------------
    # 6. Methode 4: Isolation Forest (Machine Learning, Multivariat)
    # -------------------------------------------------------------------
    # Isolation Forest schaut sich Preis UND Versandkosten (freight_value) gleichzeitig an!
    # Er baut Entscheidungsbäume. Punkte, die sehr schnell "isoliert" (abgetrennt) werden können, sind Ausreißer.
    iso_forest = IsolationForest(contamination=0.05, random_state=42) # Wir nehmen an, 5% der Daten sind Ausreißer
    
    # Fit und Predict auf beiden Spalten. Gibt 1 für "Normal" und -1 für "Ausreißer" zurück.
    iso_preds = iso_forest.fit_predict(df_sample[['price', 'freight_value']])
    outliers_iso = iso_preds == -1

    # -------------------------------------------------------------------
    # 7. Ergebnisse zusammenfassen & Visualisieren
    # -------------------------------------------------------------------
    print("\nGefundene Ausreißer pro Methode:")
    print(f"1. IQR:            {outliers_iqr.sum()} Ausreißer")
    print(f"2. Z-Score:        {outliers_z.sum()} Ausreißer")
    print(f"3. Robust Z-Score: {outliers_robust_z.sum()} Ausreißer")
    print(f"4. Iso-Forest:     {outliers_iso.sum()} Ausreißer (Multivariat)")

    print("\nGeneriere 2x2 Visualisierungs-Grid...")
    sns.set_theme(style="whitegrid")
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    # Hilfsfunktion für einheitliche Scatterplots
    def plot_outliers(ax, mask, title):
        # Normale Punkte (Blau/Grau, halbtransparent)
        ax.scatter(df_sample['price'][~mask], df_sample['freight_value'][~mask], 
                   c='#34495e', alpha=0.4, s=15, label='Normal')
        # Ausreißer (Leuchtend Rot)
        ax.scatter(df_sample['price'][mask], df_sample['freight_value'][mask], 
                   c='#e74c3c', alpha=0.8, s=25, label='Ausreißer')
        ax.set_title(title, fontsize=14, fontweight="bold")
        ax.set_xlabel("Preis (BRL)")
        ax.set_ylabel("Versandkosten (BRL)")
        ax.legend()

    # Plot 1: IQR (Schneidet ab einem festen Preis gnadenlos vertikal ab)
    plot_outliers(axes[0, 0], outliers_iqr, f"1. IQR Methode ({outliers_iqr.sum()} Ausreißer)")

    # Plot 2: Z-Score (Ist durch die extreme Rechtsschiefe verfälscht, findet weniger)
    plot_outliers(axes[0, 1], outliers_z, f"2. Klassischer Z-Score ({outliers_z.sum()} Ausreißer)")

    # Plot 3: Robust Z / MAD (Sehr aggressiv bei schiefen Daten, findet sehr viele)
    plot_outliers(axes[1, 0], outliers_robust_z, f"3. Robuster Z-Score / MAD ({outliers_robust_z.sum()} Ausreißer)")

    # Plot 4: Isolation Forest (Erkennt auch Punkte als Ausreißer, die z.B. einen normalen Preis, aber extrem hohe Versandkosten haben!)
    plot_outliers(axes[1, 1], outliers_iso, f"4. Isolation Forest (Multivariat, {outliers_iso.sum()} Ausreißer)")

    plt.suptitle("Vergleich von Outlier-Detection-Methoden (Preis vs. Versandkosten)", 
                 fontsize=18, fontweight="bold", y=1.02)
    plt.tight_layout()
    sns.despine()
    
    plt.savefig(GRAPHIC_PATH, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✅ Grafik erfolgreich gespeichert unter:\n{GRAPHIC_PATH}")

if __name__ == "__main__":
    main()
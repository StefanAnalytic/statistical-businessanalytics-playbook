"""
Griff 51: Deep Anomaly Detection (Isolation Forest)

Was ist das und was macht das Skript?
In großen Datenmengen (wie Millionen von Transaktionen) gibt es immer wieder Datenpunkte, 
die völlig aus dem Muster fallen: Betrugsversuche (Fraud), Systemfehler, B2B-Großeinkäufe 
in einem B2C-Shop oder logistische Extremfälle. 

Klassische univariate Ausreißer-Erkennung (z.B. "Alles über 1000€ ist ein Ausreißer") 
scheitert hier, da Anomalien oft in der Kombination der Features liegen (Multivariat).
Beispiel: Eine 50€ Bestellung ist normal. Eine Bestellung mit 50kg Gewicht ist normal. 
Aber eine 50€ Bestellung mit 50kg Gewicht und 24 Ratenzahlungen ist hochgradig verdächtig!

Der Isolation Forest ist ein unüberwachter Machine-Learning-Algorithmus, der perfekt 
für solche Fälle ist. Er baut zufällige Entscheidungsbäume. Die Kern-Idee: 
Normale Datenpunkte sind tief in den Bäumen versteckt. Anomalien sind so "anders", 
dass sie schon nach sehr wenigen Splits (früh im Baum) isoliert werden.

Business Case in diesem Skript:
Wir suchen nach den 1% extremsten, anomalen Bestellungen im Olist-Marktplatz. 
Dafür füttern wir den Algorithmus mit einer Kombination aus Gesamtpreis, Frachtkosten, 
Produktgewicht, Anzahl der Items und der gewählten Ratenzahlungen.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

# -------------------------------------------------------------------
# 1. Setup & Pfade
# -------------------------------------------------------------------
ITEMS_PATH = "/Users/stefanbechthold/statistik_business_analytics/data/olist_order_items_dataset.csv"
PAYMENTS_PATH = "/Users/stefanbechthold/statistik_business_analytics/data/olist_order_payments_dataset.csv"
PRODUCTS_PATH = "/Users/stefanbechthold/statistik_business_analytics/data/olist_products_dataset.csv"

GRAPHIC_DIR = "/Users/stefanbechthold/statistik_business_analytics/Grafiken/13_Monitoring_MLOps_und_Governance"
GRAPHIC_PATH = os.path.join(GRAPHIC_DIR, "51_isolation_forest_anomaly_detection.png")

os.makedirs(GRAPHIC_DIR, exist_ok=True)

def main():
    print("🚀 Starte Griff 51: Multivariate Anomaly Detection (Isolation Forest)...")
    
    # -------------------------------------------------------------------
    # 2. Daten laden & Multivariate Features bauen
    # -------------------------------------------------------------------
    df_items = pd.read_csv(ITEMS_PATH)
    df_payments = pd.read_csv(PAYMENTS_PATH)
    df_products = pd.read_csv(PRODUCTS_PATH)
    
    # Gewicht an die Items mergen
    df_items = pd.merge(df_items, df_products[['product_id', 'product_weight_g']], on='product_id', how='left')
    
    # Auf Bestell-Ebene aggregieren
    order_features = df_items.groupby('order_id').agg({
        'price': 'sum',                     # Gesamtpreis
        'freight_value': 'sum',             # Gesamte Frachtkosten
        'product_weight_g': 'sum',          # Gesamtgewicht
        'order_item_id': 'max'              # Anzahl der Items
    }).reset_index()
    order_features.rename(columns={'order_item_id': 'item_count'}, inplace=True)
    
    # Ratenzahlungen (Installments) von den Payments holen
    # Wenn ein Kunde mit mehreren Methoden zahlt, nehmen wir das Maximum der Raten
    payment_features = df_payments.groupby('order_id').agg({
        'payment_installments': 'max'
    }).reset_index()
    
    # Master-Tabelle für das Training
    df_merged = pd.merge(order_features, payment_features, on='order_id', how='inner')
    df_clean = df_merged.dropna().copy()
    
    # Sample ziehen, falls der Datensatz zu groß für eine schnelle Demo ist
    df_sample = df_clean.sample(n=30000, random_state=42).reset_index(drop=True)
    
    features = ['price', 'freight_value', 'product_weight_g', 'item_count', 'payment_installments']
    X = df_sample[features]

    # -------------------------------------------------------------------
    # 3. Isolation Forest trainieren
    # -------------------------------------------------------------------
    # Skalierung ist für Isolation Forest nicht zwingend, hilft aber oft bei der Stabilität
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Wir nehmen an, dass ca. 1% der Bestellungen echte Anomalien sind (Contamination)
    iso_forest = IsolationForest(n_estimators=100, contamination=0.01, random_state=42)
    
    # Fit & Predict: Gibt 1 für normale Datenpunkte und -1 für Anomalien zurück
    df_sample['anomaly_label'] = iso_forest.fit_predict(X_scaled)
    
    # Anomaly Score berechnen (Je negativer der Wert, desto anomaler der Punkt)
    df_sample['anomaly_score'] = iso_forest.decision_function(X_scaled)
    
    # Trennung für die Visualisierung
    normal_data = df_sample[df_sample['anomaly_label'] == 1]
    anomalies = df_sample[df_sample['anomaly_label'] == -1]
    
    print(f"Gefundene Anomalien: {len(anomalies)} von {len(df_sample)} Bestellungen.")

    # -------------------------------------------------------------------
    # 4. Visualisierung im Dark Mode Design
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
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 7), facecolor='#121212')

    # --- Plot 1: Scatterplot (Preis vs. Gewicht) mit markierten Anomalien ---
    # Wir nutzen Log-Skalen, da Ausreißer sonst den Plot komplett unlesbar machen
    ax1.scatter(normal_data['product_weight_g'], normal_data['price'], 
                color='#3498db', alpha=0.3, s=15, label='Normale Bestellungen (99%)', edgecolor='none')
    
    ax1.scatter(anomalies['product_weight_g'], anomalies['price'], 
                color='#e74c3c', alpha=0.9, s=50, marker='X', label='Isolierte Anomalien (1%)', edgecolor='black')
    
    ax1.set_xscale('log')
    ax1.set_yscale('log')
    ax1.set_title("1. Multivariate Ausreißer (Log-Skala)", fontsize=14, fontweight='bold', color='white')
    ax1.set_xlabel("Gesamtgewicht in Gramm (Log)")
    ax1.set_ylabel("Gesamtpreis in BRL (Log)")
    
    legend1 = ax1.legend(loc='lower right', facecolor='#1e1e1e', edgecolor='#333333')
    plt.setp(legend1.get_texts(), color='white')

    # Info-Box Plot 1
    info_1 = (
        "Multivariate Anomalien:\n"
        "-----------------------\n"
        "Die roten X-Markierungen liegen oft an den extremen Rändern.\n"
        "Aber Achtung: Einige Anomalien liegen mitten im Cluster!\n"
        "Warum? Weil sie z.B. völlig normale Preise haben, aber in\n"
        "24 Raten bezahlt wurden oder aus 15 Einzel-Items bestehen.\n"
        "Der Algorithmus 'sieht' in 5 Dimensionen gleichzeitig."
    )
    props1 = dict(boxstyle='round,pad=1', facecolor='#1e1e1e', alpha=0.9, edgecolor='#e74c3c', linewidth=1.5)
    ax1.text(0.05, 0.95, info_1, transform=ax1.transAxes, fontsize=11,
             verticalalignment='top', bbox=props1, color='white')

    # --- Plot 2: Verteilung der Anomaly Scores ---
    sns.histplot(normal_data['anomaly_score'], bins=50, color='#3498db', alpha=0.7, 
                 label='Normale Daten (> 0)', kde=True, ax=ax2, edgecolor='none')
    sns.histplot(anomalies['anomaly_score'], bins=20, color='#e74c3c', alpha=0.9, 
                 label='Anomalien (< 0)', kde=False, ax=ax2, edgecolor='black')
    
    # Null-Linie (Threshold des Algorithmus)
    ax2.axvline(0, color='white', linestyle='--', linewidth=2, label='Decision Threshold (0)')
    
    ax2.set_title("2. Isolation Forest Score Verteilung", fontsize=14, fontweight='bold', color='white')
    ax2.set_xlabel("Anomaly Score (Negativ = Isoliert = Anomalie)")
    ax2.set_ylabel("Anzahl der Bestellungen")
    
    legend2 = ax2.legend(loc='upper left', facecolor='#1e1e1e', edgecolor='#333333')
    plt.setp(legend2.get_texts(), color='white')

    # Info-Box Plot 2
    info_2 = (
        "Business Operations (Fraud & QA):\n"
        "---------------------------------\n"
        "Anstatt manuelle Regeln zu pflegen ('Wenn Preis > 1000...'),\n"
        "lassen wir den Isolation Forest jeden Tag die 1% extremsten\n"
        "Bestellungen (Score < 0) herausfiltern.\n\n"
        "Diese kleine, rote Liste geht dann zur manuellen Prüfung an\n"
        "das Fraud-Team oder den Logistik-Support (B2B Bestellungen? Systemfehler?)."
    )
    props2 = dict(boxstyle='round,pad=1', facecolor='#1e1e1e', alpha=0.9, edgecolor='#3498db', linewidth=1.5)
    ax2.text(0.05, 0.5, info_2, transform=ax2.transAxes, fontsize=11,
             verticalalignment='center', bbox=props2, color='white')

    plt.suptitle("MLOps Monitoring: Deep Anomaly Detection mit Isolation Forests", 
                 fontsize=18, fontweight='bold', color='white', y=1.05)
    plt.tight_layout()
    
    # Speichern der Grafik
    plt.savefig(GRAPHIC_PATH, dpi=300, bbox_inches='tight', facecolor=fig.get_facecolor())
    plt.close()
    
    print(f"✅ Grafik erfolgreich gespeichert unter:\n{GRAPHIC_PATH}")

if __name__ == "__main__":
    main()
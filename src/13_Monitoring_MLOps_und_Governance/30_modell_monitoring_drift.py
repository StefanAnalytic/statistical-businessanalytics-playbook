"""
Griff 30: Modell-Performance in Prod überwachen? (Monitoring: Drift Detection & Control Charts)

Was ist das und was macht das Skript?
Sobald ein Modell produktiv gesetzt wurde, beginnt sein schleichender Qualitätsverlust (Model Decay). 
Datenstrukturen ändern sich über die Zeit (Data Drift), oder der Zusammenhang zwischen Features 
und Zielvariable verschiebt sich (Concept Drift).

Dieses Skript implementiert ein Frühwarnsystem für Production-Pipelines:
1. Data Drift Detection (K-S Test): Ein statistischer Test (Kolmogorov-Smirnov), der prüft, 
   ob sich die Verteilung der aktuellen Eingangsdaten signifikant von den Trainingsdaten unterscheidet.
2. EWMA Control Charts (Exponentially Weighted Moving Average): Ein Kontrolldiagramm aus der 
   Qualitätssicherung, das gleitende Durchschnitte nutzt, um subtile Trends oder plötzliche 
   Ausreißer in der Modell-Performance (z.B. sinkende Accuracy) sofort zu erkennen.

Business-Nutzen:
Anstatt auf Kundenbeschwerden zu warten, erkennt das Monitoring automatisch, wann ein Modell 
neu trainiert (Retraining) oder die Datenpipeline überprüft werden muss.

Voraussetzung: 
Das Skript simuliert eine "Produktionsphase", in der sich die Datenqualität künstlich verschlechtert,
um die Drift-Erkennung zu demonstrieren.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import ks_2samp

# -------------------------------------------------------------------
# 1. Setup & Pfade
# -------------------------------------------------------------------
# Wir nutzen den Credit Card Fraud Datensatz als Basis für das Monitoring-Beispiel
DATA_PATH = "/Users/stefanbechthold/statistik_business_analytics/data/creditcard.csv"
GRAPHIC_DIR = "/Users/stefanbechthold/statistik_business_analytics/Grafiken/14_Monitoring"
GRAPHIC_PATH = os.path.join(GRAPHIC_DIR, "30_model_monitoring_drift.png")

os.makedirs(GRAPHIC_DIR, exist_ok=True)

def main():
    print("Starte Griff 30: Production Monitoring & Drift Detection...")

    # -------------------------------------------------------------------
    # 2. Daten laden & Simulation von Production-Drift
    # -------------------------------------------------------------------
    if not os.path.exists(DATA_PATH):
        print(f"Datei nicht gefunden: {DATA_PATH}. Erzeuge synthetische Monitoring-Daten...")
        # Fallback: Erzeuge Daten, falls CSV nicht lokal vorliegt
        train_data = np.random.normal(0, 1, 1000)
        prod_data = np.concatenate([np.random.normal(0, 1, 500), np.random.normal(0.5, 1.2, 500)])
    else:
        df = pd.read_csv(DATA_PATH)
        # Wir nehmen eine Spalte (z.B. 'V1') als Referenz
        train_data = df['V1'].iloc[:5000].values
        # Simulation: In der Produktion verschieben sich die Werte (Drift)
        prod_data = df['V1'].iloc[5000:10000].values * 1.5 + 0.5 

    # -------------------------------------------------------------------
    # 3. Drift Detection (Kolmogorov-Smirnov Test)
    # -------------------------------------------------------------------
    # Der K-S Test vergleicht zwei Verteilungen. 
    # H0: Beide Stichproben stammen aus der gleichen Verteilung.
    statistic, p_value = ks_2samp(train_data, prod_data)
    is_drift = p_value < 0.05

    # -------------------------------------------------------------------
    # 4. Performance Monitoring (EWMA Control Chart)
    # -------------------------------------------------------------------
    # Simulation von Modell-Accuracy über 100 Tage
    np.random.seed(42)
    days = np.arange(1, 101)
    # Basis-Accuracy schwankt um 0.95, sackt ab Tag 70 ab (Drift)
    accuracy = np.concatenate([
        np.random.normal(0.95, 0.01, 70),
        np.random.normal(0.90, 0.02, 30)
    ])
    
    df_monitor = pd.DataFrame({'Day': days, 'Accuracy': accuracy})
    # Berechnung des EWMA (Gleitender Durchschnitt mit Gewichtung auf aktuelle Werte)
    df_monitor['EWMA'] = df_monitor['Accuracy'].ewm(span=10).mean()
    # Kontrollgrenzen (3-Sigma)
    center_line = df_monitor['Accuracy'].iloc[:20].mean()
    std_dev = df_monitor['Accuracy'].iloc[:20].std()
    ucl = center_line + 3 * std_dev # Upper Control Limit
    lcl = center_line - 3 * std_dev # Lower Control Limit

    # -------------------------------------------------------------------
    # 5. Visualisierung im Dark Mode Design
    # -------------------------------------------------------------------
    plt.style.use('dark_background')
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 7), facecolor='#121212')
    
    # Plot 1: Verteilungs-Vergleich (Feature Drift)
    sns.kdeplot(train_data, ax=ax1, label='Training (Baseline)', color='#3498db', fill=True, alpha=0.3)
    sns.kdeplot(prod_data, ax=ax1, label='Production (Current)', color='#e74c3c', fill=True, alpha=0.3)
    ax1.set_title(f"Feature Drift Detection (K-S Test)\nDrift erkannt: {is_drift} (p={p_value:.4f})", 
                  fontsize=14, fontweight='bold', color='white')
    ax1.legend()

    # Plot 2: EWMA Control Chart (Model Performance Drift)
    ax2.plot(df_monitor['Day'], df_monitor['Accuracy'], color='#555555', alpha=0.5, label='Daily Accuracy')
    ax2.plot(df_monitor['Day'], df_monitor['EWMA'], color='#f1c40f', linewidth=2, label='EWMA (Trend)')
    
    # Kontrollgrenzen einzeichnen
    ax2.axhline(ucl, color='#e74c3c', linestyle='--', label='UCL (3σ)')
    ax2.axhline(lcl, color='#e74c3c', linestyle='--', label='LCL (3σ)')
    ax2.axhline(center_line, color='#2ecc71', linestyle='-', alpha=0.6, label='Baseline Mean')
    
    # Alarm-Markierung ab dem Punkt, an dem LCL unterschritten wird
    alarms = df_monitor[df_monitor['EWMA'] < lcl]
    if not alarms.empty:
        ax2.scatter(alarms['Day'], alarms['EWMA'], color='#e74c3c', s=50, zorder=5, label='Performance Alarm')

    ax2.set_title("Model Performance Monitoring (EWMA Chart)", fontsize=14, fontweight='bold', color='white')
    ax2.set_xlabel("Tage in Produktion")
    ax2.set_ylabel("Accuracy")
    ax2.legend(loc='lower left', fontsize=9)

    plt.suptitle("MLOps Monitoring: Frühwarnsystem für Data & Model Drift", fontsize=18, color='white', y=1.02)
    plt.tight_layout()
    
    # Speichern
    plt.savefig(GRAPHIC_PATH, dpi=300, bbox_inches='tight', facecolor=fig.get_facecolor())
    plt.close()
    
    print(f"✅ Monitoring-Dashboard erfolgreich gespeichert:\n{GRAPHIC_PATH}")

if __name__ == "__main__":
    main()
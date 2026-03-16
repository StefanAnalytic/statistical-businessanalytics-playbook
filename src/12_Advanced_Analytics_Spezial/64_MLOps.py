"""
Griff 59: MLOps & Retraining Policies (Model Registry, Drift & CI/CD Stubs)

Was ist das und was macht das Skript?
Ein Machine-Learning-Modell in Produktion ist wie ein Auto: Es verliert mit der Zeit 
an Wert (Performance). Wenn sich das Verhalten der Kunden, die Wirtschaft oder die 
Plattform ändert, passen die alten Muster, die das Modell gelernt hat, nicht mehr.
Das nennt man "Concept Drift" oder "Data Drift".

MLOps (Machine Learning Operations) löst dieses Problem durch automatisierte Lifecycles:
1. Monitoring: Wir überwachen die Modell-Performance (z.B. ROC-AUC) in Echtzeit.
2. Retraining Policy: Wir definieren klare Regeln. Z.B. "Wenn AUC < 0.75 fällt, 
   trainiere das Modell automatisch mit den Daten der letzten 30 Tage neu."
3. CI/CD & Model Registry: Das neu trainierte Modell wird versioniert (z.B. v1.2), 
   automatisch getestet und (wenn besser als das alte) nahtlos in Produktion ausgetauscht.

Business Case in diesem Skript:
Wir simulieren den Lebenszyklus unseres "Delivery Delay Prediction" Modells über 12 Monate. 
Anfangs ist das Modell top (AUC = 0.85). Aber durch Veränderungen in der Logistik 
(Data Drift bei Frachtkosten und Lieferwegen) sinkt die Performance. Wir implementieren 
einen Retraining-Trigger, der das Modell automatisch "heilt", sobald es unter die 
Schmerzgrenze fällt. Gleichzeitig visualisieren wir den Data Drift, der das Problem verursacht hat.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# -------------------------------------------------------------------
# 1. Setup & Pfade
# -------------------------------------------------------------------
# (Hier nutzen wir simulierte Drift-Daten, da Olist-Daten für echten 
# Langzeit-Drift über Jahre hinweg zu kurz sind)
GRAPHIC_DIR = "/Users/stefanbechthold/statistik_business_analytics/Grafiken/13_Monitoring_MLOps_und_Governance"
GRAPHIC_PATH = os.path.join(GRAPHIC_DIR, "59_mlops_retraining_policy.png")

os.makedirs(GRAPHIC_DIR, exist_ok=True)

def simulate_production_lifecycle(months=12, initial_auc=0.85, drift_rate=0.015, threshold=0.75):
    """
    Simuliert die Performance eines Modells in Produktion über X Monate.
    Implementiert eine einfache Retraining-Policy.
    """
    np.random.seed(42)
    history = []
    retrain_events = []
    
    current_auc = initial_auc
    model_version = 1.0
    
    for month in range(1, months + 1):
        # 1. Natürliches Rauschen in der Performance hinzufügen
        noise = np.random.normal(0, 0.01)
        
        # 2. Concept Drift abziehen (Modell wird jeden Monat schlechter)
        current_auc = current_auc - drift_rate + noise
        
        record = {
            'Month': month,
            'AUC': current_auc,
            'Version': f"v{model_version:.1f}",
            'Action': 'Monitor'
        }
        
        # 3. Retraining Policy Trigger prüfen
        if current_auc < threshold:
            record['Action'] = 'Retrain Triggered'
            retrain_events.append(month)
            
            # CI/CD Stub: Modell wird mit neuen Daten trainiert und deployt
            model_version += 0.1
            
            # Nach dem Retraining springt die Performance wieder hoch (nicht ganz auf Start-Niveau)
            current_auc = initial_auc - np.random.uniform(0.01, 0.03) 
            
        history.append(record)
        
    return pd.DataFrame(history), retrain_events

def main():
    print("🚀 Starte Griff 59: MLOps Lifecycle & Retraining Policies...")
    
    # -------------------------------------------------------------------
    # 2. Lifecycle simulieren
    # -------------------------------------------------------------------
    # Schmerzgrenze für das Business: Das Modell darf nicht schlechter als 0.75 AUC werden
    alert_threshold = 0.75
    df_history, retrain_months = simulate_production_lifecycle(
        months=24, initial_auc=0.86, drift_rate=0.012, threshold=alert_threshold
    )

    # -------------------------------------------------------------------
    # 3. Data Drift simulieren (Warum wurde das Modell schlechter?)
    # -------------------------------------------------------------------
    # Wir simulieren eine Feature-Verteilung (z.B. 'freight_value') im Trainingsdatensatz (Monat 0)
    # und in Produktion (Monat 8), wo die Inflation/Spritpreise die Frachtkosten verändert haben.
    np.random.seed(10)
    train_data = np.random.gamma(shape=2.0, scale=15.0, size=5000) # Training (Alt)
    prod_data = np.random.gamma(shape=3.0, scale=18.0, size=5000)  # Produktion (Neu, drifted)

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

    # --- Plot 1: Model Performance Monitoring & Retraining ---
    ax1.plot(df_history['Month'], df_history['AUC'], color='#3498db', linewidth=3, marker='o', label='Live Model AUC')
    
    # Alarm-Grenze einzeichnen
    ax1.axhline(alert_threshold, color='#e74c3c', linestyle='--', linewidth=2, label=f'Retrain Trigger (< {alert_threshold})')
    
    # Retraining-Events markieren
    for month in retrain_months:
        ax1.axvline(month, color='#2ecc71', linestyle=':', linewidth=2)
        ax1.scatter(month, alert_threshold, color='#e74c3c', s=100, zorder=5) # Trigger Punkt
        
        # Versions-Nummer über den neuen Zyklus schreiben
        version = df_history[df_history['Month'] == month]['Version'].values[0]
        ax1.text(month + 0.5, 0.84, f"Deploy {version}", color='#2ecc71', fontweight='bold', fontsize=10)

    # Version 1.0 Label
    ax1.text(1, 0.85, "Deploy v1.0", color='#3498db', fontweight='bold', fontsize=10)

    ax1.set_title("1. CI/CD Pipeline: Automatisches Model Retraining", fontsize=14, fontweight='bold', color='white')
    ax1.set_xlabel("Produktionszeitraum (Monate)")
    ax1.set_ylabel("ROC-AUC Score (Validierung auf Live-Daten)")
    ax1.set_ylim(0.70, 0.90)
    ax1.set_xticks(np.arange(1, 25, 2))
    
    legend1 = ax1.legend(loc='lower left', facecolor='#1e1e1e', edgecolor='#333333')
    plt.setp(legend1.get_texts(), color='white')

    # Info-Box Plot 1
    info_1 = (
        "Die MLOps Retraining Policy:\n"
        "----------------------------\n"
        "Ein Modell verrottet in Produktion ('Model Decay').\n"
        "Anstatt manuell einzugreifen, überwacht das System den Score.\n"
        "Fällt der AUC unter 0.75 (Rote Linie), feuert ein Webhook.\n"
        "Die CI/CD Pipeline zieht frische Daten, trainiert v1.1,\n"
        "testet es und rollt es als Shadow-Deployment aus.\n"
        "Der Performance-Drop wird automatisch gestoppt!"
    )
    props1 = dict(boxstyle='round,pad=1', facecolor='#1e1e1e', alpha=0.9, edgecolor='#2ecc71', linewidth=1.5)
    ax1.text(0.35, 0.20, info_1, transform=ax1.transAxes, fontsize=11,
             verticalalignment='center', bbox=props1, color='white')

    # --- Plot 2: Data Drift (Die Ursache) ---
    sns.kdeplot(train_data, color='#3498db', fill=True, alpha=0.5, label='Training Data (Monat 0)', ax=ax2, edgecolor='none')
    sns.kdeplot(prod_data, color='#e74c3c', fill=True, alpha=0.5, label='Live Data (Monat 8 - Drifted)', ax=ax2, edgecolor='none')
    
    # Mediane zeigen die Verschiebung
    ax2.axvline(np.median(train_data), color='#3498db', linestyle='--', linewidth=2)
    ax2.axvline(np.median(prod_data), color='#e74c3c', linestyle='--', linewidth=2)

    ax2.set_title("2. Root Cause Analysis: Feature Data Drift", fontsize=14, fontweight='bold', color='white')
    ax2.set_xlabel("Feature Wert (z.B. 'freight_value' in BRL)")
    ax2.set_ylabel("Dichte (Relative Häufigkeit)")
    ax2.set_xlim(0, 150)
    
    legend2 = ax2.legend(loc='upper right', facecolor='#1e1e1e', edgecolor='#333333')
    plt.setp(legend2.get_texts(), color='white')

    # Info-Box Plot 2
    info_2 = (
        "Warum verrottet das Modell?\n"
        "---------------------------\n"
        "Das Modell (v1.0) hat gelernt, dass Pakete mit 30 BRL Frachtkosten\n"
        "oft pünktlich sind (Blaue Verteilung).\n\n"
        "8 Monate später haben sich die Logistik-Kosten durch Inflation\n"
        "verschoben (Rote Verteilung). 30 BRL ist jetzt ungewöhnlich billig!\n"
        "Das Modell wendet alte Logik auf eine neue Realität an und\n"
        "macht Fehler. Genau deshalb brauchen wir frische Trainingsdaten!"
    )
    props2 = dict(boxstyle='round,pad=1', facecolor='#1e1e1e', alpha=0.9, edgecolor='#e74c3c', linewidth=1.5)
    ax2.text(0.20, 0.45, info_2, transform=ax2.transAxes, fontsize=11,
             verticalalignment='center', bbox=props2, color='white')

    plt.suptitle("MLOps Lifecycle: Model Monitoring, Data Drift & CI/CD", 
                 fontsize=18, fontweight='bold', color='white', y=1.05)
    plt.tight_layout()
    
    # Speichern der Grafik
    plt.savefig(GRAPHIC_PATH, dpi=300, bbox_inches='tight', facecolor=fig.get_facecolor())
    plt.close()
    
    print(f"✅ Grafik erfolgreich gespeichert unter:\n{GRAPHIC_PATH}")

if __name__ == "__main__":
    main()
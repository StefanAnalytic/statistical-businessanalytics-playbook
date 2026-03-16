"""
Griff 24: Robuste A/B-Test-Pipeline? (Sequential Testing & Das "Peeking" Problem)

Was ist das und was macht das Skript?
Der größte Fehler im Business Analytics: Das "Peeking" (Vorzeitiges Ablesen).
Ein Produktmanager startet einen A/B-Test, schaut jeden Tag auf das Dashboard und stoppt 
den Test jubelnd an Tag 5, weil der p-Wert gerade unter 0.05 gerutscht ist. 
Das Problem: Wenn man oft genug testet, wird JEDER Test irgendwann durch reinen Zufall 
"signifikant" (False Positive / P-Hacking).

Dieses Skript simuliert einen A/A-Test (Variante A und Variante B sind exakt gleich!).
Wir nutzen die reale Basis-Conversion-Rate für Kreditkartenzahlungen aus dem Olist-Datensatz.
Wir simulieren 40 Tage Traffic und berechnen JEDEN TAG kumuliert den p-Wert.
Wir beweisen visuell, dass der Test zwischendurch "signifikant" wird, obwohl kein Effekt existiert!

Wofür ist das im Business gut?
Es zeigt, warum du entweder feste Laufzeiten (Sample Sizes aus Griff 23) brauchst, 
ohne vorher in die Daten zu schauen, ODER eine fortgeschrittene "Sequential Testing" Pipeline 
(z. B. mit korrigierten Alpha-Grenzen wie O'Brien-Fleming), die das Risiko für False Positives 
kontrolliert.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.stats.proportion import proportions_ztest

# -------------------------------------------------------------------
# 1. Setup & Pfade
# -------------------------------------------------------------------
# Wir wechseln in den Ordner "10_Kausalitaet_und_Experimente"
PAYMENTS_PATH = "/Users/stefanbechthold/statistik_business_analytics/data/olist_order_payments_dataset.csv"

GRAPHIC_DIR = "/Users/stefanbechthold/statistik_business_analytics/Grafiken/10_Kausalitaet_und_Experimente"
GRAPHIC_PATH = os.path.join(GRAPHIC_DIR, "24_ab_test_pipeline_peeking.png")

os.makedirs(GRAPHIC_DIR, exist_ok=True)

def main():
    print("🚀 Starte Griff 24: Robuste A/B-Test Pipeline (Sequential Testing)...")
    
    # -------------------------------------------------------------------
    # 2. Historische Baseline laden
    # -------------------------------------------------------------------
    print(f"Lade historische Zahlungsdaten von: {PAYMENTS_PATH}")
    df_payments = pd.read_csv(PAYMENTS_PATH)
    
    # Baseline Metrik: Wie viel Prozent der Bestellungen werden mit Kreditkarte bezahlt?
    total_orders = len(df_payments)
    cc_orders = len(df_payments[df_payments['payment_type'] == 'credit_card'])
    baseline_cr = cc_orders / total_orders
    
    print(f"Baseline Conversion Rate (Kreditkarte): {baseline_cr*100:.2f}%")

    # -------------------------------------------------------------------
    # 3. Den A/A-Test simulieren (Tag für Tag)
    # -------------------------------------------------------------------
    print("\nSimuliere A/B Test über 40 Tage (mit exakt gleicher Conversion Rate)...")
    
    np.random.seed(101) # Dieser Seed provoziert einen schönen "Peeking"-Moment
    
    days = 40
    daily_traffic_per_variant = 500
    
    # Arrays für die Ergebnisse (Kumuliert über die Zeit)
    p_values = []
    cr_A_history = []
    cr_B_history = []
    
    # Laufende Zähler
    cumulative_success_A = 0
    cumulative_success_B = 0
    cumulative_traffic_A = 0
    cumulative_traffic_B = 0
    
    for day in range(1, days + 1):
        # Wir simulieren den Traffic für diesen Tag. 
        # BEIDE Gruppen haben in der Realität exakt die gleiche Conversion (baseline_cr)!
        success_A = np.random.binomial(n=daily_traffic_per_variant, p=baseline_cr)
        success_B = np.random.binomial(n=daily_traffic_per_variant, p=baseline_cr)
        
        # Kumulieren (Wir addieren die Daten der Vortage dazu)
        cumulative_success_A += success_A
        cumulative_success_B += success_B
        cumulative_traffic_A += daily_traffic_per_variant
        cumulative_traffic_B += daily_traffic_per_variant
        
        # Aktuelle Conversion Rates berechnen
        current_cr_A = cumulative_success_A / cumulative_traffic_A
        current_cr_B = cumulative_success_B / cumulative_traffic_B
        cr_A_history.append(current_cr_A)
        cr_B_history.append(current_cr_B)
        
        # Z-Test für die kumulierten Daten DIESES Tages durchführen
        counts = np.array([cumulative_success_A, cumulative_success_B])
        nobs = np.array([cumulative_traffic_A, cumulative_traffic_B])
        stat, pval = proportions_ztest(count=counts, nobs=nobs, alternative='two-sided')
        
        p_values.append(pval)

    # -------------------------------------------------------------------
    # 4. Wunderschöne Visualisierung (2-teiliges Dashboard)
    # -------------------------------------------------------------------
    print("\nGeneriere Visualisierungs-Grid (Das Peeking Problem)...")
    sns.set_theme(style="whitegrid")
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    
    day_array = np.arange(1, days + 1)

    # --- Plot 1: Kumulierte Conversion Rates ---
    axes[0].plot(day_array, [cr * 100 for cr in cr_A_history], color='#3498db', linewidth=2, label='Variante A')
    axes[0].plot(day_array, [cr * 100 for cr in cr_B_history], color='#e67e22', linewidth=2, label='Variante B')
    axes[0].axhline(baseline_cr * 100, color='black', linestyle='--', alpha=0.5, label='Wahre Conversion (Baseline)')
    
    axes[0].set_title("1. Kumulierte Conversion Rate über Zeit", fontsize=14, fontweight="bold")
    axes[0].set_xlabel("Tag des Experiments")
    axes[0].set_ylabel("Conversion Rate (%)")
    axes[0].legend()
    
    # Y-Achse zoomen, um das Rauschen sichtbar zu machen
    axes[0].set_ylim((baseline_cr * 100) - 4, (baseline_cr * 100) + 4)

    # --- Plot 2: P-Value Fluktuation (Das Peeking-Problem) ---
    axes[1].plot(day_array, p_values, color='#9b59b6', linewidth=3, marker='o', markersize=4)
    
    # Die magische 0.05 Signifikanz-Grenze
    axes[1].axhline(0.05, color='#e74c3c', linestyle='-', linewidth=2, label='Signifikanz-Grenze (Alpha = 0.05)')
    
    # Wir heben den Bereich rot hervor, wo der p-Wert fälschlicherweise unter 0.05 fällt
    axes[1].fill_between(day_array, 0, 0.05, color='#e74c3c', alpha=0.15)
    
    # Suche den ersten Tag, an dem der p-Wert "signifikant" wird
    peeking_day = -1
    for i, p in enumerate(p_values):
        if p < 0.05:
            peeking_day = i + 1
            break
            
    if peeking_day != -1:
        axes[1].scatter(peeking_day, p_values[peeking_day-1], color='red', s=150, zorder=5, marker='X')
        axes[1].annotate(f'Falscher Jubel!\nManager stoppt an Tag {peeking_day}', 
                         xy=(peeking_day, p_values[peeking_day-1]), xytext=(peeking_day + 2, 0.15),
                         arrowprops=dict(facecolor='black', shrink=0.05, width=1.5, headwidth=8),
                         fontsize=11, fontweight='bold', color='#c0392b')

    axes[1].set_title("2. P-Wert Entwicklung (Warum 'Peeking' gefährlich ist)", fontsize=14, fontweight="bold")
    axes[1].set_xlabel("Tag des Experiments")
    axes[1].set_ylabel("P-Wert (Wahrscheinlichkeit, dass Unterschied Zufall ist)")
    axes[1].set_ylim(0, 1.0)
    axes[1].legend(loc='upper right')

    # Info-Box
    info_text = (
        "Das 'Optional Stopping' Problem:\n"
        "Obwohl beide Varianten exakt gleich performen,\n"
        "sorgt der pure statistische Zufall dafür, dass die\n"
        "Linie zwischendurch unter 0.05 rutscht. Wer hier\n"
        "den Test beendet, verbucht einen 'Erfolg', der\n"
        "in Wahrheit ein False Positive ist!"
    )
    props = dict(boxstyle='round,pad=0.8', facecolor='white', alpha=0.9, edgecolor='#e74c3c', linewidth=2)
    axes[1].text(0.05, 0.95, info_text, transform=axes[1].transAxes, fontsize=11,
                 verticalalignment='top', horizontalalignment='left', bbox=props, color='#2c3e50')

    # Layout optimieren und speichern
    plt.suptitle("A/B Testing Governance: Die Gefahr von Daily Peeking", 
                 fontsize=18, fontweight="bold", y=1.05)
    plt.tight_layout()
    sns.despine()
    
    plt.savefig(GRAPHIC_PATH, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✅ Grafik erfolgreich gespeichert unter:\n{GRAPHIC_PATH}")

if __name__ == "__main__":
    main()
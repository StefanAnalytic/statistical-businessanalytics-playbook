"""
Griff 13: Ändert sich die Conversion-Rate? (Proportion Test / z-test für Anteile)

Was ist das und was macht das Skript?
Oft wollen wir im Business nicht Durchschnittswerte (wie Umsatz) vergleichen, sondern 
Anteile oder Quoten (Proportions). Klassische Beispiele: 
- Conversion-Rate (Käufer vs. Nicht-Käufer)
- Churn-Rate (Kündiger vs. Bleiber)
- Late-Delivery-Rate (Zu spät geliefert vs. Pünktlich geliefert)

Dieses Skript nutzt den "2-Sample Z-Test for Proportions" (z-Test für Anteile).
Wir vergleichen die "Late Delivery Rate" (Verspätungsquote) zwischen den beiden 
größten brasilianischen Bundesstaaten: São Paulo (SP) und Rio de Janeiro (RJ).

Der Test prüft mathematisch: Ist der Unterschied in der Verspätungsquote zwischen 
SP und RJ reiner Zufall, oder gibt es ein signifikantes, strukturelles Logistik-Problem?

Wofür ist das im Business gut?
A/B-Testing für Conversion Rates! Wenn Variante A 5% Conversion hat (bei 1000 Besuchern) 
und Variante B 6% (bei 1200 Besuchern), sagt dir dieser Test exakt, ob Variante B 
wirklich besser ist, oder ob du das Ergebnis ignorieren solltest.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Statsmodels liefert uns den fertigen Z-Test für Anteile
from statsmodels.stats.proportion import proportions_ztest, proportion_confint

# -------------------------------------------------------------------
# 1. Setup & Pfade
# -------------------------------------------------------------------
ORDERS_PATH = "/Users/stefanbechthold/statistik_business_analytics/data/olist_orders_dataset.csv"
CUSTOMERS_PATH = "/Users/stefanbechthold/statistik_business_analytics/data/olist_customers_dataset.csv"

# Wir bleiben in Einheit 3 (Hypothesentests)
GRAPHIC_DIR = "/Users/stefanbechthold/statistik_business_analytics/Grafiken/03_Hypothesentests_und_Inferenz"
GRAPHIC_PATH = os.path.join(GRAPHIC_DIR, "13_proportion_ztest.png")

os.makedirs(GRAPHIC_DIR, exist_ok=True)

def main():
    print("🚀 Starte Griff 13: Z-Test für Anteile (Proportion Test)...")
    
    # -------------------------------------------------------------------
    # 2. Daten laden, verknüpfen & Feature Engineering
    # -------------------------------------------------------------------
    print("Lade Bestell- und Kundendaten...")
    df_orders = pd.read_csv(ORDERS_PATH)
    df_customers = pd.read_csv(CUSTOMERS_PATH)
    
    # Wir brauchen die Kunden-Tabelle, um zu wissen, aus welchem Bundesstaat die Bestellung kommt
    df_merged = pd.merge(df_orders, df_customers, on='customer_id', how='inner')
    
    print("Berechne 'Verspätet'-Metrik...")
    # Umwandlung der Text-Daten in echte Datum-Formate (erlaubt Fehler bei kaputten Daten mit coerce)
    df_merged['delivered_date'] = pd.to_datetime(df_merged['order_delivered_customer_date'], errors='coerce')
    df_merged['estimated_date'] = pd.to_datetime(df_merged['order_estimated_delivery_date'], errors='coerce')
    
    # Wir werfen Bestellungen raus, die noch nicht geliefert wurden (leeres Lieferdatum)
    df_valid = df_merged.dropna(subset=['delivered_date', 'estimated_date']).copy()
    
    # Eine Bestellung ist "Late" (Verspätet = True), wenn das Lieferdatum nach dem geschätzten Datum liegt
    df_valid['is_late'] = df_valid['delivered_date'] > df_valid['estimated_date']
    
    # Wir filtern auf die beiden Bundesstaaten: SP (São Paulo) und RJ (Rio de Janeiro)
    df_analysis = df_valid[df_valid['customer_state'].isin(['SP', 'RJ'])]

    # -------------------------------------------------------------------
    # 3. Aggregation: Erfolge (Verspätungen) und Versuche (Total) zählen
    # -------------------------------------------------------------------
    # Für den Test brauchen wir zwei Arrays:
    # 1. count: Wie oft trat das Ereignis ein? (Anzahl der Verspätungen)
    # 2. nobs (Number of observations): Wie viele Versuche gab es gesamt? (Alle Bestellungen)
    
    sp_data = df_analysis[df_analysis['customer_state'] == 'SP']
    rj_data = df_analysis[df_analysis['customer_state'] == 'RJ']
    
    count_sp_late = sp_data['is_late'].sum()
    total_sp = len(sp_data)
    
    count_rj_late = rj_data['is_late'].sum()
    total_rj = len(rj_data)
    
    # Die Arrays für den Statsmodels-Test aufbauen
    counts = np.array([count_sp_late, count_rj_late])
    nobs = np.array([total_sp, total_rj])
    
    rate_sp = count_sp_late / total_sp
    rate_rj = count_rj_late / total_rj
    
    print(f"\nSão Paulo (SP): {count_sp_late} verspätet von {total_sp} gesamt -> {rate_sp*100:.2f}%")
    print(f"Rio (RJ):       {count_rj_late} verspätet von {total_rj} gesamt -> {rate_rj*100:.2f}%")

    # -------------------------------------------------------------------
    # 4. Z-Test für Anteile (Proportions Z-Test)
    # -------------------------------------------------------------------
    print("\nFühre Z-Test für die Anteile durch...")
    # stat: Der Z-Score (wie viele Standardabweichungen sind die Quoten voneinander entfernt?)
    # pval: Der p-Wert (Ist der Unterschied signifikant?)
    stat, pval = proportions_ztest(count=counts, nobs=nobs, alternative='two-sided')
    
    # Wir berechnen zusätzlich die 95% Konfidenzintervalle für die Grafik
    ci_lower, ci_upper = proportion_confint(count=counts, nobs=nobs, alpha=0.05, method='normal')
    
    print(f"Z-Statistik: {stat:.4f}")
    print(f"P-Wert:      {pval:.4e}")
    
    alpha = 0.05
    if pval < alpha:
        interpretation = "Signifikanter Unterschied!\nLogistik in RJ ist strukturell schlechter."
        color_sig = "#e74c3c" # Rot, weil Verspätungen schlecht sind
    else:
        interpretation = "Kein signifikanter Unterschied.\nAbweichungen sind statistischer Zufall."
        color_sig = "#27ae60" # Grün

    # -------------------------------------------------------------------
    # 5. Wunderschöne Visualisierung
    # -------------------------------------------------------------------
    print("Generiere Visualisierung (Bar Chart mit Konfidenzintervallen)...")
    sns.set_theme(style="whitegrid")
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    states = ['São Paulo (SP)', 'Rio de Janeiro (RJ)']
    rates = [rate_sp, rate_rj]
    
    # Fehlerbalken berechnen (Distanz vom Anteil zum Rand des Konfidenzintervalls)
    yerr_lower = [rates[0] - ci_lower[0], rates[1] - ci_lower[1]]
    yerr_upper = [ci_upper[0] - rates[0], ci_upper[1] - rates[1]]
    yerr = [yerr_lower, yerr_upper]
    
    # Balkendiagramm zeichnen
    bars = ax.bar(states, rates, yerr=yerr, capsize=12, color=['#3498db', '#e67e22'], 
                  alpha=0.85, edgecolor='black', linewidth=1.5, width=0.5)
    
    ax.set_title("Vergleich der Verspätungsquote (Late Delivery Rate): SP vs. RJ", 
                 fontsize=16, fontweight="bold", pad=20)
    ax.set_ylabel("Anteil verspäteter Lieferungen", fontsize=12)
    
    # Y-Achse als Prozente formatieren (z.B. 10% statt 0.1)
    ax.set_ylim(0, max(ci_upper) * 1.3) # Etwas Luft nach oben für den Text
    vals = ax.get_yticks()
    ax.set_yticklabels(['{:,.1%}'.format(x) for x in vals])
    
    # Die genauen Prozentzahlen in die Balken schreiben
    for idx, bar in enumerate(bars):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height/2,
                f"{rates[idx]*100:.1f}%\n(n={nobs[idx]})",
                ha='center', va='center', color='white', fontweight='bold', fontsize=12)
                
    # Textbox mit den statistischen Testergebnissen
    stats_text = (
        f"Ergebnisse des Z-Tests (Anteile):\n"
        f"---------------------------------\n"
        f"Absoluter Unterschied: +{(rate_rj - rate_sp)*100:.2f} Prozentpunkte\n"
        f"Z-Statistik: {stat:.2f}\n"
        f"p-Wert: {pval:.5f}\n"
        f"Fazit: {interpretation}"
    )
    
    props = dict(boxstyle='round,pad=0.8', facecolor='white', alpha=0.9, edgecolor=color_sig, linewidth=2)
    ax.text(0.05, 0.95, stats_text, transform=ax.transAxes, fontsize=11,
            verticalalignment='top', bbox=props, color='#2c3e50')

    # Layout optimieren und speichern
    plt.tight_layout()
    sns.despine()
    
    plt.savefig(GRAPHIC_PATH, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✅ Grafik erfolgreich gespeichert unter:\n{GRAPHIC_PATH}")

if __name__ == "__main__":
    main()
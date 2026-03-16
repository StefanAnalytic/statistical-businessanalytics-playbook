"""
Griff 63: Privacy & Governance Basics (Differential Privacy)

Was ist das und was macht das Skript?
Wenn wir aggregierte Daten (z.B. Durchschnittsumsätze) an externe Partner weitergeben 
oder in öffentlichen Dashboards zeigen, denken viele: "Es sind ja nur Durchschnitte, 
da ist der Datenschutz gewahrt." Das ist ein gefährlicher Irrtum! 

Wenn ein "Whale" (ein extrem reicher Kunde) in eine Postleitzahl zieht und der 
Durchschnittsumsatz dort am nächsten Tag massiv ansteigt, kann man diesen Kunden 
durch einfache Differenzbildung re-identifizieren (Linkage Attack).

Die Lösung: "Differential Privacy" (DP). 
Wir mischen mathematisch kalibriertes Rauschen (Laplace Noise) in unsere Aggregate. 
Der Parameter "Epsilon" (ε) steuert das Privacy-Budget:
- Kleines Epsilon: Viel Rauschen, hohe Privatsphäre, ungenauere Daten (geringe Utility).
- Großes Epsilon: Wenig Rauschen, geringe Privatsphäre, exakte Daten (hohe Utility).

Business Case in diesem Skript:
Wir berechnen den durchschnittlichen Bestellwert pro Bundesstaat. Um zu verhindern, 
dass man einzelne B2B-Großkunden aus den Daten herausrechnen kann, legen wir einen 
Differential-Privacy-Filter darüber. Wir visualisieren den Trade-off: Wie stark 
verfälscht das Rauschen unsere Business-Reports bei verschiedenen Epsilon-Werten?
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# -------------------------------------------------------------------
# 1. Setup & Pfade
# -------------------------------------------------------------------
PAYMENTS_PATH = "/Users/stefanbechthold/statistik_business_analytics/data/olist_order_payments_dataset.csv"
ORDERS_PATH = "/Users/stefanbechthold/statistik_business_analytics/data/olist_orders_dataset.csv"
CUSTOMERS_PATH = "/Users/stefanbechthold/statistik_business_analytics/data/olist_customers_dataset.csv"

GRAPHIC_DIR = "/Users/stefanbechthold/statistik_business_analytics/Grafiken/13_Monitoring_MLOps_und_Governance"
GRAPHIC_PATH = os.path.join(GRAPHIC_DIR, "58_differential_privacy_tradeoff.png")

os.makedirs(GRAPHIC_DIR, exist_ok=True)

def laplace_mechanism(true_value, sensitivity, epsilon):
    """
    Fügt Laplace-Rauschen zu einem Wert hinzu, um Differential Privacy zu garantieren.
    """
    # Scale (b) = Sensitivität / Epsilon
    scale = sensitivity / epsilon
    noise = np.random.laplace(loc=0, scale=scale)
    return true_value + noise

def main():
    print("🚀 Starte Griff 58: Data Governance & Differential Privacy...")
    
    # -------------------------------------------------------------------
    # 2. Daten laden & Echte Aggregate berechnen
    # -------------------------------------------------------------------
    df_payments = pd.read_csv(PAYMENTS_PATH)
    df_orders = pd.read_csv(ORDERS_PATH)
    df_customers = pd.read_csv(CUSTOMERS_PATH)
    
    # Gesamtzahlung pro Bestellung
    order_values = df_payments.groupby('order_id')['payment_value'].sum().reset_index()
    
    # Mit Bundesstaaten mergen
    df_merged = pd.merge(order_values, df_orders[['order_id', 'customer_id']], on='order_id')
    df_merged = pd.merge(df_merged, df_customers[['customer_id', 'customer_state']], on='customer_id')
    
    # Ausreißer "clippen" (Sensitivität definieren)
    # Für DP müssen wir wissen, wie stark EIN EINZELNER Kunde den Durchschnitt maximal verändern kann.
    # Daher cappen wir extreme B2B-Bestellungen bei 1000 BRL, um die Sensitivität (Delta f) zu begrenzen.
    clip_max = 1000.0
    df_merged['payment_value_clipped'] = df_merged['payment_value'].clip(upper=clip_max)
    
    # Wir berechnen die wahren Durchschnitte für die Top 5 Staaten
    top_states = df_merged['customer_state'].value_counts().head(5).index
    df_top = df_merged[df_merged['customer_state'].isin(top_states)]
    
    state_stats = df_top.groupby('customer_state').agg(
        true_mean=('payment_value_clipped', 'mean'),
        count=('payment_value_clipped', 'count')
    ).reset_index()
    
    # -------------------------------------------------------------------
    # 3. Differential Privacy anwenden
    # -------------------------------------------------------------------
    epsilon_target = 0.5 # Unser gewähltes Privacy Budget für den Report
    
    # Sensitivität des Mittelwerts = Maximale Änderung durch einen Kunden / Anzahl der Kunden im Staat
    state_stats['sensitivity'] = clip_max / state_stats['count']
    
    # Rauschen hinzufügen
    np.random.seed(42) # Für Reproduzierbarkeit im Plot
    state_stats['dp_mean'] = state_stats.apply(
        lambda row: laplace_mechanism(row['true_mean'], row['sensitivity'], epsilon_target), axis=1
    )
    
    # Trade-off Kurve berechnen (Verschiedene Epsilons testen)
    epsilons = np.linspace(0.05, 2.0, 50)
    errors = []
    
    # Wir nehmen den kleinsten der Top-Staaten als Härtetest (weniger Daten = Rauschen wirkt stärker)
    test_state = state_stats.sort_values('count').iloc[0]
    
    for eps in epsilons:
        # 100 Simulationen pro Epsilon, um den durchschnittlichen Fehler zu glätten
        sim_errors = []
        for _ in range(100):
            noisy_val = laplace_mechanism(test_state['true_mean'], test_state['sensitivity'], eps)
            sim_errors.append(abs(test_state['true_mean'] - noisy_val))
        errors.append(np.mean(sim_errors))

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

    # --- Plot 1: Wahrer vs. DP-Durchschnitt (Barchart) ---
    x = np.arange(len(state_stats['customer_state']))
    width = 0.35

    ax1.bar(x - width/2, state_stats['true_mean'], width, label='Wahrer Umsatz (Unsicher)', color='#3498db', edgecolor='none')
    ax1.bar(x + width/2, state_stats['dp_mean'], width, label=f'DP Umsatz (ε={epsilon_target})', color='#f1c40f', edgecolor='none')

    ax1.set_title("1. Data Sharing: Dashboard-Werte mit Privacy-Rauschen", fontsize=14, fontweight='bold', color='white')
    ax1.set_xlabel("Bundesstaat")
    ax1.set_ylabel("Durchschnittlicher Bestellwert (BRL)")
    ax1.set_xticks(x)
    ax1.set_xticklabels(state_stats['customer_state'])
    
    legend1 = ax1.legend(loc='upper right', facecolor='#1e1e1e', edgecolor='#333333')
    plt.setp(legend1.get_texts(), color='white')

    # Info-Box Plot 1
    info_1 = (
        "Die Verschleierung (Laplace Noise):\n"
        "-----------------------------------\n"
        "Wir geben die gelben Werte an externe Partner.\n"
        "Sie weichen leicht von der Realität (Blau) ab.\n"
        "Dadurch wird das Gesamtmuster der Daten bewahrt,\n"
        "aber es ist mathematisch unmöglich zu beweisen,\n"
        "ob ein spezifischer Großkunde in der Berechnung war."
    )
    props1 = dict(boxstyle='round,pad=1', facecolor='#1e1e1e', alpha=0.9, edgecolor='#f1c40f', linewidth=1.5)
    ax1.text(0.05, 0.95, info_1, transform=ax1.transAxes, fontsize=11,
             verticalalignment='top', bbox=props1, color='white')

    # --- Plot 2: Der Privacy-Utility Tradeoff ---
    ax2.plot(epsilons, errors, color='#e74c3c', linewidth=3)
    
    # Markierung für unser gewähltes Epsilon
    ax2.axvline(epsilon_target, color='white', linestyle='--', linewidth=2, label=f'Unser Budget (ε={epsilon_target})')
    
    ax2.set_title(f"2. Privacy vs. Utility Trade-off (Beispiel-Staat: {test_state['customer_state']})", fontsize=14, fontweight='bold', color='white')
    ax2.set_xlabel("Privacy Budget Epsilon ε (→ Weniger Datenschutz, mehr Genauigkeit)")
    ax2.set_ylabel("Erwarteter Fehler im Report (BRL)")
    
    legend2 = ax2.legend(loc='upper right', facecolor='#1e1e1e', edgecolor='#333333')
    plt.setp(legend2.get_texts(), color='white')

    # Info-Box Plot 2
    info_2 = (
        "Compliance & Governance Check:\n"
        "------------------------------\n"
        "Links (ε nahe 0): Absolute Privatsphäre, aber das Rauschen\n"
        "ist so groß, dass unsere Business-Metrik unbrauchbar wird (hoher Fehler).\n\n"
        "Rechts (ε groß): Der Fehler geht gegen 0, aber wir riskieren\n"
        "die Re-Identifizierung von Kunden.\n"
        "Die rote Kurve zwingt Management und Data Science zu einer\n"
        "bewussten, quantifizierbaren Entscheidung über das Daten-Risiko."
    )
    props2 = dict(boxstyle='round,pad=1', facecolor='#1e1e1e', alpha=0.9, edgecolor='#e74c3c', linewidth=1.5)
    ax2.text(0.35, 0.45, info_2, transform=ax2.transAxes, fontsize=11,
             verticalalignment='center', bbox=props2, color='white')

    plt.suptitle("Data Governance: Differential Privacy für aggregierte Reports", 
                 fontsize=18, fontweight='bold', color='white', y=1.05)
    plt.tight_layout()
    
    # Speichern der Grafik
    plt.savefig(GRAPHIC_PATH, dpi=300, bbox_inches='tight', facecolor=fig.get_facecolor())
    plt.close()
    
    print(f"✅ Grafik erfolgreich gespeichert unter:\n{GRAPHIC_PATH}")

if __name__ == "__main__":
    main()
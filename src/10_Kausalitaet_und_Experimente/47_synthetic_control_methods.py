"""
Griff 47: Policy eval mit Vergleichseinheit? (Synthetic Control Methods)

Was ist das und was macht das Skript?
Wie messen wir den Effekt einer Maßnahme, wenn wir keinen A/B-Test machen konnten?
Beispiel: Eine neue Marketing-Kampagne oder Subvention wird in GANZ São Paulo (SP) ausgerollt. 
Wir können nicht SP mit Rio de Janeiro (RJ) vergleichen, weil die Staaten völlig 
unterschiedlich groß sind und andere Trends haben.

Die Synthetic Control Method (SCM) löst dieses Problem. Anstatt SP mit EINEM anderen 
Staat zu vergleichen, mischt der Algorithmus mehrere "Spender-Staaten" (Donor Pool) 
mathematisch so zusammen, dass sie das Verhalten von SP VOR der Kampagne perfekt imitieren.
Dieser "Synthetische SP" dient als Kontrollgruppe für die Zeit NACH der Kampagne.
Die Differenz zwischen dem echten SP und dem synthetischen SP ist der kausale Effekt!

Business Case in diesem Skript:
Wir simulieren einen Policy-Rollout: Am 1. Januar 2018 startet in São Paulo (SP) eine 
massive Werbekampagne. Wir nutzen andere große Staaten (RJ, MG, RS, PR, SC) als Donor Pool.
Wir trainieren SCM darauf, SP im Jahr 2017 zu kopieren, und beobachten dann die "Lücke" (Gap),
die sich ab 2018 zwischen Realität und Synthese auftut.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.optimize import minimize

# -------------------------------------------------------------------
# 1. Setup & Pfade
# -------------------------------------------------------------------
ORDERS_PATH = "/Users/stefanbechthold/statistik_business_analytics/data/olist_orders_dataset.csv"
PAYMENTS_PATH = "/Users/stefanbechthold/statistik_business_analytics/data/olist_order_payments_dataset.csv"
CUSTOMERS_PATH = "/Users/stefanbechthold/statistik_business_analytics/data/olist_customers_dataset.csv"

GRAPHIC_DIR = "/Users/stefanbechthold/statistik_business_analytics/Grafiken/10_Kausalitaet_und_Experimente"
GRAPHIC_PATH = os.path.join(GRAPHIC_DIR, "47_synthetic_control_method.png")

os.makedirs(GRAPHIC_DIR, exist_ok=True)

def main():
    print("🚀 Starte Griff 47: Synthetic Control Method (Policy Evaluation)...")
    
    # -------------------------------------------------------------------
    # 2. Daten laden & Aggregieren (Umsatz pro Staat & Monat)
    # -------------------------------------------------------------------
    df_orders = pd.read_csv(ORDERS_PATH)
    df_payments = pd.read_csv(PAYMENTS_PATH)
    df_customers = pd.read_csv(CUSTOMERS_PATH)
    
    df_merged = pd.merge(df_orders[['order_id', 'customer_id', 'order_purchase_timestamp']], 
                         df_payments[['order_id', 'payment_value']], on='order_id')
    df_merged = pd.merge(df_merged, df_customers[['customer_id', 'customer_state']], on='customer_id')
    
    df_merged['purchase_date'] = pd.to_datetime(df_merged['order_purchase_timestamp'])
    df_merged['month'] = df_merged['purchase_date'].dt.to_period('M')
    
    # Monatlicher Umsatz pro Bundesstaat
    state_revenue = df_merged.groupby(['month', 'customer_state'])['payment_value'].sum().reset_index()
    
    # Pivot-Tabelle: Datum als Index, Staaten als Spalten
    df_pivot = state_revenue.pivot(index='month', columns='customer_state', values='payment_value').fillna(0)
    
    # Filter auf relevanten Zeitraum (Jan 2017 bis Aug 2018)
    df_pivot = df_pivot.loc['2017-01':'2018-08']
    
    # Da SP massiv größer ist als der Rest, skalieren wir die Daten (z.B. Umsatz pro 100k Einwohner fiktiv)
    # Für das Skript skalieren wir SP einfach künstlich runter, damit eine konvexe Kombination (Summe Gewichte=1) möglich wird
    df_pivot['SP'] = df_pivot['SP'] * 0.3 
    
    # Wir fügen einen fiktiven Kampagnen-Effekt für SP ab Jan 2018 hinzu (+25%)
    # Das ist der Effekt, den unser Modell später finden muss!
    intervention_month = pd.Period('2018-01', freq='M')
    df_pivot.loc[intervention_month:, 'SP'] *= 1.25

    # -------------------------------------------------------------------
    # 3. Synthetic Control Setup (Optimierung)
    # -------------------------------------------------------------------
    target_state = 'SP'
    donor_pool = ['RJ', 'MG', 'RS', 'PR', 'SC']
    
    pre_period = df_pivot[df_pivot.index < intervention_month]
    post_period = df_pivot[df_pivot.index >= intervention_month]
    
    y_pre = pre_period[target_state].values
    X_pre = pre_period[donor_pool].values
    
    # Zielfunktion für die Optimierung (Mean Squared Error zwischen echtem und synthetischem SP)
    def objective(w, X, y):
        return np.mean((y - np.dot(X, w))**2)
    
    # Nebenbedingungen: Gewichte müssen sich zu 1 addieren (Konvexität)
    cons = ({'type': 'eq', 'fun': lambda w: np.sum(w) - 1})
    # Schranken: Keine negativen Gewichte (Extrapolation verbieten)
    bounds = [(0, 1) for _ in range(len(donor_pool))]
    # Startwerte (gleichverteilt)
    init_w = np.ones(len(donor_pool)) / len(donor_pool)
    
    # Optimierung ausführen (Finden der perfekten Spender-Mischung)
    res = minimize(objective, init_w, args=(X_pre, y_pre), method='SLSQP', bounds=bounds, constraints=cons)
    optimal_weights = res.x
    
    # Synthetischen Verlauf für VOR und NACH der Intervention berechnen
    synthetic_sp_pre = np.dot(X_pre, optimal_weights)
    synthetic_sp_post = np.dot(post_period[donor_pool].values, optimal_weights)
    synthetic_sp = np.concatenate([synthetic_sp_pre, synthetic_sp_post])
    
    df_pivot['Synthetic_SP'] = synthetic_sp
    df_pivot['Treatment_Effect'] = df_pivot['SP'] - df_pivot['Synthetic_SP']

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
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 12), facecolor='#121212', gridspec_kw={'height_ratios': [2, 1.5]})

    dates = df_pivot.index.astype(str)
    interv_idx = list(df_pivot.index).index(intervention_month)

    # --- Plot 1: Echter vs. Synthetischer Staat ---
    ax1.plot(dates, df_pivot['SP'], color='#3498db', linewidth=3, label='Echtes São Paulo (SP)')
    ax1.plot(dates, df_pivot['Synthetic_SP'], color='#e74c3c', linewidth=3, linestyle='--', label='Synthetisches São Paulo (Kontrollgruppe)')
    
    ax1.axvline(x=interv_idx, color='white', linestyle=':', linewidth=2, label='Start der Kampagne (Jan 2018)')
    ax1.fill_between(dates, df_pivot['SP'], df_pivot['Synthetic_SP'], 
                     where=df_pivot.index >= intervention_month, 
                     color='#2ecc71', alpha=0.3, label='Kausaler Lift (+25%)')
    
    ax1.set_title("1. Synthetic Control: Rekonstruktion eines Staates für A/B-Testing", fontsize=14, fontweight='bold', color='white')
    ax1.set_ylabel("Umsatz (Skaliert)")
    ax1.tick_params(axis='x', rotation=45)
    
    legend1 = ax1.legend(loc='upper left', facecolor='#1e1e1e', edgecolor='#333333')
    plt.setp(legend1.get_texts(), color='white')

    # Info-Box Plot 1
    weight_str = ", ".join([f"{state}: {w*100:.0f}%" for state, w in zip(donor_pool, optimal_weights) if w > 0.01])
    info_1 = (
        "Die Magie des Algorithmus:\n"
        f"Das synthetische SP besteht aus: {weight_str}.\n"
        "VOR der Kampagne (links) liegen die Linien fast exakt aufeinander.\n"
        "Der 'Parallel Trends' Check ist bestanden!\n"
        "NACH der Kampagne (rechts) reißt das echte SP nach oben aus.\n"
        "Das Synthetische SP zeigt uns, was ohne Kampagne passiert wäre."
    )
    props1 = dict(boxstyle='round,pad=1', facecolor='#1e1e1e', alpha=0.9, edgecolor='#3498db', linewidth=1.5)
    ax1.text(0.4, 0.15, info_1, transform=ax1.transAxes, fontsize=11,
             verticalalignment='center', bbox=props1, color='white')

    # --- Plot 2: Der Kausale Effekt (Gap Plot) ---
    # Zeigt nur die Differenz zwischen Echt und Synthetisch
    ax2.plot(dates, df_pivot['Treatment_Effect'], color='#2ecc71', linewidth=3, marker='o', label='Treatment Effekt (Gap)')
    ax2.axhline(0, color='white', linestyle='-', linewidth=1, alpha=0.5)
    ax2.axvline(x=interv_idx, color='white', linestyle=':', linewidth=2)
    
    ax2.set_title("2. Kausaler Netto-Effekt (Gap zwischen Realität und Counterfactual)", fontsize=14, fontweight='bold', color='white')
    ax2.set_ylabel("Zusätzlicher Umsatz (BRL)")
    ax2.tick_params(axis='x', rotation=45)

    # Info-Box Plot 2
    info_2 = (
        "Business Reporting:\n"
        "-------------------\n"
        "Wir berichten nicht einfach 'Umsatz ist gestiegen'.\n"
        "Wir können nun beziffern: Die Kampagne hat isoliert\n"
        "für einen Netto-Uplift von Y BRL pro Monat gesorgt."
    )
    props2 = dict(boxstyle='round,pad=1', facecolor='#1e1e1e', alpha=0.9, edgecolor='#2ecc71', linewidth=1.5)
    ax2.text(0.05, 0.85, info_2, transform=ax2.transAxes, fontsize=11,
             verticalalignment='top', bbox=props2, color='white')

    plt.suptitle("Causal Inference: Policy Evaluation mit Synthetic Controls", 
                 fontsize=18, fontweight='bold', color='white', y=1.02)
    plt.tight_layout()
    
    # Speichern der Grafik
    plt.savefig(GRAPHIC_PATH, dpi=300, bbox_inches='tight', facecolor=fig.get_facecolor())
    plt.close()
    
    print(f"✅ Grafik erfolgreich gespeichert unter:\n{GRAPHIC_PATH}")

if __name__ == "__main__":
    main()
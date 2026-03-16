"""
Griff 45: Experimente verlässlich designen? (Randomization & Balance Checks)

Was ist das und was macht das Skript?
Bevor wir die Ergebnisse eines A/B-Tests (Experiment) auswerten, müssen wir sicherstellen, 
dass das Experiment überhaupt verlässlich designt und durchgeführt wurde. 
Der größte Feind eines A/B-Tests ist eine kaputte Randomisierung (Selection Bias).

Wenn Gruppe A (Treatment) zufällig viel mehr "Premium-Kunden" enthält als Gruppe B (Control), 
wissen wir am Ende nicht, ob ein höherer Umsatz an unserer genialen Marketing-Idee lag 
oder einfach daran, dass Gruppe A von vornherein reicher war.

Die Lösung: Randomization & Balance Checks.
Wir prüfen VOR der Auswertung, ob alle wichtigen Metriken (Covariates) zwischen den Gruppen 
gleichmäßig verteilt sind. Der Goldstandard dafür ist die "Standardized Mean Difference" (SMD). 
Ein SMD nahe 0 bedeutet perfekte Balance. Ein SMD > 0.1 deutet auf ein Problem in der Randomisierung hin.

Business Case in diesem Skript:
Wir simulieren einen A/B-Test im Olist-Datensatz: 50% der Bestellungen erhalten fiktiv 
einen "Versandkosten-Gutschein" (Treatment). Anschließend führen wir einen strengen 
Balance Check für wichtige Variablen (Preis, Gewicht, Lieferzeit) durch, um zu beweisen, 
dass unsere Gruppen vor dem Start des Experiments statistische "Zwillinge" sind.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import ttest_ind

# -------------------------------------------------------------------
# 1. Setup & Pfade
# -------------------------------------------------------------------
ORDERS_PATH = "/Users/stefanbechthold/statistik_business_analytics/data/olist_orders_dataset.csv"
ITEMS_PATH = "/Users/stefanbechthold/statistik_business_analytics/data/olist_order_items_dataset.csv"
PRODUCTS_PATH = "/Users/stefanbechthold/statistik_business_analytics/data/olist_products_dataset.csv"

GRAPHIC_DIR = "/Users/stefanbechthold/statistik_business_analytics/Grafiken/10_Kausalitaet_und_Experimente"
GRAPHIC_PATH = os.path.join(GRAPHIC_DIR, "45_experiment_balance_checks.png")

os.makedirs(GRAPHIC_DIR, exist_ok=True)

def calculate_smd(df, covariate, treatment_col):
    """Berechnet die Standardized Mean Difference (SMD) für eine Variable."""
    treated = df[df[treatment_col] == 1][covariate]
    control = df[df[treatment_col] == 0][covariate]
    
    mean_t, var_t = treated.mean(), treated.var()
    mean_c, var_c = control.mean(), control.var()
    
    # Gepoolte Standardabweichung
    pooled_sd = np.sqrt((var_t + var_c) / 2)
    
    if pooled_sd == 0:
        return 0
    return (mean_t - mean_c) / pooled_sd

def main():
    print("🚀 Starte Griff 45: A/B-Test Design & Randomization Checks...")
    
    # -------------------------------------------------------------------
    # 2. Daten laden & Variablen für den Balance Check bauen
    # -------------------------------------------------------------------
    df_orders = pd.read_csv(ORDERS_PATH)
    df_items = pd.read_csv(ITEMS_PATH)
    df_products = pd.read_csv(PRODUCTS_PATH)
    
    # Lieferzeit berechnen
    df_orders['purchase_time'] = pd.to_datetime(df_orders['order_purchase_timestamp'], errors='coerce')
    df_orders['delivered_time'] = pd.to_datetime(df_orders['order_delivered_customer_date'], errors='coerce')
    df_orders['delivery_days'] = (df_orders['delivered_time'] - df_orders['purchase_time']).dt.total_seconds() / (24 * 3600)
    
    # Gewicht an Items mergen
    df_items = pd.merge(df_items, df_products[['product_id', 'product_weight_g']], on='product_id', how='left')
    
    # Auf Order-Ebene aggregieren (Unsere Covariates)
    order_features = df_items.groupby('order_id').agg({
        'price': 'sum',
        'freight_value': 'sum',
        'product_weight_g': 'sum'
    }).reset_index()
    
    df_merged = pd.merge(df_orders[['order_id', 'delivery_days']], order_features, on='order_id', how='inner')
    df_clean = df_merged.dropna().copy()
    
    # Ausreißer filtern für stabile Varianzen
    df_clean = df_clean[(df_clean['price'] < 1000) & (df_clean['delivery_days'] < 60) & (df_clean['delivery_days'] > 0)]
    
    # -------------------------------------------------------------------
    # 3. A/B-Test Zuweisung simulieren (Randomization)
    # -------------------------------------------------------------------
    np.random.seed(42)
    # 50/50 Zuweisung in Treatment (1) und Control (0)
    df_clean['Treatment'] = np.random.binomial(1, 0.5, size=len(df_clean))
    
    # Um zu zeigen, wie ein KAPUTTER Test aussieht, simulieren wir einen zweiten Bias-Test:
    # Hier geben wir teuren Bestellungen (Preis > 150) eine höhere Wahrscheinlichkeit für das Treatment!
    prob_bias = np.where(df_clean['price'] > 150, 0.7, 0.3)
    df_clean['Treatment_Biased'] = np.random.binomial(1, prob_bias)

    # -------------------------------------------------------------------
    # 4. Balance Metrics (SMD) berechnen
    # -------------------------------------------------------------------
    covariates = ['price', 'freight_value', 'product_weight_g', 'delivery_days']
    covariate_labels = ['Bestellwert (Preis)', 'Frachtkosten', 'Produktgewicht', 'Lieferzeit (Tage)']
    
    smd_random = []
    smd_biased = []
    p_values_random = []
    
    for cov in covariates:
        # SMD für saubere Randomisierung
        smd_random.append(abs(calculate_smd(df_clean, cov, 'Treatment')))
        
        # SMD für kaputte Randomisierung
        smd_biased.append(abs(calculate_smd(df_clean, cov, 'Treatment_Biased')))
        
        # Klassischer t-Test als Ergänzung
        t_stat, p_val = ttest_ind(df_clean[df_clean['Treatment'] == 1][cov], 
                                  df_clean[df_clean['Treatment'] == 0][cov], 
                                  equal_var=False)
        p_values_random.append(p_val)

    # -------------------------------------------------------------------
    # 5. Visualisierung im Dark Mode Design
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

    # --- Plot 1: Das "Love Plot" (SMD Vergleich) ---
    # Standard-Plot in der kausalen Inferenz, um Balance zu beweisen
    y_pos = np.arange(len(covariates))
    
    # Biased Randomization (Kaputt)
    ax1.scatter(smd_biased, y_pos, color='#e74c3c', s=150, marker='X', zorder=3, label='Kaputte Randomisierung (Biased)')
    # Saubere Randomisierung
    ax1.scatter(smd_random, y_pos, color='#2ecc71', s=150, marker='o', zorder=3, label='Saubere Randomisierung (A/B Test)')
    
    # Linien für bessere Lesbarkeit
    for i in range(len(covariates)):
        ax1.plot([0, max(max(smd_biased), 0.25)], [i, i], color='#333333', zorder=1)

    # Schwellenwert-Linien (0.1 gilt als Faustregel für "gut ausbalanciert")
    ax1.axvline(0.1, color='#f1c40f', linestyle='--', linewidth=2, zorder=2, label='Schwellenwert (0.1 SMD)')
    ax1.axvline(0.0, color='white', linestyle='-', linewidth=1, zorder=2)
    
    ax1.set_yticks(y_pos)
    ax1.set_yticklabels(covariate_labels, fontsize=12, fontweight='bold')
    ax1.set_title("1. Balance Check (Love Plot)\nSind die Gruppen vor Testbeginn identisch?", 
                  fontsize=14, fontweight='bold', color='white')
    ax1.set_xlabel("Standardized Mean Difference (SMD)\nNäher an 0 = Besser ausbalanciert")
    
    legend1 = ax1.legend(loc='lower right', facecolor='#1e1e1e', edgecolor='#333333')
    plt.setp(legend1.get_texts(), color='white')

    # --- Plot 2: Direkter Verteilungsvergleich einer kaputten Variable ---
    # Wir zeigen, warum die "Biased" Verteilung den A/B-Test ruinieren würde
    sns.kdeplot(df_clean[df_clean['Treatment_Biased'] == 1]['price'], 
                color='#e74c3c', fill=True, alpha=0.4, label='Treatment (Biased)', ax=ax2)
    sns.kdeplot(df_clean[df_clean['Treatment_Biased'] == 0]['price'], 
                color='#3498db', fill=True, alpha=0.4, label='Control (Biased)', ax=ax2)
    
    ax2.set_title("2. Gefahr: Selection Bias bei kaputter Randomisierung", 
                  fontsize=14, fontweight='bold', color='white')
    ax2.set_xlabel("Bestellwert (Preis in BRL)")
    ax2.set_ylabel("Dichte")
    ax2.set_xlim(0, 400)
    
    legend2 = ax2.legend(loc='upper right', facecolor='#1e1e1e', edgecolor='#333333')
    plt.setp(legend2.get_texts(), color='white')

    # Info-Box Plot 2
    info_text = (
        "Business Impact (A/B Testing):\n"
        "------------------------------\n"
        "Ein Love Plot (links) ist in Tech-Konzernen Pflicht, bevor\n"
        "ein Experiment überhaupt gestartet oder ausgewertet wird.\n\n"
        "Der Plot rechts zeigt das Horror-Szenario:\n"
        "Die rote Treatment-Gruppe hat von Haus aus viel mehr teure\n"
        "Bestellungen. Würden wir jetzt den Test auswerten, würden\n"
        "wir fälschlicherweise jubeln, dass unser Treatment den\n"
        "Umsatz gesteigert hat – dabei war die Gruppe einfach\n"
        "schon vorher reicher (Simpson's Paradox / Selection Bias)!"
    )
    props = dict(boxstyle='round,pad=1', facecolor='#1e1e1e', alpha=0.9, edgecolor='#e74c3c', linewidth=1.5)
    ax2.text(0.35, 0.4, info_text, transform=ax2.transAxes, fontsize=11,
             verticalalignment='center', bbox=props, color='white')

    plt.suptitle("Experiment Design: Randomization & Covariate Balance Checks", 
                 fontsize=18, fontweight='bold', color='white', y=1.05)
    plt.tight_layout()
    
    # Speichern der Grafik
    plt.savefig(GRAPHIC_PATH, dpi=300, bbox_inches='tight', facecolor=fig.get_facecolor())
    plt.close()
    
    print(f"✅ Grafik erfolgreich gespeichert unter:\n{GRAPHIC_PATH}")

if __name__ == "__main__":
    main()
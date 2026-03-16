"""
Griff 34: Verteilungen über Quantile untersuchen? (Quantile Regression)

Was ist das und was macht das Skript?
Die klassische lineare Regression (OLS) modelliert den bedingten Erwartungswert (Mittelwert) 
der Zielvariable. Bei stark verzerrten Daten (wie Umsätzen) oder wenn unabhängige Variablen 
unterschiedliche Auswirkungen auf verschiedene Kundensegmente haben, greift das zu kurz.

Die Quantilsregression (Quantile Regression) löst dieses Problem. Sie erlaubt es, den 
Effekt eines Features (z.B. Versandkosten) spezifisch für den Median (50. Quantil) oder 
die Ränder der Verteilung (z.B. 10. Quantil für Billigkäufer, 90. Quantil für "Whales") zu schätzen.

Vorteile:
1. Robustheit: Die Median-Regression (q=0.5) ist im Gegensatz zu OLS absolut robust gegen Ausreißer.
2. Heterogene Effekte: Sie deckt auf, ob ein Feature bei Premium-Kunden einen stärkeren 
   Hebel hat als bei Schnäppchenjägern.

Business Case in diesem Skript:
Wir analysieren den Zusammenhang zwischen den Versandkosten (freight_value) und dem 
gesamten Bestellwert (payment_value). Wir vergleichen OLS mit der Quantilsregression für 
q=0.10, q=0.50 und q=0.90 und visualisieren, wie sich die Koeffizienten verändern.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.formula.api as smf

# -------------------------------------------------------------------
# 1. Setup & Pfade
# -------------------------------------------------------------------
PAYMENTS_PATH = "/Users/stefanbechthold/statistik_business_analytics/data/olist_order_payments_dataset.csv"
ITEMS_PATH = "/Users/stefanbechthold/statistik_business_analytics/data/olist_order_items_dataset.csv"

GRAPHIC_DIR = "/Users/stefanbechthold/statistik_business_analytics/Grafiken/16_Quantile_Regression"
GRAPHIC_PATH = os.path.join(GRAPHIC_DIR, "34_quantile_regression.png")

os.makedirs(GRAPHIC_DIR, exist_ok=True)

def main():
    # -------------------------------------------------------------------
    # 2. Daten laden & bereinigen
    # -------------------------------------------------------------------
    df_payments = pd.read_csv(PAYMENTS_PATH)
    df_items = pd.read_csv(ITEMS_PATH)
    
    # Aggregation auf Bestell-Ebene
    order_payments = df_payments.groupby('order_id')['payment_value'].sum().reset_index()
    order_freight = df_items.groupby('order_id')['freight_value'].sum().reset_index()
    
    df_merged = pd.merge(order_payments, order_freight, on='order_id', how='inner')
    
    # Wir filtern extreme Ausreißer heraus, um die visuelle Darstellung nicht zu sprengen
    # (Obwohl Quantilsregression robust wäre, hilft es der Plot-Übersichtlichkeit)
    q_max_pay = df_merged['payment_value'].quantile(0.95)
    q_max_freight = df_merged['freight_value'].quantile(0.95)
    
    df_clean = df_merged[(df_merged['payment_value'] < q_max_pay) & 
                         (df_merged['freight_value'] < q_max_freight) & 
                         (df_merged['freight_value'] > 0)].copy()

    # Um die Berechnungszeit für das Skript zu optimieren, ziehen wir ein Sample
    df_sample = df_clean.sample(n=5000, random_state=42)

    # -------------------------------------------------------------------
    # 3. Modelle trainieren (OLS vs. Quantile Regression)
    # -------------------------------------------------------------------
    formula = "payment_value ~ freight_value"
    
    # OLS (Mittelwert-Regression) als Baseline
    ols_model = smf.ols(formula, df_sample).fit()
    
    # Quantile Regressions (10%, 50% Median, 90%)
    quantiles = [0.10, 0.50, 0.90]
    q_models = {}
    
    for q in quantiles:
        q_models[q] = smf.quantreg(formula, df_sample).fit(q=q)

    # -------------------------------------------------------------------
    # 4. Koeffizienten über viele Quantile berechnen (Für Plot 2)
    # -------------------------------------------------------------------
    quantiles_range = np.arange(0.05, 0.96, 0.05)
    coefs = []
    lower_ci = []
    upper_ci = []
    
    for q in quantiles_range:
        res = smf.quantreg(formula, df_sample).fit(q=q)
        coefs.append(res.params['freight_value'])
        lower_ci.append(res.conf_int().loc['freight_value', 0])
        upper_ci.append(res.conf_int().loc['freight_value', 1])

    # -------------------------------------------------------------------
    # 5. Visualisierung im Dark Mode Design
    # -------------------------------------------------------------------
    plt.style.use('dark_background')
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 7), facecolor='#121212')
    ax1.set_facecolor('#121212')
    ax2.set_facecolor('#121212')

    # X-Werte für die Plot-Linien
    x_pred = np.linspace(df_sample['freight_value'].min(), df_sample['freight_value'].max(), 50)
    
    # --- Plot 1: Scatterplot & Regressionsgeraden ---
    sns.scatterplot(x=df_sample['freight_value'], y=df_sample['payment_value'], 
                    color='#3498db', alpha=0.3, edgecolor='none', s=15, ax=ax1)
    
    # OLS Linie
    y_ols = ols_model.params['Intercept'] + ols_model.params['freight_value'] * x_pred
    ax1.plot(x_pred, y_ols, color='#e74c3c', linewidth=2, linestyle='--', label='OLS (Mittelwert)')
    
    # Quantils-Linien
    colors = {0.10: '#9b59b6', 0.50: '#2ecc71', 0.90: '#f1c40f'}
    labels = {0.10: '10. Quantil (Günstig)', 0.50: '50. Quantil (Median)', 0.90: '90. Quantil (Teuer)'}
    
    for q in quantiles:
        y_q = q_models[q].params['Intercept'] + q_models[q].params['freight_value'] * x_pred
        ax1.plot(x_pred, y_q, color=colors[q], linewidth=2.5, label=labels[q])

    ax1.set_title("1. Heterogene Effekte: Versand vs. Bestellwert", fontsize=14, fontweight='bold', color='white')
    ax1.set_xlabel("Versandkosten (BRL)")
    ax1.set_ylabel("Gesamter Bestellwert (BRL)")
    
    legend = ax1.legend(loc='upper left', facecolor='#1e1e1e', edgecolor='#333333')
    plt.setp(legend.get_texts(), color='white')

    # --- Plot 2: Steigung (Koeffizient) über die Verteilung ---
    ax2.plot(quantiles_range, coefs, color='#3498db', linewidth=3, marker='o', label='Quantil-Koeffizient (Steigung)')
    ax2.fill_between(quantiles_range, lower_ci, upper_ci, color='#3498db', alpha=0.2, label='95% Konfidenzintervall')
    
    # OLS Koeffizient als horizontale Konstante zum Vergleich
    ax2.axhline(ols_model.params['freight_value'], color='#e74c3c', linestyle='--', linewidth=2, label='OLS Koeffizient (Konstant)')
    
    ax2.set_title("2. Einfluss der Versandkosten auf das Bestell-Level", fontsize=14, fontweight='bold', color='white')
    ax2.set_xlabel("Quantil der Zielvariable (0.05 bis 0.95)")
    ax2.set_ylabel("Steigungskoeffizient (Effekt pro BRL Versand)")
    
    legend2 = ax2.legend(loc='upper left', facecolor='#1e1e1e', edgecolor='#333333')
    plt.setp(legend2.get_texts(), color='white')

    # Info-Box für Plot 2
    info_text = (
        "Business Interpretation:\n"
        "-----------------------\n"
        "Die klassische OLS-Regression (rot) geht davon aus, dass\n"
        "ein BRL mehr Versandkosten bei ALLEN Kunden den exakt\n"
        "gleichen Effekt auf den Bestellwert hat.\n\n"
        "Die Quantilsregression (blau) deckt auf:\n"
        "Bei Premium-Bestellungen (>80. Quantil) steigt der\n"
        "Umsatz-Hebel der Versandkosten signifikant steiler an\n"
        "als bei günstigen Basis-Bestellungen (<20. Quantil)."
    )
    props = dict(boxstyle='round,pad=1', facecolor='#1e1e1e', alpha=0.9, edgecolor='#3498db', linewidth=1.5)
    ax2.text(0.02, 0.45, info_text, transform=ax2.transAxes, fontsize=11,
             verticalalignment='center', bbox=props, color='#e0e0e0')

    plt.suptitle("Quantile Regression: Verteilungseffekte statt Durchschnittswerte", fontsize=18, fontweight='bold', color='white', y=1.05)
    plt.tight_layout()
    
    # Speichern
    plt.savefig(GRAPHIC_PATH, dpi=300, bbox_inches='tight', facecolor=fig.get_facecolor())
    plt.close()

if __name__ == "__main__":
    main()
    
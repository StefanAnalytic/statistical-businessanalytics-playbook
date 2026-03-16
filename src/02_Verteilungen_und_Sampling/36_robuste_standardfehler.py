"""
Griff 36: Robustheit der Standardfehler prüfen (Clustered SEs & Bootstrapping)

Was ist das und was macht das Skript?
Eine der häufigsten und gefährlichsten Fehlannahmen in der linearen Regression (OLS) ist, 
dass alle Datenpunkte unabhängig voneinander sind. Im Business-Kontext ist das selten der Fall:
Kunden leben im selben Bundesstaat, Bestellungen kommen aus derselben Filiale. 

Wenn Beobachtungen innerhalb einer Gruppe (Cluster) korrelieren, unterschätzt das Standard-OLS-Modell 
die Standardfehler (Standard Errors, SE). Das führt zu extrem engen Konfidenzintervallen und zu 
"falsch-positiven" Signifikanzen (p-Werten < 0.05, obwohl der Effekt zufällig ist).

Lösung: Clustered Standard Errors (bzw. Cluster-Bootstrapping). 
Diese Methode korrigiert die Unsicherheitsschätzung, indem sie die Abhängigkeit innerhalb der 
Cluster berücksichtigt. Die Punktschätzung (der Koeffizient) bleibt gleich, aber unser 
Vertrauensbereich (Konfidenzintervall) wird breiter und "ehrlicher".

Business Case in diesem Skript:
Verschlechtert eine längere Lieferzeit die Kundenbewertung? 
Wir nutzen Bestelldaten und Reviews. Da die Logistik-Infrastruktur und die Erwartungshaltung 
in Brasilien extrem vom Bundesstaat (customer_state) abhängen, clustern wir die Standardfehler 
auf Bundesstaats-Ebene. Wir zeigen, wie das naiven OLS-Modell sich eine zu hohe Sicherheit anmaßt.
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
ORDERS_PATH = "/Users/stefanbechthold/statistik_business_analytics/data/olist_orders_dataset.csv"
REVIEWS_PATH = "/Users/stefanbechthold/statistik_business_analytics/data/olist_order_reviews_dataset.csv"
CUSTOMERS_PATH = "/Users/stefanbechthold/statistik_business_analytics/data/olist_customers_dataset.csv"

# Speichern im Ordner für Sampling & Verteilungen (da SE-Korrekturen hier verortet sind)
GRAPHIC_DIR = "/Users/stefanbechthold/statistik_business_analytics/Grafiken/02_Verteilungen_und_Sampling"
GRAPHIC_PATH = os.path.join(GRAPHIC_DIR, "36_clustered_standard_errors.png")

os.makedirs(GRAPHIC_DIR, exist_ok=True)

def main():
    # -------------------------------------------------------------------
    # 2. Daten laden & bereiten
    # -------------------------------------------------------------------
    df_orders = pd.read_csv(ORDERS_PATH)
    df_reviews = pd.read_csv(REVIEWS_PATH)
    df_customers = pd.read_csv(CUSTOMERS_PATH)
    
    # Datum konvertieren und Lieferdauer berechnen
    df_orders['purchase_time'] = pd.to_datetime(df_orders['order_purchase_timestamp'], errors='coerce')
    df_orders['delivered_time'] = pd.to_datetime(df_orders['order_delivered_customer_date'], errors='coerce')
    df_orders['delivery_days'] = (df_orders['delivered_time'] - df_orders['purchase_time']).dt.total_seconds() / (24 * 3600)
    
    # Erste Bewertung pro Bestellung nehmen
    reviews_unique = df_reviews.drop_duplicates(subset=['order_id'])[['order_id', 'review_score']]
    
    # Mergen: Orders + Reviews + Customers (für das Cluster 'customer_state')
    df_merged = pd.merge(df_orders[['order_id', 'customer_id', 'delivery_days']], reviews_unique, on='order_id', how='inner')
    df_merged = pd.merge(df_merged, df_customers[['customer_id', 'customer_state']], on='customer_id', how='inner')
    
    df_clean = df_merged.dropna().copy()
    
    # Ausreißer in der Lieferzeit (z.B. > 60 Tage) entfernen für stabile Koeffizienten
    df_clean = df_clean[(df_clean['delivery_days'] > 0) & (df_clean['delivery_days'] <= 60)]

    # -------------------------------------------------------------------
    # 3. Modellierung: Naive OLS vs. Clustered SE OLS
    # -------------------------------------------------------------------
    formula = "review_score ~ delivery_days"
    
    # Modell 1: Naives OLS (Ignoriert Cluster-Abhängigkeiten)
    ols_naive = smf.ols(formula, data=df_clean).fit()
    
    # Modell 2: OLS mit Clustered Standard Errors
    # cov_type='cluster' teilt dem Modell mit, dass Fehler innerhalb der Gruppen (Bundesstaaten) korrelieren
    ols_clustered = smf.ols(formula, data=df_clean).fit(
        cov_type='cluster', 
        cov_kwds={'groups': df_clean['customer_state']}
    )

    # Werte für die Visualisierung extrahieren (Koeffizient und Konfidenzintervalle für 'delivery_days')
    coef = ols_naive.params['delivery_days'] # Punktschätzer ist bei beiden Modellen identisch!
    
    ci_naive = ols_naive.conf_int().loc['delivery_days']
    err_naive = ci_naive[1] - coef  # Distanz vom Koeffizienten zur oberen Grenze
    
    ci_clustered = ols_clustered.conf_int().loc['delivery_days']
    err_clustered = ci_clustered[1] - coef

    # -------------------------------------------------------------------
    # 4. Visualisierung im Dark Mode Design
    # -------------------------------------------------------------------
    plt.style.use('dark_background')
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7), facecolor='#121212')
    ax1.set_facecolor('#121212')
    ax2.set_facecolor('#121212')

    # --- Plot 1: Warum Clustern? Die Verteilung der Lieferzeiten pro Bundesstaat ---
    # Wir zeigen die Top 10 Bundesstaaten nach Bestellvolumen
    top_states = df_clean['customer_state'].value_counts().head(10).index
    plot_data = df_clean[df_clean['customer_state'].isin(top_states)]
    
    sns.boxplot(x='customer_state', y='delivery_days', data=plot_data, 
                order=top_states, palette='viridis', ax=ax1, 
                boxprops=dict(alpha=0.8), flierprops=dict(marker='o', markersize=2, alpha=0.2))
    
    ax1.set_title("1. Die Struktur der Daten: Logistik clustert nach Regionen", fontsize=14, fontweight='bold', color='white')
    ax1.set_xlabel("Bundesstaat (Customer State)")
    ax1.set_ylabel("Lieferzeit (Tage)")
    
    # Info-Box in Plot 1
    info_text_1 = (
        "Warum Standard-OLS hier irrt:\n"
        "Kunden in 'SP' (São Paulo) erhalten Pakete\n"
        "schneller als in 'BA' (Bahia). Ihre Bewertungen\n"
        "sind strukturell an ihre Region gebunden.\n"
        "Diese Abhängigkeit (Intraclass Correlation)\n"
        "verletzt die OLS-Annahme der Unabhängigkeit!"
    )
    props1 = dict(boxstyle='round,pad=1', facecolor='#1e1e1e', alpha=0.9, edgecolor='#f39c12', linewidth=1.5)
    ax1.text(0.05, 0.95, info_text_1, transform=ax1.transAxes, fontsize=11,
             verticalalignment='top', bbox=props1, color='#e0e0e0')

    # --- Plot 2: Vergleich der Konfidenzintervalle (Error Bars) ---
    models = ['Naives OLS\n(Unabhängigkeit angenommen)', 'OLS mit Clustered SE\n(Abhängigkeit korrigiert)']
    errors = [err_naive, err_clustered]
    colors = ['#e74c3c', '#2ecc71']

    # Errorbar Plot zeichnen
    for i, (model, err, color) in enumerate(zip(models, errors, colors)):
        ax2.errorbar(coef, i, xerr=err, fmt='o', color=color, markersize=12, 
                     linewidth=4, capsize=8, capthick=3)

    ax2.set_yticks([0, 1])
    ax2.set_yticklabels(models, fontsize=12, fontweight='bold')
    ax2.set_ylim(-0.5, 1.5)
    
    ax2.set_title("2. Der Effekt: Ehrlichere (breitere) Unsicherheits-Schätzung", fontsize=14, fontweight='bold', color='white')
    ax2.set_xlabel("Geschätzter Effekt pro Tag Lieferzeit auf die Bewertung (Sterne)")

    # Null-Linie einzeichnen (wichtig für Signifikanz)
    ax2.axvline(0, color='white', linestyle='--', linewidth=1, alpha=0.5)

    # Info-Box in Plot 2
    info_text_2 = (
        "Business Insight (Standardfehler):\n"
        "-----------------------------------\n"
        "Der geschätzte Effekt (Punkt) bleibt exakt gleich.\n"
        f"Aber das Konfidenzintervall beim Clustered-Modell\n"
        f"ist {err_clustered/err_naive:.1f}x breiter als beim naiven OLS-Modell!\n\n"
        "Würden wir nicht clustern, wären wir uns unserer\n"
        "Aussage viel zu sicher. Im Extremfall führt das\n"
        "dazu, dass Projekte freigegeben werden, deren\n"
        "Wirkung eigentlich noch im Zufallsbereich liegt."
    )
    props2 = dict(boxstyle='round,pad=1', facecolor='#1e1e1e', alpha=0.9, edgecolor='#2ecc71', linewidth=1.5)
    ax2.text(0.05, 0.05, info_text_2, transform=ax2.transAxes, fontsize=11,
             verticalalignment='bottom', bbox=props2, color='#e0e0e0')

    plt.suptitle("Statistical Inference: Clustered Standard Errors zur Vermeidung von False Positives", fontsize=18, fontweight='bold', color='white', y=1.05)
    plt.tight_layout()
    
    # Speichern
    plt.savefig(GRAPHIC_PATH, dpi=300, bbox_inches='tight', facecolor=fig.get_facecolor())
    plt.close()
    
if __name__ == "__main__":
    main()
"""
Griff 31: Was ist die zentrale Tendenz robust? (Mittelwerte & Robuste Maße)

Was ist das und was macht das Skript?
Der arithmetische Mittelwert (Average) ist die bekannteste, aber auch gefährlichste Metrik 
im Business Analytics. Bei extrem schiefen Verteilungen (z.B. Umsatz pro Kunde, wo wenige 
"Whales" Millionen ausgeben und viele "Minnows" nur Cents) zieht der Durchschnitt stark 
nach oben und verfälscht das Bild des "typischen" Kunden.

Dieses Skript berechnet und vergleicht verschiedene Maße der zentralen Tendenz:
1. Arithmetischer Mittelwert (Mean): Sensibel für Ausreißer.
2. Median: Der 50%-Punkt. Absolut robust gegen Ausreißer, aber ignoriert die Höhe der Extremwerte.
3. Trimmed Mean: Schneidet die extremsten X% (z.B. Top/Bottom 5%) ab und bildet dann den Durchschnitt.
4. Winsorized Mean: Capped (begrenzt) die Extremwerte auf das X-te Perzentil, anstatt sie zu löschen.
5. Geometrisches Mittel (Geometric Mean): Ideal für Wachstumsraten oder stark rechtsschiefe, streng positive Daten.

Business Case in diesem Skript:
Wir berechnen den "Customer Lifetime Value" (Gesamtumsatz pro Kunde) im Olist-Datensatz. 
Da E-Commerce-Umsätze extrem rechtsschief sind, visualisieren wir, wie stark der normale 
Mittelwert lügt und warum robuste Maße für das Reporting unerlässlich sind.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import trim_mean, gmean
from scipy.stats.mstats import winsorize

# -------------------------------------------------------------------
# 1. Setup & Pfade
# -------------------------------------------------------------------
PAYMENTS_PATH = "/Users/stefanbechthold/statistik_business_analytics/data/olist_order_payments_dataset.csv"
ORDERS_PATH = "/Users/stefanbechthold/statistik_business_analytics/data/olist_orders_dataset.csv"
CUSTOMERS_PATH = "/Users/stefanbechthold/statistik_business_analytics/data/olist_customers_dataset.csv"

GRAPHIC_DIR = "/Users/stefanbechthold/statistik_business_analytics/Grafiken/01_EDA"
GRAPHIC_PATH = os.path.join(GRAPHIC_DIR, "31_robuste_zentrale_tendenz.png")

os.makedirs(GRAPHIC_DIR, exist_ok=True)

def main():
    # -------------------------------------------------------------------
    # 2. Daten laden & Umsatz pro Kunde aggregieren
    # -------------------------------------------------------------------
    df_payments = pd.read_csv(PAYMENTS_PATH)
    df_orders = pd.read_csv(ORDERS_PATH)
    df_customers = pd.read_csv(CUSTOMERS_PATH)
    
    # Verknüpfung, um die eindeutige Kunden-ID (customer_unique_id) zu erhalten
    df_ord_cust = pd.merge(df_orders[['order_id', 'customer_id']], 
                           df_customers[['customer_id', 'customer_unique_id']], 
                           on='customer_id')
    df_merged = pd.merge(df_payments[['order_id', 'payment_value']], 
                         df_ord_cust, 
                         on='order_id')
    
    # Gesamtumsatz pro eindeutigem Kunden berechnen
    customer_spend = df_merged.groupby('customer_unique_id')['payment_value'].sum()
    
    # Für geometrisches Mittel müssen alle Werte strikt positiv sein
    customer_spend = customer_spend[customer_spend > 0]
    data = customer_spend.values

    # -------------------------------------------------------------------
    # 3. Metriken der zentralen Tendenz berechnen
    # -------------------------------------------------------------------
    mean_val = np.mean(data)
    median_val = np.median(data)
    
    # Trimmed Mean: Wir verwerfen die unteren 5% und oberen 5% komplett
    trimmed_val = trim_mean(data, proportiontocut=0.05)
    
    # Winsorized Mean: Wir kappen bei 5% und 95% (Werte darüber/darunter werden auf die Grenzen gesetzt)
    winsorized_data = winsorize(data, limits=[0.05, 0.05])
    winsorized_val = np.mean(winsorized_data)
    
    # Geometrisches Mittel: n-te Wurzel aus dem Produkt aller Werte
    geom_val = gmean(data)

    # -------------------------------------------------------------------
    # 4. Visualisierung im Darkmode
    # -------------------------------------------------------------------
    plt.style.use('dark_background')
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 7), facecolor='#121212')
    ax1.set_facecolor('#121212')
    ax2.set_facecolor('#121212')
    
    # Farben für die Metriken
    colors = {
        'Mean': '#e74c3c',
        'Winsorized Mean': '#e67e22',
        'Trimmed Mean': '#f1c40f',
        'Geometric Mean': '#9b59b6',
        'Median': '#2ecc71'
    }

    # --- Plot 1: Verteilung & Position der Metriken ---
    # Da die Verteilung extrem rechtsschief ist, begrenzen wir die X-Achse für den Plot 
    # auf das 98. Perzentil, damit man überhaupt etwas erkennen kann.
    plot_limit = np.percentile(data, 98)
    plot_data = data[data <= plot_limit]
    
    sns.histplot(plot_data, bins=50, color='#3498db', alpha=0.5, kde=True, ax=ax1, edgecolor='none')
    
    # Vertikale Linien für jede Metrik einzeichnen
    ax1.axvline(mean_val, color=colors['Mean'], linestyle='-', linewidth=2, label=f"Mean: {mean_val:.2f} BRL")
    ax1.axvline(winsorized_val, color=colors['Winsorized Mean'], linestyle='--', linewidth=2, label=f"Winsorized: {winsorized_val:.2f} BRL")
    ax1.axvline(trimmed_val, color=colors['Trimmed Mean'], linestyle='-.', linewidth=2, label=f"Trimmed: {trimmed_val:.2f} BRL")
    ax1.axvline(geom_val, color=colors['Geometric Mean'], linestyle=':', linewidth=2, label=f"Geometric: {geom_val:.2f} BRL")
    ax1.axvline(median_val, color=colors['Median'], linestyle='-', linewidth=2, label=f"Median: {median_val:.2f} BRL")
    
    ax1.set_title("1. Verteilung des Kundenumsatzes (bis 98. Perzentil)", fontsize=14, fontweight='bold', color='white')
    ax1.set_xlabel("Gesamtumsatz pro Kunde (BRL)")
    ax1.set_ylabel("Anzahl der Kunden")
    
    # Legende anpassen
    legend = ax1.legend(loc='upper right', facecolor='#1e1e1e', edgecolor='#333333')
    plt.setp(legend.get_texts(), color='white')

    # --- Plot 2: Direkter Vergleich der Metriken (Bar Chart) ---
    metrics_names = list(colors.keys())
    metrics_values = [mean_val, winsorized_val, trimmed_val, geom_val, median_val]
    bar_colors = list(colors.values())
    
    bars = ax2.barh(metrics_names, metrics_values, color=bar_colors, edgecolor='none')
    ax2.invert_yaxis()  # Mean oben
    
    ax2.set_title("2. Robuste Maße vs. Arithmetisches Mittel", fontsize=14, fontweight='bold', color='white')
    ax2.set_xlabel("Umsatz in BRL")
    
    # Werte an die Balken schreiben
    for bar in bars:
        width = bar.get_width()
        ax2.text(width + 2, bar.get_y() + bar.get_height()/2, 
                 f"{width:.2f} BRL", 
                 ha='left', va='center', color='white', fontweight='bold', fontsize=11)

    # Erklärende Info-Box
    info_text = (
        "Business Insight:\n"
        "------------------\n"
        "Der normale 'Mean' wird massiv durch wenige Extremkäufer verzerrt.\n"
        "Er suggeriert einen typischen Kundenumsatz, der in der Realität\n"
        "für fast 70% der Kunden unerreichbar ist.\n\n"
        "Trimmed und Winsorized Means dämpfen diesen Effekt ab, indem sie\n"
        "Ausreißer ignorieren oder begrenzen. Der Median liefert den\n"
        "absoluten Mittelpunkt der Gesellschaft."
    )
    props = dict(boxstyle='round,pad=1', facecolor='#1e1e1e', alpha=0.9, edgecolor='#3498db', linewidth=1.5)
    ax2.text(0.5, 0.2, info_text, transform=ax2.transAxes, fontsize=11,
             verticalalignment='center', horizontalalignment='left', bbox=props, color='#e0e0e0')

    plt.suptitle("Zentrale Tendenz bei schiefen Verteilungen (Robuste Statistik)", fontsize=18, fontweight='bold', color='white', y=1.05)
    plt.tight_layout()
    
    # Speichern
    plt.savefig(GRAPHIC_PATH, dpi=300, bbox_inches='tight', facecolor=fig.get_facecolor())
    plt.close()

if __name__ == "__main__":
    main()
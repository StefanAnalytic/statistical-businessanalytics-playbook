"""
Griff 44: Customer-Economics (LTV) berechnen? (Cohort Analysis & Retention Curves)

Was ist das und was macht das Skript?
Der Customer Lifetime Value (LTV oder CLV) und die Retention Rate (Kundenbindung) 
sind die wichtigsten Metriken im E-Commerce. Wenn wir wissen, wie viel ein Kunde 
über seine gesamte Lebenszeit bei uns ausgibt, wissen wir auch, wie viel wir im 
Marketing für seine Akquise (Customer Acquisition Cost - CAC) ausgeben dürfen.

Die Basis dafür ist die Kohortenanalyse (Cohort Analysis):
Kunden werden in "Kohorten" eingeteilt, basierend auf dem Monat ihres ersten Einkaufs. 
Dann beobachten wir diese spezifische Gruppe über die Zeit:
1. Retention Heatmap: Wie viel Prozent der Januar-Kohorte haben im Februar, 
   März usw. erneut gekauft?
2. Cumulative LTV: Wie entwickelt sich der durchschnittliche kumulierte Umsatz 
   einer Kohorte über die Zeit?

Business Case in diesem Skript:
Wir berechnen die Retention-Matrix und die LTV-Kurve für den Olist-Datensatz. 
ACHTUNG BUSINESS-REALITÄT: Olist ist ein Marktplatz, auf dem Kunden oft nur ein 
einziges Mal kaufen. Wir werden sehen, dass die Retention-Raten extrem niedrig sind 
(unter 1%). Ein perfektes Beispiel dafür, warum LTV-Modelle stark vom Geschäftsmodell abhängen!
"""
x
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.colors as mcolors

# -------------------------------------------------------------------
# 1. Setup & Pfade
# -------------------------------------------------------------------
ORDERS_PATH = "/Users/stefanbechthold/statistik_business_analytics/data/olist_orders_dataset.csv"
PAYMENTS_PATH = "/Users/stefanbechthold/statistik_business_analytics/data/olist_order_payments_dataset.csv"
CUSTOMERS_PATH = "/Users/stefanbechthold/statistik_business_analytics/data/olist_customers_dataset.csv"

GRAPHIC_DIR = "/Users/stefanbechthold/statistik_business_analytics/Grafiken/04_Metriken_und_Wachstum"
GRAPHIC_PATH = os.path.join(GRAPHIC_DIR, "44_cohort_analysis_ltv.png")

os.makedirs(GRAPHIC_DIR, exist_ok=True)

def main():
    print("🚀 Starte Griff 44: Cohort Analysis, Retention & LTV...")
    
    # -------------------------------------------------------------------
    # 2. Daten laden & Kohorten definieren
    # -------------------------------------------------------------------
    df_orders = pd.read_csv(ORDERS_PATH)
    df_payments = pd.read_csv(PAYMENTS_PATH)
    df_customers = pd.read_csv(CUSTOMERS_PATH)
    
    # Nur ausgelieferte Bestellungen betrachten
    df_orders = df_orders[df_orders['order_status'] == 'delivered'].copy()
    
    # Bestelldatum in Monat/Jahr umwandeln
    df_orders['order_purchase_timestamp'] = pd.to_datetime(df_orders['order_purchase_timestamp'])
    df_orders['order_month'] = df_orders['order_purchase_timestamp'].dt.to_period('M')
    
    # Gesamtwert pro Bestellung berechnen
    order_values = df_payments.groupby('order_id')['payment_value'].sum().reset_index()
    
    # Mergen: Orders + Customers (für unique ID) + Payments
    df_merged = pd.merge(df_orders[['order_id', 'customer_id', 'order_month']], 
                         df_customers[['customer_id', 'customer_unique_id']], on='customer_id', how='inner')
    df_merged = pd.merge(df_merged, order_values, on='order_id', how='inner')
    
    # Kohorte bestimmen: Monat des ERSTEN Einkaufs jedes Kunden
    df_merged['cohort_month'] = df_merged.groupby('customer_unique_id')['order_month'].transform('min')
    
    # Kohorten-Index berechnen (Wie viele Monate sind seit dem ersten Kauf vergangen?)
    def get_date_int(df, column):
        year = df[column].dt.year
        month = df[column].dt.month
        return year, month

    order_year, order_month = get_date_int(df_merged, 'order_month')
    cohort_year, cohort_month = get_date_int(df_merged, 'cohort_month')
    
    years_diff = order_year - cohort_year
    months_diff = order_month - cohort_month
    
    # Index 0 ist der Monat des Erstkaufs, Index 1 der Folgemonat usw.
    df_merged['cohort_index'] = years_diff * 12 + months_diff
    
    # Filter: Wir betrachten nur das Jahr 2017, um eine saubere, vollständige Matrix zu erhalten
    df_2017 = df_merged[(df_merged['cohort_month'] >= '2017-01') & (df_merged['cohort_month'] <= '2017-12')].copy()

    # -------------------------------------------------------------------
    # 3. Retention Matrix berechnen
    # -------------------------------------------------------------------
    # Zählen der eindeutigen Kunden pro Kohorte und Index
    cohort_data = df_2017.groupby(['cohort_month', 'cohort_index'])['customer_unique_id'].nunique().reset_index()
    
    # Pivot-Tabelle: Zeilen = Kohortenmonat, Spalten = Index (0, 1, 2...)
    cohort_counts = cohort_data.pivot(index='cohort_month', columns='cohort_index', values='customer_unique_id')
    
    # Retention-Rate in % berechnen (Anzahl in Index X / Anzahl in Index 0)
    cohort_sizes = cohort_counts.iloc[:, 0]
    retention = cohort_counts.divide(cohort_sizes, axis=0) * 100
    
    # -------------------------------------------------------------------
    # 4. Cumulative LTV berechnen
    # -------------------------------------------------------------------
    # Umsatz pro Kohorte und Index aggregieren
    cohort_revenue = df_2017.groupby(['cohort_month', 'cohort_index'])['payment_value'].sum().reset_index()
    revenue_pivot = cohort_revenue.pivot(index='cohort_month', columns='cohort_index', values='payment_value')
    
    # Kumulierten Umsatz berechnen (Wie viel hat die Kohorte bis Monat X insgesamt ausgegeben?)
    cumulative_revenue = revenue_pivot.cumsum(axis=1)
    
    # Um den "LTV" (durchschnittlichen Wert pro Kunde) zu erhalten, teilen wir durch die initiale Kohortengröße
    cumulative_ltv = cumulative_revenue.divide(cohort_sizes, axis=0)

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
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8), facecolor='#121212')

    # --- Plot 1: Retention Heatmap ---
    # Wir schneiden Index 0 (immer 100%) ab, da er die Farbskala dominiert und man die feinen Unterschiede sonst nicht sieht.
    retention_plot = retention.iloc[:, 1:12] 
    
    # Eigene Colormap für den Darkmode
    cmap = sns.cubehelix_palette(start=2.8, rot=.1, as_cmap=True, reverse=True)
    
    sns.heatmap(retention_plot, annot=True, fmt=".2f", cmap=cmap, vmin=0.0, vmax=1.0, 
                linewidths=.5, ax=ax1, cbar_kws={'label': 'Retention Rate (%)'})
    
    ax1.set_title("1. Cohort Retention Heatmap (Monate 1 bis 11)", fontsize=14, fontweight='bold', color='white')
    ax1.set_xlabel("Monate nach Erstkauf (Cohort Index)")
    ax1.set_ylabel("Kohorte (Monat des Erstkaufs)")
    ax1.set_yticklabels([str(x) for x in retention_plot.index], rotation=0)

    # Info-Box Plot 1
    info_1 = (
        "Olist = Kein Abo-Geschäft:\n"
        "Die Retention im E-Commerce Marktplatz ist extrem niedrig.\n"
        "Nur etwa 0.3% bis 0.5% der Kunden kaufen in den Monaten\n"
        "nach ihrem Erstkauf erneut ein. Das bedeutet, das Wachstum\n"
        "wird fast ausschließlich durch Neukunden getrieben."
    )
    props = dict(boxstyle='round,pad=1', facecolor='#1e1e1e', alpha=0.9, edgecolor='#9b59b6', linewidth=1.5)
    ax1.text(0.5, -0.2, info_1, transform=ax1.transAxes, fontsize=11,
             verticalalignment='top', horizontalalignment='center', bbox=props, color='white')

    # --- Plot 2: Cumulative LTV Curves ---
    # Wir plotten die ersten 6 Kohorten des Jahres 2017
    colors = sns.color_palette("Set2", 6)
    cohorts_to_plot = cumulative_ltv.index[:6]
    
    for i, cohort in enumerate(cohorts_to_plot):
        data = cumulative_ltv.loc[cohort].dropna()
        ax2.plot(data.index, data.values, marker='o', linewidth=2.5, color=colors[i], label=str(cohort))
        
    ax2.set_title("2. Cumulative LTV (Umsatz-Entwicklung pro Kunde)", fontsize=14, fontweight='bold', color='white')
    ax2.set_xlabel("Monate nach Erstkauf")
    ax2.set_ylabel("Kumulierter Umsatz pro Akquiriertem Kunde (BRL)")
    
    legend2 = ax2.legend(title="Kohorte", loc='upper left', facecolor='#1e1e1e', edgecolor='#333333')
    plt.setp(legend2.get_title(), color='white')
    plt.setp(legend2.get_texts(), color='white')

    # Info-Box Plot 2
    info_2 = (
        "Business Economics (CAC vs. LTV):\n"
        "---------------------------------\n"
        "Die Kurven steigen nach Monat 0 kaum noch an (Flache Linie).\n"
        "Der Kunde gibt beim Erstkauf z.B. 160 BRL aus, danach aber\n"
        "nichts mehr. Konsequenz für das Marketing-Budget:\n"
        "Die Akquisekosten (CAC) müssen bereits beim ersten Kauf\n"
        "vollständig amortisiert werden, da kein 'Back-End' Umsatz folgt!"
    )
    props2 = dict(boxstyle='round,pad=1', facecolor='#1e1e1e', alpha=0.9, edgecolor='#e74c3c', linewidth=1.5)
    ax2.text(0.45, 0.4, info_2, transform=ax2.transAxes, fontsize=11,
             verticalalignment='center', bbox=props2, color='white')

    plt.suptitle("Customer Economics: Kohorten, Retention & Lifetime Value", 
                 fontsize=18, fontweight='bold', color='white', y=1.05)
    plt.tight_layout()
    
    # Speichern der Grafik
    plt.savefig(GRAPHIC_PATH, dpi=300, bbox_inches='tight', facecolor=fig.get_facecolor())
    plt.close()
    
    print(f"✅ Grafik erfolgreich gespeichert unter:\n{GRAPHIC_PATH}")

if __name__ == "__main__":
    main()
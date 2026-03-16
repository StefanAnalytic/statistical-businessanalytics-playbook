"""
Griff 32: Wie messen wir Wachstum korrekt? (Growth Rates, CAGR, Log-Returns)

Was ist das und was macht das Skript?
Wachstum (Growth) ist die wichtigste Metrik für Startups und E-Commerce-Unternehmen. 
Allerdings wird Wachstum oft falsch berechnet oder durch hohe Volatilität verzerrt.

Dieses Skript berechnet und visualisiert vier robuste Methoden zur Wachstumsmessung:
1. Simple Growth Rate (MoM - Month-over-Month): Die klassische prozentuale Veränderung. 
   Problem: Sie ist asymmetrisch (+50% und -50% bringen dich nicht zum Ausgangswert zurück).
2. Log-Returns (Logarithmische Renditen): Berechnet sich durch ln(V_neu / V_alt). 
   Vorteil: Log-Returns sind symmetrisch und zeitlich additiv. Sie eignen sich perfekt 
   für volatile Zeitreihen und statistische Modellierungen.
3. Rolling Growth: Ein gleitender Durchschnitt glättet kurzfristige Rausch-Ausschläge 
   und zeigt den wahren, zugrundeliegenden Trend.
4. CAGR (Compound Annual Growth Rate): Die "geglättete" jährliche Wachstumsrate. Sie 
   ignoriert die Volatilität dazwischen und berechnet, mit welchem konstanten Prozentsatz 
   das Unternehmen wachsen müsste, um vom Start- zum Endwert zu gelangen.

Business Case in diesem Skript:
Wir berechnen das monatliche Umsatzwachstum (MRR-Proxy) des Olist-Marktplatzes für 
das Jahr 2017 bis Mitte 2018 und vergleichen die verschiedenen Wachstumsmetriken.
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

GRAPHIC_DIR = "/Users/stefanbechthold/statistik_business_analytics/Grafiken/04_Metriken_und_Wachstum"
GRAPHIC_PATH = os.path.join(GRAPHIC_DIR, "32_wachstum_cagr_logreturns.png")

os.makedirs(GRAPHIC_DIR, exist_ok=True)

def main():
    # -------------------------------------------------------------------
    # 2. Daten laden & Monatlichen Umsatz aggregieren
    # -------------------------------------------------------------------
    df_payments = pd.read_csv(PAYMENTS_PATH)
    df_orders = pd.read_csv(ORDERS_PATH)
    
    # Verknüpfen und Datum konvertieren
    df_merged = pd.merge(df_orders[['order_id', 'order_purchase_timestamp']], 
                         df_payments[['order_id', 'payment_value']], 
                         on='order_id')
    
    df_merged['purchase_date'] = pd.to_datetime(df_merged['order_purchase_timestamp'])
    
    # Filtern auf einen sauberen, kontinuierlichen Zeitraum (Jan 2017 - Aug 2018)
    mask = (df_merged['purchase_date'] >= '2017-01-01') & (df_merged['purchase_date'] < '2018-09-01')
    df_clean = df_merged[mask].copy()
    
    # Monatliche Aggregation (MRR Proxy)
    df_clean.set_index('purchase_date', inplace=True)
    monthly_revenue = df_clean['payment_value'].resample('MS').sum().reset_index()
    monthly_revenue.columns = ['Month', 'Revenue']

    # -------------------------------------------------------------------
    # 3. Wachstums-Metriken berechnen
    # -------------------------------------------------------------------
    # 1. Simple Month-over-Month (MoM) Growth
    monthly_revenue['MoM_Growth'] = monthly_revenue['Revenue'].pct_change()
    
    # 2. Log-Returns: ln(Heutiger Wert / Gestriger Wert)
    monthly_revenue['Log_Return'] = np.log(monthly_revenue['Revenue'] / monthly_revenue['Revenue'].shift(1))
    
    # 3. Rolling Mean (3 Monate) für Trend-Glättung
    monthly_revenue['Rolling_3M_Revenue'] = monthly_revenue['Revenue'].rolling(window=3).mean()
    
    # 4. CAGR Berechnung (Compound Annual Growth Rate)
    # Formel: (Endwert / Startwert) ^ (1 / Anzahl_Jahre) - 1
    start_value = monthly_revenue['Revenue'].iloc[0]
    end_value = monthly_revenue['Revenue'].iloc[-1]
    
    # Zeitraum in Jahren (Anzahl der Monate / 12)
    num_months = len(monthly_revenue) - 1
    years = num_months / 12.0
    
    cagr = (end_value / start_value) ** (1 / years) - 1

    # -------------------------------------------------------------------
    # 4. Visualisierung im Darkmode
    # -------------------------------------------------------------------
    plt.style.use('dark_background')
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 12), facecolor='#121212', gridspec_kw={'height_ratios': [1.5, 1]})
    ax1.set_facecolor('#121212')
    ax2.set_facecolor('#121212')
    
    months_str = monthly_revenue['Month'].dt.strftime('%Y-%m')

    # --- Plot 1: Absoluter Umsatz & Rolling Trend ---
    ax1.bar(months_str, monthly_revenue['Revenue'], color='#3498db', alpha=0.6, label='Monatlicher Umsatz (BRL)')
    ax1.plot(months_str, monthly_revenue['Rolling_3M_Revenue'], color='#e74c3c', linewidth=3, marker='o', label='3-Monats-Trend (Geglättet)')
    
    ax1.set_title("1. Absolute Wachstumsentwicklung & Trendglättung", fontsize=14, fontweight='bold', color='white')
    ax1.set_ylabel("Umsatz in BRL")
    ax1.tick_params(axis='x', rotation=45)
    ax1.legend(loc='upper left', facecolor='#1e1e1e', edgecolor='#333333')

    # Info-Box für CAGR in Plot 1
    cagr_text = (
        f"Langfristiges Wachstum (CAGR):\n"
        f"-------------------------------\n"
        f"Start ({months_str.iloc[0]}): {start_value:,.0f} BRL\n"
        f"Ende ({months_str.iloc[-1]}): {end_value:,.0f} BRL\n"
        f"CAGR (Annualisiert): {cagr*100:.1f} % p.a.\n\n"
        f"CAGR ignoriert die Volatilität und\n"
        f"zeigt das 'echte' Basiswachstum."
    )
    props = dict(boxstyle='round,pad=1', facecolor='#1e1e1e', alpha=0.9, edgecolor='#e74c3c', linewidth=1.5)
    ax1.text(0.02, 0.6, cagr_text, transform=ax1.transAxes, fontsize=11,
             verticalalignment='center', bbox=props, color='#e0e0e0', fontweight='bold')

    # --- Plot 2: Volatilität messen (MoM vs. Log-Returns) ---
    # Wir plotten ab Index 1, da Index 0 NaN bei pct_change ist
    ax2.plot(months_str[1:], monthly_revenue['MoM_Growth'][1:] * 100, color='#2ecc71', linewidth=2, marker='s', label='Simple MoM Growth (%)')
    ax2.plot(months_str[1:], monthly_revenue['Log_Return'][1:] * 100, color='#f1c40f', linewidth=2, linestyle='--', marker='^', label='Log-Return (%)')
    
    ax2.axhline(0, color='white', linewidth=1, linestyle='-')
    ax2.set_title("2. Relative Wachstumsraten: Simple Growth vs. Log-Returns", fontsize=14, fontweight='bold', color='white')
    ax2.set_ylabel("Veränderung in %")
    ax2.tick_params(axis='x', rotation=45)
    ax2.legend(loc='upper right', facecolor='#1e1e1e', edgecolor='#333333')

    # Info-Box für Log-Returns in Plot 2
    log_text = (
        "Warum Log-Returns?\n"
        "Bei extremen Sprüngen (z.B. Black Friday) überschätzt die simple %-Rate\n"
        "das Wachstum mathematisch. Log-Returns sind additiv und symmetrisch.\n"
        "Sie eignen sich besser für Data-Science-Modelle (z.B. Time Series Forecasting)."
    )
    ax2.text(0.02, 0.85, log_text, transform=ax2.transAxes, fontsize=11,
             verticalalignment='top', bbox=props, color='#e0e0e0')

    plt.suptitle("Business Growth Analytics: Korrekte Messung von Wachstum", fontsize=18, fontweight='bold', color='white', y=1.02)
    plt.tight_layout()
    
    # Speichern der Grafik
    plt.savefig(GRAPHIC_PATH, dpi=300, bbox_inches='tight', facecolor=fig.get_facecolor())
    plt.close()

if __name__ == "__main__":
    main()
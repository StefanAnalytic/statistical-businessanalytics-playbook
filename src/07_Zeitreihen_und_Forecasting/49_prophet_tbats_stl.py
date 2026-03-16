"""
Griff 49: Bessere Zeitreihen-Tools (Prophet & STL Decomposition)

Was ist das und was macht das Skript?
Klassische ARIMA-Modelle sind mächtig, aber sie tun sich schwer mit komplexen, 
überlappenden Saisonalitäten (z.B. wöchentliche Schwankungen UND jährliche Trends 
gleichzeitig) sowie mit unregelmäßigen Feiertagen (Holidays).

Hier kommen moderne Tools ins Spiel:
1. STL (Seasonal and Trend decomposition using Loess): Zerlegt eine Zeitreihe robust 
   in ihre drei wahren Bestandteile: Trend (Langfristig), Saisonalität (Zyklisch) 
   und Residuen (Rauschen/Anomalien). Perfekt für die EDA von Zeitreihen.
2. Prophet (von Meta/Facebook): Ein additives Regressionsmodell, das extrem gut 
   mit fehlenden Daten, Ausreißern und multiplen Saisonalitäten umgehen kann. 
   Es ist der heutige Industrie-Standard für skalierbares Business-Forecasting.

Business Case in diesem Skript:
Wir analysieren das tägliche Bestellvolumen (Order Count) bei Olist. Zuerst zerlegen 
wir die Zeitreihe mit STL, um das typische Wochentags-Muster zu verstehen (Wann kaufen 
die Brasilianer am meisten?). Danach trainieren wir ein Prophet-Modell, fügen 
Feiertagseffekte hinzu und prognostizieren die nächsten 30 Tage in die Zukunft.

WICHTIG: Benötigt das Paket 'prophet' -> pip install prophet
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.seasonal import STL

try:
    from prophet import Prophet
except ImportError:
    print("FEHLER: Die Bibliothek 'prophet' fehlt. Bitte ausführen: pip install prophet")
    exit()

# -------------------------------------------------------------------
# 1. Setup & Pfade
# -------------------------------------------------------------------
ORDERS_PATH = "/Users/stefanbechthold/statistik_business_analytics/data/olist_orders_dataset.csv"

GRAPHIC_DIR = "/Users/stefanbechthold/statistik_business_analytics/Grafiken/07_Zeitreihen_und_Forecasting"
GRAPHIC_PATH = os.path.join(GRAPHIC_DIR, "49_prophet_stl_forecasting.png")

os.makedirs(GRAPHIC_DIR, exist_ok=True)

def main():
    print("🚀 Starte Griff 49: Advanced Time Series (STL & Prophet)...")
    
    # -------------------------------------------------------------------
    # 2. Daten laden & Tägliche Zeitreihe aggregieren
    # -------------------------------------------------------------------
    df_orders = pd.read_csv(ORDERS_PATH)
    
    # Nur valide Bestellungen
    df_orders = df_orders[df_orders['order_status'] != 'canceled'].copy()
    
    # Datum konvertieren
    df_orders['purchase_date'] = pd.to_datetime(df_orders['order_purchase_timestamp']).dt.date
    
    # Aggregation: Anzahl der Bestellungen pro Tag
    daily_orders = df_orders.groupby('purchase_date').size().reset_index(name='orders')
    daily_orders['purchase_date'] = pd.to_datetime(daily_orders['purchase_date'])
    
    # Auf den validen Kernzeitraum filtern (2017 bis Mitte 2018)
    mask = (daily_orders['purchase_date'] >= '2017-01-01') & (daily_orders['purchase_date'] <= '2018-08-31')
    daily_orders = daily_orders[mask].copy()
    
    # Index für STL setzen und fehlende Tage mit 0 füllen
    ts_data = daily_orders.set_index('purchase_date').asfreq('D', fill_value=0)

    # -------------------------------------------------------------------
    # 3. STL Decomposition (Trend & Saisonalität extrahieren)
    # -------------------------------------------------------------------
    # period=7 für wöchentliche Saisonalität
    stl = STL(ts_data['orders'], period=7, robust=True)
    res = stl.fit()
    
    ts_data['trend'] = res.trend
    ts_data['seasonal'] = res.seasonal
    ts_data['resid'] = res.resid

    # -------------------------------------------------------------------
    # 4. Forecasting mit Meta Prophet
    # -------------------------------------------------------------------
    # Prophet erwartet zwingend die Spaltennamen 'ds' (Datum) und 'y' (Wert)
    df_prophet = daily_orders.rename(columns={'purchase_date': 'ds', 'orders': 'y'})
    
    # Modell initialisieren (inklusive länderspezifischer Feiertage)
    m = Prophet(yearly_seasonality=True, weekly_seasonality=True, daily_seasonality=False)
    m.add_country_holidays(country_name='BR') # Brasilianische Feiertage für Olist!
    
    m.fit(df_prophet)
    
    # Dataframe für die Zukunft erstellen (30 Tage Forecast)
    future = m.make_future_dataframe(periods=30)
    forecast = m.predict(future)

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
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 12), facecolor='#121212')

    # --- Plot 1: STL Decomposition (Fokus auf wöchentliche Saisonalität) ---
    # Wir zeigen nur einen 2-Monats-Ausschnitt, damit man das Zick-Zack-Muster gut erkennt
    plot_start = '2017-09-01'
    plot_end = '2017-10-31'
    ts_zoom = ts_data[plot_start:plot_end]
    
    ax1.plot(ts_zoom.index, ts_zoom['orders'], color='#3498db', alpha=0.5, marker='o', label='Echte Bestellungen')
    ax1.plot(ts_zoom.index, ts_zoom['trend'], color='#e74c3c', linewidth=3, label='Geglätteter Trend (STL)')
    
    # Saisonalität als Balken um den Nullpunkt unten im Chart andeuten
    ax1.bar(ts_zoom.index, ts_zoom['seasonal'], color='#2ecc71', alpha=0.4, label='Wöchentliche Saisonalität (+/-)')
    
    ax1.set_title("1. STL Decomposition: Rauschen entfernen, wahren Trend erkennen (Zoom auf 2 Monate)", 
                  fontsize=14, fontweight='bold', color='white')
    ax1.set_ylabel("Anzahl Bestellungen")
    ax1.set_xlim(pd.to_datetime(plot_start), pd.to_datetime(plot_end))
    
    legend1 = ax1.legend(loc='upper left', facecolor='#1e1e1e', edgecolor='#333333')
    plt.setp(legend1.get_texts(), color='white')

    # Info-Box Plot 1
    info_1 = (
        "Business Insight (Wochentage):\n"
        "------------------------------\n"
        "STL isoliert das zyklische Kaufverhalten der Brasilianer.\n"
        "Die grünen Balken zeigen: An Wochentagen (Di/Mi) wird\n"
        "am meisten bestellt (positive Saisonalität).\n"
        "Am Wochenende (Sa/So) bricht das Volumen massiv ein (negative Balken).\n"
        "Das ist essenziell für die Personaleinsatzplanung im Support!"
    )
    props1 = dict(boxstyle='round,pad=1', facecolor='#1e1e1e', alpha=0.9, edgecolor='#2ecc71', linewidth=1.5)
    ax1.text(0.65, 0.95, info_1, transform=ax1.transAxes, fontsize=11,
             verticalalignment='top', bbox=props1, color='white')

    # --- Plot 2: Prophet Forecast ---
    # Historische Daten
    ax2.plot(df_prophet['ds'], df_prophet['y'], color='#3498db', alpha=0.4, label='Historie')
    
    # Forecast Linie
    ax2.plot(forecast['ds'], forecast['yhat'], color='#f1c40f', linewidth=2, label='Prophet Forecast')
    
    # Unsicherheits-Intervall (Konfidenz)
    ax2.fill_between(forecast['ds'], forecast['yhat_lower'], forecast['yhat_upper'], 
                     color='#f1c40f', alpha=0.2, label='80% Konfidenzintervall')
    
    # Feiertage (Ausreißer nach oben/unten) markieren
    # Wir suchen die stärksten Feiertags-Effekte im Modell
    holidays = forecast[forecast['holidays'].abs() > 50]
    ax2.scatter(holidays['ds'], holidays['yhat'], color='#e74c3c', s=50, zorder=5, label='Erkannte Feiertags-Effekte')
    
    # Trennlinie Train / Forecast
    forecast_start = df_prophet['ds'].max()
    ax2.axvline(forecast_start, color='white', linestyle='--', linewidth=2, label='Start Forecast (30 Tage)')

    ax2.set_title("2. Meta Prophet: Skalierbares Forecasting mit Feiertagen & multipler Saisonalität", 
                  fontsize=14, fontweight='bold', color='white')
    ax2.set_ylabel("Anzahl Bestellungen")
    ax2.set_xlim(pd.to_datetime('2017-06-01'), forecast['ds'].max())
    
    legend2 = ax2.legend(loc='upper left', facecolor='#1e1e1e', edgecolor='#333333')
    plt.setp(legend2.get_texts(), color='white')

    # Info-Box Plot 2
    info_2 = (
        "Warum Prophet?\n"
        "---------------\n"
        "Es ist ein 'Additive Regression Model' (Trend + Season + Holidays).\n"
        "Es baut den Forecast (gelbe Linie) extrem robust auf, ohne bei\n"
        "Ausreißern (wie dem Black Friday) sofort aus der Kurve zu fliegen.\n"
        "Die roten Punkte zeigen, wo das Modell Feiertage aus Brasilien\n"
        "automatisch als Erklärung für Anomalien herangezogen hat."
    )
    props2 = dict(boxstyle='round,pad=1', facecolor='#1e1e1e', alpha=0.9, edgecolor='#f1c40f', linewidth=1.5)
    ax2.text(0.02, 0.45, info_2, transform=ax2.transAxes, fontsize=11,
             verticalalignment='center', bbox=props2, color='white')

    plt.suptitle("Advanced Forecasting: STL Decomposition & Prophet", 
                 fontsize=18, fontweight='bold', color='white', y=1.05)
    plt.tight_layout()
    
    # Speichern
    plt.savefig(GRAPHIC_PATH, dpi=300, bbox_inches='tight', facecolor=fig.get_facecolor())
    plt.close()
    
    print(f"✅ Grafik erfolgreich gespeichert unter:\n{GRAPHIC_PATH}")

if __name__ == "__main__":
    main()
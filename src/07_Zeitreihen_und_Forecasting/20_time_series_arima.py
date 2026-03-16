"""
Griff 20: Wie gut forecaste ich die Zukunft? (Time Series: Decompose, ADF, ARIMA)

Was ist das und was macht das Skript?
Wir verlassen die "normale" Statistik und betreten die Welt der Zeitreihen (Time Series).
Wenn Daten über die Zeit erhoben werden (Umsatz pro Woche), hängen die Werte voneinander ab.
Der Umsatz von heute wird oft vom Umsatz von gestern beeinflusst.

Dieses Skript macht 3 entscheidende Schritte für einen sauberen Forecast (Vorhersage):
1. Decompose (Zerlegung): Bricht den wöchentlichen Umsatz in 3 Bausteine auf: 
   Trend (Geht es generell bergauf?), Seasonality (Gibt es wiederkehrende Muster, z.B. Monatsende?) 
   und Residuals (Zufälliges Rauschen).
2. ADF-Test (Augmented Dickey-Fuller): Prüft auf "Stationarität". Das heißt: Sind Mittelwert 
   und Schwankung über die Zeit konstant? ARIMA-Modelle verlangen zwingend stationäre Daten!
3. ARIMA Forecast: Ein klassisches Modell zur Vorhersage. Es nutzt vergangene Werte (AR = AutoRegressive),
   gleicht Trends aus (I = Integrated) und nutzt vergangene Fehler (MA = Moving Average), 
   um die nächsten Wochen vorherzusagen.

Business Case in diesem Skript:
Wir nehmen alle echten Olist-Bestellungen aus 2017 und 2018, fassen sie zu wöchentlichen 
Bestellvolumina zusammen und versuchen, die nächsten 8 Wochen (2 Monate) in die Zukunft zu prognostizieren!
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Statsmodels für alle Zeitreihen-Analysen
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.arima.model import ARIMA

import warnings
warnings.filterwarnings("ignore") # Unterdrückt irrelevante Warnungen von Statsmodels

# -------------------------------------------------------------------
# 1. Setup & Pfade
# -------------------------------------------------------------------
# Wir wechseln nun in den Ordner "07_Zeitreihen_und_Forecasting"
ORDERS_PATH = "/Users/stefanbechthold/statistik_business_analytics/data/olist_orders_dataset.csv"

GRAPHIC_DIR = "/Users/stefanbechthold/statistik_business_analytics/Grafiken/07_Zeitreihen_und_Forecasting"
GRAPHIC_PATH = os.path.join(GRAPHIC_DIR, "20_zeitreihen_arima_forecast.png")

os.makedirs(GRAPHIC_DIR, exist_ok=True)

def main():
    print("🚀 Starte Griff 20: Time Series Forecasting (Decompose, ADF, ARIMA)...")
    
    # -------------------------------------------------------------------
    # 2. Daten laden & Zeitreihe aufbauen
    # -------------------------------------------------------------------
    print(f"Lade Bestelldaten von: {ORDERS_PATH}")
    df = pd.read_csv(ORDERS_PATH)
    
    # Text-Datum in ein echtes Pandas-Datetime-Objekt umwandeln
    df['purchase_time'] = pd.to_datetime(df['order_purchase_timestamp'])
    
    # Wir filtern die Daten auf das "normale" Geschäftsjahr. 
    # Olist startete 2016 sehr langsam (viele Lücken), Ende 2018 stoppen die Daten abrupt.
    df_clean = df[(df['purchase_time'] >= '2017-01-01') & (df['purchase_time'] <= '2018-08-01')].copy()
    
    # Für Zeitreihen MUSS das Datum der Index (die Zeilenbeschriftung) der Tabelle sein!
    df_clean.set_index('purchase_time', inplace=True)
    
    # Resampling: Wir fassen alle Bestellungen pro WOCHE ('W') zusammen und zählen sie (.size())
    ts_weekly = df_clean.resample('W').size()
    
    print(f"Zeitreihe erstellt: {len(ts_weekly)} Wochen an Daten.")

    # -------------------------------------------------------------------
    # 3. Time Series Decomposition (Zerlegung)
    # -------------------------------------------------------------------
    print("\nFühre Seasonal Decomposition durch...")
    # Wir zerlegen die Reihe. period=4 steht für ein ca. monatliches Muster (4 Wochen).
    # Model 'additive' bedeutet: Wert = Trend + Saison + Zufall
    decomposition = seasonal_decompose(ts_weekly, model='additive', period=4)
    
    trend = decomposition.trend
    seasonal = decomposition.seasonal
    residual = decomposition.resid

    # -------------------------------------------------------------------
    # 4. Stationarität prüfen (ADF-Test)
    # -------------------------------------------------------------------
    print("\nFühre Augmented Dickey-Fuller (ADF) Test durch...")
    # Die Nullhypothese des ADF-Tests ist: "Die Reihe ist NICHT stationär" (hat also einen Trend).
    # Wenn der p-Wert < 0.05 ist, können wir das ablehnen und jubeln: "Sie ist stationär!"
    
    adf_result = adfuller(ts_weekly)
    p_value = adf_result[1]
    print(f"ADF p-Wert (Originaldaten): {p_value:.4f}")
    
    if p_value < 0.05:
        print("-> Daten sind stationär! (Gut für einfache Modelle)")
    else:
        print("-> Daten sind NICHT stationär! (Wir müssen differenzieren / Das 'I' in ARIMA übernimmt das)")

    # Wir berechnen manuell eine "Differenzierung" (Wert von dieser Woche MINUS Wert von letzter Woche),
    # um in der Grafik zu zeigen, wie man Trends mathematisch "tötet" und Stationarität erzwingt.
    ts_differenced = ts_weekly.diff().dropna()
    adf_result_diff = adfuller(ts_differenced)

    # -------------------------------------------------------------------
    # 5. ARIMA Forecast (Modellierung & Vorhersage)
    # -------------------------------------------------------------------
    print("\nTrainiere ARIMA Modell und erstelle Forecast für die nächsten 8 Wochen...")
    
    # Wir trainieren auf allen Daten, die wir haben.
    # order=(p, d, q):
    # p = 1 (Nutze die letzte Woche als Feature)
    # d = 1 (Differenziere 1 Mal, da unsere Daten einen leichten Aufwärtstrend haben)
    # q = 1 (Nutze den letzten Prognosefehler zur Korrektur)
    model = ARIMA(ts_weekly, order=(1, 1, 1))
    fitted_model = model.fit()
    
    # Prognose für die nächsten 8 Wochen in die Zukunft!
    forecast_steps = 8
    forecast = fitted_model.get_forecast(steps=forecast_steps)
    
    # Die Vorhersage-Werte
    predicted_mean = forecast.predicted_mean
    # Die 95% Konfidenzintervalle (Wie sicher ist sich das Modell?)
    conf_int = forecast.conf_int(alpha=0.05)
    
    # Wir extrahieren auch die Fit-Werte auf den Trainingsdaten, um zu sehen, 
    # wie gut das Modell die Vergangenheit "verstanden" hat.
    in_sample_pred = fitted_model.predict(start=ts_weekly.index[1], end=ts_weekly.index[-1])

    # -------------------------------------------------------------------
    # 6. Wunderschöne Visualisierung (Dashboard)
    # -------------------------------------------------------------------
    print("Generiere Visualisierungs-Grid...")
    sns.set_theme(style="whitegrid")
    
    # Wir bauen ein Layout mit 3 Bereichen (1 großer oben, 2 kleine unten)
    fig = plt.figure(figsize=(16, 12))
    ax_forecast = plt.subplot2grid((2, 2), (0, 0), colspan=2)
    ax_decomp = plt.subplot2grid((2, 2), (1, 0))
    ax_adf = plt.subplot2grid((2, 2), (1, 1))

    # --- Plot 1: Der Forecast (Oben) ---
    # Echte historische Daten
    ax_forecast.plot(ts_weekly.index, ts_weekly.values, color='#2c3e50', linewidth=2, label='Echte Bestellungen (Wöchentlich)')
    
    # Modell-Fit auf der Vergangenheit
    ax_forecast.plot(in_sample_pred.index, in_sample_pred.values, color='#3498db', linestyle='--', alpha=0.7, label='ARIMA Fit (Vergangenheit)')
    
    # Der Forecast in die Zukunft (Rot)
    ax_forecast.plot(predicted_mean.index, predicted_mean.values, color='#e74c3c', linewidth=3, label='Forecast (Nächste 8 Wochen)')
    
    # Konfidenzintervall schattieren
    ax_forecast.fill_between(predicted_mean.index, conf_int.iloc[:, 0], conf_int.iloc[:, 1], color='#e74c3c', alpha=0.15, label='95% Konfidenzintervall')
    
    ax_forecast.set_title("1. Business Forecast: Wöchentliches Bestellvolumen (ARIMA Modell)", fontsize=16, fontweight="bold")
    ax_forecast.set_ylabel("Anzahl Bestellungen")
    ax_forecast.legend(loc='upper left')

    # --- Plot 2: Seasonal Decomposition (Unten Links) ---
    # Wir plotten den isolierten Trend und die Saisonalität
    ax_decomp.plot(trend.index, trend.values, color='#e67e22', linewidth=3, label='Trend (Langfristig)')
    # Saisonalität verschieben wir leicht nach oben, um sie im gleichen Plot sichtbar zu machen
    offset = trend.mean()
    ax_decomp.plot(seasonal.index, seasonal.values + offset, color='#2ecc71', linewidth=1.5, alpha=0.8, label='Saisonalität (+ Offset)')
    
    ax_decomp.set_title("2. Decomposition: Trend & Muster isoliert", fontsize=14, fontweight="bold")
    ax_decomp.set_ylabel("Komponenten-Wert")
    ax_decomp.legend(loc='best')

    # --- Plot 3: Differenzierung & ADF-Test (Unten Rechts) ---
    # Zeigt, wie Stationarität aussieht (kein Trend mehr, konstanter Mittelwert um 0)
    ax_adf.plot(ts_differenced.index, ts_differenced.values, color='#9b59b6', linewidth=1.5)
    ax_adf.axhline(0, color='black', linestyle='--', linewidth=1)
    
    ax_adf.set_title("3. Stationarität: Differenzierte Daten (Week over Week)", fontsize=14, fontweight="bold")
    ax_adf.set_ylabel("Veränderung zur Vorwoche")
    
    # Info-Box für die ADF-Testergebnisse in den Plot einfügen
    adf_text = (
        f"ADF-Test Ergebnisse:\n"
        f"------------------------\n"
        f"Original p-Wert: {p_value:.4f}\n"
        f"Differenziert p-Wert: {adf_result_diff[1]:.4e}\n\n"
        f"Fazit: Die einfache Differenzierung\n"
        f"zerstört den Trend und macht die\n"
        f"Daten perfekt stationär für ARIMA!"
    )
    props = dict(boxstyle='round,pad=0.8', facecolor='white', alpha=0.9, edgecolor='gray', linewidth=1.5)
    ax_adf.text(0.05, 0.05, adf_text, transform=ax_adf.transAxes, fontsize=11,
                verticalalignment='bottom', horizontalalignment='left', bbox=props, color='#2c3e50')

    # Layout optimieren und speichern
    plt.suptitle("Zeitreihenanalyse & Forecasting Pipeline", 
                 fontsize=18, fontweight="bold", y=1.03)
    plt.tight_layout()
    sns.despine()
    
    plt.savefig(GRAPHIC_PATH, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✅ Grafik erfolgreich gespeichert unter:\n{GRAPHIC_PATH}")

if __name__ == "__main__":
    main()
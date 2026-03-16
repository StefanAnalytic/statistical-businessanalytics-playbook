"""
Griff 21: Wie bewerte ich meinen Forecast fürs Business? (Forecast-Metriken: MAE, RMSE, MAPE, sMAPE)

Was ist das und was macht das Skript?
In Griff 20 haben wir einen Forecast (Vorhersage) erstellt. Aber wie gut ist er wirklich? 
Dafür gibt es spezielle Fehler-Metriken. Jede hat Vor- und Nachteile fürs Business:

1. MAE (Mean Absolute Error): "Im Schnitt liegen wir um X Pakete daneben." Sehr intuitiv!
2. RMSE (Root Mean Squared Error): Bestraft große Ausreißer extrem hart. Wichtig, wenn 
   große Fehler viel teurer sind als viele kleine Fehler (z.B. bei verderblicher Ware).
3. MAPE (Mean Absolute Percentage Error): "Wir liegen im Schnitt um X% daneben." 
   Das versteht jeder Manager! Problem: Er explodiert, wenn der echte Wert nahe 0 ist.
4. sMAPE (Symmetric MAPE): Behebt das MAPE-Problem teilweise, indem er den Fehler 
   symmetrisch auf Vorhersage und echten Wert bezieht. Oft der Standard in Forecasting-Wettbewerben.

Business Case in diesem Skript:
Wir simulieren einen Forecast für wöchentliche Bestellungen. Dann berechnen wir alle 4 Metriken 
und visualisieren, wie unterschiedlich sie den Vorhersagefehler "bestrafen".
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# -------------------------------------------------------------------
# 1. Setup & Pfade
# -------------------------------------------------------------------
# Ordner: 07_Zeitreihen_und_Forecasting
ORDERS_PATH = "/Users/stefanbechthold/statistik_business_analytics/data/olist_orders_dataset.csv"
GRAPHIC_DIR = "/Users/stefanbechthold/statistik_business_analytics/Grafiken/07_Zeitreihen_und_Forecasting"
GRAPHIC_PATH = os.path.join(GRAPHIC_DIR, "21_forecast_metriken.png")

os.makedirs(GRAPHIC_DIR, exist_ok=True)

def main():
    print("🚀 Starte Griff 21: Forecast-Metriken (MAE, RMSE, MAPE, sMAPE)...")
    
    # -------------------------------------------------------------------
    # 2. Reale Daten laden & Zeitreihe aufbauen
    # -------------------------------------------------------------------
    print("Lade Bestelldaten und baue wöchentliche Zeitreihe...")
    df = pd.read_csv(ORDERS_PATH)
    df['purchase_time'] = pd.to_datetime(df['order_purchase_timestamp'])
    
    # Wir filtern ein stabiles Zeitfenster (2017 bis Mitte 2018)
    df_clean = df[(df['purchase_time'] >= '2017-03-01') & (df['purchase_time'] <= '2018-07-01')].copy()
    df_clean.set_index('purchase_time', inplace=True)
    
    # Wöchentliche Bestellungen zählen
    ts_weekly = df_clean.resample('W').size()
    
    # Wir nehmen die letzten 20 Wochen als unsere "Test-Periode"
    test_actuals = ts_weekly[-20:]
    
    # Wir simulieren einen leicht fehlerhaften Forecast für diese 20 Wochen.
    # Er folgt dem Trend, verschätzt sich aber ab und zu (besonders stark bei einem Peak).
    np.random.seed(42)
    # Ein bisschen Rauschen hinzufügen + systematisch bei extremen Werten danebenliegen
    test_forecast = test_actuals * 0.95 + np.random.normal(0, 150, len(test_actuals))
    test_forecast = np.maximum(test_forecast, 0) # Keine negativen Vorhersagen erlauben
    
    # Wir fügen absichtlich EINEN gigantischen Fehler ein, um zu zeigen, wie der RMSE explodiert.
    test_forecast.iloc[10] = test_actuals.iloc[10] - 800

    print(f"Analysiere {len(test_actuals)} Wochen Forecast-Daten...")

    # -------------------------------------------------------------------
    # 3. Metriken berechnen (Die mathematischen Formeln)
    # -------------------------------------------------------------------
    print("\nBerechne Business-Metriken...")
    
    y_true = test_actuals.values
    y_pred = test_forecast.values
    
    # Die absoluten Fehler pro Woche
    errors = y_true - y_pred
    abs_errors = np.abs(errors)
    squared_errors = errors ** 2
    
    # 1. MAE (Mean Absolute Error)
    mae = np.mean(abs_errors)
    
    # 2. RMSE (Root Mean Squared Error)
    rmse = np.sqrt(np.mean(squared_errors))
    
    # 3. MAPE (Mean Absolute Percentage Error) - In Prozent
    # Achtung: Wenn y_true == 0 ist, teilt man durch Null! (Hier nicht der Fall, da >0 Bestellungen)
    mape = np.mean(abs_errors / y_true) * 100
    
    # 4. sMAPE (Symmetric MAPE) - In Prozent
    smape = np.mean(2.0 * abs_errors / (np.abs(y_true) + np.abs(y_pred))) * 100
    
    print("-" * 40)
    print(f"MAE:   {mae:.1f} Bestellungen Fehl-Differenz")
    print(f"RMSE:  {rmse:.1f} Bestellungen (stark gewichtet durch den großen Peak-Fehler!)")
    print(f"MAPE:  {mape:.1f} %")
    print(f"sMAPE: {smape:.1f} %")
    print("-" * 40)

    # -------------------------------------------------------------------
    # 4. Wunderschöne Visualisierung (Dashboard)
    # -------------------------------------------------------------------
    print("\nGeneriere Visualisierungs-Grid...")
    sns.set_theme(style="whitegrid")
    
    fig = plt.figure(figsize=(16, 10))
    
    # Grid Layout: Großes Chart oben (Zeitreihe), unten links Absolut, unten rechts Prozentual
    ax_ts = plt.subplot2grid((2, 2), (0, 0), colspan=2)
    ax_abs = plt.subplot2grid((2, 2), (1, 0))
    ax_pct = plt.subplot2grid((2, 2), (1, 1))

    # --- Plot 1: Zeitreihe (Actuals vs. Forecast) ---
    ax_ts.plot(test_actuals.index, y_true, marker='o', color='#2c3e50', linewidth=2, label='Echte Bestellungen (Actuals)')
    ax_ts.plot(test_actuals.index, y_pred, marker='x', color='#e74c3c', linewidth=2, linestyle='--', label='Vorhersage (Forecast)')
    
    # Fehler grafisch als vertikale Linien (Residuen) darstellen
    for idx, (t, p) in zip(test_actuals.index, zip(y_true, y_pred)):
        ax_ts.plot([idx, idx], [t, p], color='gray', linestyle=':', linewidth=1.5, alpha=0.7)
        
    ax_ts.set_title("1. Forecast-Bewertung: Wie weit weicht die Vorhersage von der Realität ab?", fontsize=14, fontweight="bold")
    ax_ts.set_ylabel("Anzahl Bestellungen")
    ax_ts.legend()

    # --- Plot 2: Absolute Fehler-Bestrafung (MAE vs RMSE) ---
    # Wir zeigen die Fehler für jede Woche. MAE sieht den Fehler linear, RMSE quadriert ihn.
    weeks = np.arange(1, len(y_true) + 1)
    width = 0.35
    
    ax_abs.bar(weeks - width/2, abs_errors, width, color='#3498db', alpha=0.8, label='Absoluter Fehler (für MAE)')
    # Wir nehmen die Wurzel der Squared Errors (was genau der Fehler ist), 
    # aber um die Bestrafung zu zeigen, plotten wir den prozentualen Anteil am RMSE.
    # Alternativ: Wir zeigen, dass Woche 10 (der Ausreißer) den RMSE massiv treibt.
    rmse_penalties = np.sqrt(squared_errors) 
    ax_abs.bar(weeks + width/2, squared_errors / np.max(squared_errors) * np.max(abs_errors), width, 
               color='#9b59b6', alpha=0.8, label='Quadrierte Bestrafung (für RMSE skaliert)')
    
    ax_abs.set_title("2. MAE vs. RMSE: Ausreißer-Bestrafung", fontsize=14, fontweight="bold")
    ax_abs.set_xlabel("Woche im Test-Set")
    ax_abs.set_ylabel("Fehler-Größe")
    ax_abs.legend()
    
    # Text in Plot 2
    ax_abs.text(0.05, 0.85, f"MAE: {mae:.0f}\nRMSE: {rmse:.0f}\n\nDer RMSE ist höher,\nweil er den riesigen Fehler\nin Woche 11 überproportional\nbestraft!", 
                transform=ax_abs.transAxes, fontsize=11, bbox=dict(facecolor='white', alpha=0.9, edgecolor='gray'))

    # --- Plot 3: Prozentuale Fehler (MAPE vs sMAPE) ---
    # Fehler in Prozent pro Woche
    mape_weekly = (abs_errors / y_true) * 100
    smape_weekly = (2.0 * abs_errors / (np.abs(y_true) + np.abs(y_pred))) * 100
    
    ax_pct.plot(weeks, mape_weekly, marker='o', color='#27ae60', linewidth=2, label='MAPE (%)')
    ax_pct.plot(weeks, smape_weekly, marker='s', color='#f39c12', linewidth=2, linestyle='--', label='sMAPE (%)')
    
    ax_pct.set_title("3. MAPE vs. sMAPE: Prozentuale Abweichung", fontsize=14, fontweight="bold")
    ax_pct.set_xlabel("Woche im Test-Set")
    ax_pct.set_ylabel("Fehler in %")
    ax_pct.legend()
    
    # Text in Plot 3
    ax_pct.text(0.05, 0.85, f"MAPE: {mape:.1f}%\nsMAPE: {smape:.1f}%\n\nsMAPE pendelt sich oft\netwas niedriger ein und ist\nrobuster, wenn reale Werte\nsehr klein werden.", 
                transform=ax_pct.transAxes, fontsize=11, bbox=dict(facecolor='white', alpha=0.9, edgecolor='gray'))

    # Layout optimieren und speichern
    plt.suptitle("Forecast Evaluation: Die richtige Metrik für den Business Case", 
                 fontsize=18, fontweight="bold", y=1.03)
    plt.tight_layout()
    sns.despine()
    
    plt.savefig(GRAPHIC_PATH, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✅ Grafik erfolgreich gespeichert unter:\n{GRAPHIC_PATH}")

if __name__ == "__main__":
    main()
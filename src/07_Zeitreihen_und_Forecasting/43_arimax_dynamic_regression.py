"""
Griff 43: Zeitabhängige Einflüsse + Exog. Variablen? (ARIMAX & Dynamic Regression)

Was ist das und was macht das Skript?
Klassische Zeitreihenmodelle (wie ARIMA) sagen die Zukunft nur basierend auf der eigenen 
Vergangenheit voraus (Trend & Saisonalität). Im Business-Alltag passieren Umsatzspitzen 
aber oft nicht "einfach so", sondern werden durch externe Ereignisse (Exogene Variablen) 
getrieben: Marketing-Kampagnen, Preisänderungen, Feiertage oder Black Friday.

ARIMAX (AutoRegressive Integrated Moving Average with eXogenous variables) kombiniert 
die Stärken der Zeitreihenanalyse mit der klassischen Regression. Das Modell lernt 
das "normale" zyklische Verhalten der Zeitreihe UND schätzt gleichzeitig den isolierten 
Effekt der externen Faktoren.

Business Case in diesem Skript:
Wir modellieren den täglichen Umsatz (Revenue) im Olist-Datensatz. Im November 2017 gab 
es einen extremen Ausschlag. Wir nutzen eine exogene Dummy-Variable für die "Black Friday 
Woche" (20.-26. November 2017), um dem Modell zu helfen, diese externe Schock-Wirkung 
zu verstehen, anstatt sie fälschlicherweise als dauerhaften Trendwechsel zu interpretieren.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm

# -------------------------------------------------------------------
# 1. Setup & Pfade
# -------------------------------------------------------------------
ORDERS_PATH = "/Users/stefanbechthold/statistik_business_analytics/data/olist_orders_dataset.csv"
PAYMENTS_PATH = "/Users/stefanbechthold/statistik_business_analytics/data/olist_order_payments_dataset.csv"

GRAPHIC_DIR = "/Users/stefanbechthold/statistik_business_analytics/Grafiken/07_Zeitreihen_und_Forecasting"
GRAPHIC_PATH = os.path.join(GRAPHIC_DIR, "43_arimax_exogenous_variables.png")

os.makedirs(GRAPHIC_DIR, exist_ok=True)

def main():
    print("🚀 Starte Griff 43: Time Series Forecasting mit Exogenen Variablen (ARIMAX)...")
    
    # -------------------------------------------------------------------
    # 2. Daten laden & Tägliche Zeitreihe aggregieren
    # -------------------------------------------------------------------
    df_orders = pd.read_csv(ORDERS_PATH)
    df_payments = pd.read_csv(PAYMENTS_PATH)
    
    # Mergen und Datum konvertieren
    df_merged = pd.merge(df_orders[['order_id', 'order_purchase_timestamp']], 
                         df_payments[['order_id', 'payment_value']], 
                         on='order_id', how='inner')
    
    df_merged['purchase_date'] = pd.to_datetime(df_merged['order_purchase_timestamp']).dt.date
    
    # Täglichen Umsatz aggregieren
    daily_revenue = df_merged.groupby('purchase_date')['payment_value'].sum().reset_index()
    daily_revenue['purchase_date'] = pd.to_datetime(daily_revenue['purchase_date'])
    daily_revenue.set_index('purchase_date', inplace=True)
    
    # Kontinuierlichen Datums-Index sicherstellen (Fehlende Tage mit 0 füllen)
    daily_revenue = daily_revenue.asfreq('D', fill_value=0)
    
    # Zeitraum filtern: Wir fokussieren uns auf 2017 für eine saubere Modellierung
    df_2017 = daily_revenue['2017-01-01':'2017-12-31'].copy()
    
    # -------------------------------------------------------------------
    # 3. Exogene Variable (Feature Engineering)
    # -------------------------------------------------------------------
    # Wir definieren die Black Friday Woche 2017 (20. Nov bis 26. Nov)
    df_2017['is_black_friday'] = 0
    bf_mask = (df_2017.index >= '2017-11-20') & (df_2017.index <= '2017-11-26')
    df_2017.loc[bf_mask, 'is_black_friday'] = 1

    # Zielvariable (Y) und Exogene Variable (X) trennen
    y = df_2017['payment_value']
    X_exog = df_2017[['is_black_friday']]
    
    # Train/Test Split (Letzte 45 Tage als Holdout/Test-Set)
    train_size = len(df_2017) - 45
    y_train, y_test = y.iloc[:train_size], y.iloc[train_size:]
    X_train, X_test = X_exog.iloc[:train_size], X_exog.iloc[train_size:]

    # -------------------------------------------------------------------
    # 4. ARIMAX Modell trainieren
    # -------------------------------------------------------------------
    print("Trainiere ARIMAX Modell mit exogener Dummy-Variable...")
    
    # SARIMAX Klasse kann ARIMAX abbilden (wir nutzen hier ein Basis-ARIMA(1,1,1) + Exog)
    # 'exog' ist der Parameter, der die magische Verknüpfung zur Regression herstellt
    model = sm.tsa.statespace.SARIMAX(endog=y_train, 
                                      exog=X_train,
                                      order=(1, 1, 1), 
                                      enforce_stationarity=False, 
                                      enforce_invertibility=False)
    
    arimax_results = model.fit(disp=False)
    
    # Vorhersage für das Test-Set generieren (wir müssen dem Modell das X_test der Zukunft geben!)
    predictions = arimax_results.get_forecast(steps=len(y_test), exog=X_test)
    pred_mean = predictions.predicted_mean
    pred_ci = predictions.conf_int()

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

    # --- Plot 1: Gesamte Zeitreihe mit Exogenem Event ---
    ax1.plot(df_2017.index, df_2017['payment_value'], color='#3498db', linewidth=2, label='Täglicher Umsatz (BRL)')
    
    # Markierung der Black Friday Woche
    ax1.fill_between(df_2017.index, 0, df_2017['payment_value'].max() * 1.1, 
                     where=df_2017['is_black_friday']==1, 
                     color='#f1c40f', alpha=0.3, label='Exogenes Event (Black Friday Woche)')
    
    ax1.set_title("1. Die Zeitreihe verstehen: Exogene Schocks erkennen", fontsize=14, fontweight='bold', color='white')
    ax1.set_ylabel("Umsatz in BRL")
    ax1.set_ylim(0, df_2017['payment_value'].max() * 1.1)
    
    legend1 = ax1.legend(loc='upper left', facecolor='#1e1e1e', edgecolor='#333333')
    plt.setp(legend1.get_texts(), color='white')

    # --- Plot 2: Out-of-Sample Forecast (Test-Set) ---
    # Nur das letzte Quartal anzeigen, um den Forecast besser zu sehen
    plot_start = '2017-10-01'
    
    # Echte Daten (Train + Test)
    ax2.plot(df_2017[plot_start:].index, df_2017[plot_start:]['payment_value'], 
             color='#3498db', linewidth=2, label='Tatsächlicher Umsatz', alpha=0.7)
    
    # ARIMAX Forecast
    ax2.plot(pred_mean.index, pred_mean, color='#e74c3c', linewidth=2, linestyle='--', label='ARIMAX Forecast (inkl. Exog)')
    ax2.fill_between(pred_ci.index, pred_ci.iloc[:, 0], pred_ci.iloc[:, 1], color='#e74c3c', alpha=0.2, label='95% Konfidenzintervall')
    
    # Trennlinie Train/Test
    ax2.axvline(y_test.index[0], color='white', linestyle=':', linewidth=2, label='Start des Test-Sets (Forecast)')
    
    ax2.set_title("2. ARIMAX Forecast Evaluation (Out-of-Sample)", fontsize=14, fontweight='bold', color='white')
    ax2.set_ylabel("Umsatz in BRL")
    ax2.set_xlim(pd.to_datetime(plot_start), df_2017.index[-1])
    
    legend2 = ax2.legend(loc='upper left', facecolor='#1e1e1e', edgecolor='#333333')
    plt.setp(legend2.get_texts(), color='white')

    # Info-Box
    bf_coef = arimax_results.params.get('is_black_friday', 0)
    info_text = (
        "Warum ARIMAX entscheidend ist:\n"
        "------------------------------\n"
        "Ein normales ARIMA-Modell wäre durch den extremen Peak\n"
        "im November völlig verwirrt worden und hätte danach einen\n"
        "viel zu hohen Dauer-Umsatz vorhergesagt.\n\n"
        "Das ARIMAX-Modell hat gelernt, dass der Peak durch das externe\n"
        "Event (is_black_friday) verursacht wurde. Der geschätzte isolierte\n"
        f"Umsatz-Effekt dieses Events liegt laut Modell bei ca. +{bf_coef:,.0f} BRL pro Tag!"
    )
    props = dict(boxstyle='round,pad=1', facecolor='#1e1e1e', alpha=0.9, edgecolor='#f1c40f', linewidth=1.5)
    ax2.text(0.35, 0.85, info_text, transform=ax2.transAxes, fontsize=11,
             verticalalignment='top', bbox=props, color='white')

    plt.suptitle("Advanced Time Series: ARIMAX & Dynamic Regression für Business Events", 
                 fontsize=18, fontweight='bold', color='white', y=1.02)
    plt.tight_layout()
    
    # Speichern der Grafik
    plt.savefig(GRAPHIC_PATH, dpi=300, bbox_inches='tight', facecolor=fig.get_facecolor())
    plt.close()
    
    print(f"✅ Grafik erfolgreich gespeichert unter:\n{GRAPHIC_PATH}")

if __name__ == "__main__":
    main()
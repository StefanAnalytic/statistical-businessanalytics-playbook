"""
Griff 50: Plötzliche Strukturänderungen finden? (Change-Point Detection: CUSUM)

Was ist das und was macht das Skript?
In der Überwachung von Business-KPIs (Monitoring) suchen wir nicht nur nach einzelnen 
Ausreißern (Anomalien), sondern nach dauerhaften Strukturbrüchen ("Regime Shifts"). 
Beispiel: Wann genau hat sich die Basis-Conversion-Rate nach einem Website-Relaunch 
verschlechtert? Ein einfaches Dashboard zeigt viel Rauschen, und das menschliche Auge 
lässt sich leicht täuschen.

Der CUSUM-Algorithmus (Cumulative Sum) löst dieses Problem. Er summiert kleine 
Abweichungen vom Erwartungswert über die Zeit auf. Sobald diese kumulierte Summe 
einen definierten Schwellenwert (Threshold) überschreitet, schlägt das System Alarm. 
Er ist extrem sensibel für kleine, aber anhaltende Veränderungen in Datenströmen.

Neuer Business Case in diesem Skript:
Wir überwachen Stornierungen (Cancellations) im Olist-Marktplatz. 
Ein Anstieg der Storno-Rate kann auf schwerwiegende Plattform-Bugs, Zahlungsprobleme 
oder Lieferengpässe hindeuten. Wir extrahieren die täglichen Stornierungen und nutzen 
CUSUM, um den exakten Tag zu finden, an dem das Storno-Level strukturell nach oben kippt, 
um das Operations-Team automatisch zu alarmieren.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# -------------------------------------------------------------------
# 1. Setup & Pfade
# -------------------------------------------------------------------
ORDERS_PATH = "/Users/stefanbechthold/statistik_business_analytics/data/olist_orders_dataset.csv"

GRAPHIC_DIR = "/Users/stefanbechthold/statistik_business_analytics/Grafiken/13_Monitoring_MLOps_und_Governance"
GRAPHIC_PATH = os.path.join(GRAPHIC_DIR, "50_cusum_change_point_detection.png")

os.makedirs(GRAPHIC_DIR, exist_ok=True)

def cusum_detection(series, target_mean, drift_allowance=0.5, threshold=5.0):
    """
    Simpler CUSUM (Cumulative Sum) Algorithmus für einen Aufwärts-Trend.
    series: Die Zeitreihe
    target_mean: Der Basis-Erwartungswert (Normalzustand)
    drift_allowance: Wie viel Rauschen tolerieren wir (Slack)
    threshold: Ab welcher kumulierten Summe gibt es einen Alarm?
    """
    cusum_pos = np.zeros(len(series))
    alarms = []
    
    for i in range(1, len(series)):
        # Wir addieren die Abweichung vom Mittelwert. 
        # Wenn der Wert wieder sinkt, fällt CUSUM zurück auf maximal 0 (max-Funktion).
        deviation = series.iloc[i] - target_mean - drift_allowance
        cusum_pos[i] = max(0, cusum_pos[i-1] + deviation)
        
        if cusum_pos[i] > threshold:
            alarms.append(series.index[i])
            
    return cusum_pos, alarms

def main():
    print("🚀 Starte Griff 50: Change-Point Detection für Stornierungen (CUSUM)...")
    
    # -------------------------------------------------------------------
    # 2. Daten laden & Storno-Zeitreihe aufbauen
    # -------------------------------------------------------------------
    df_orders = pd.read_csv(ORDERS_PATH)
    
    df_orders['purchase_date'] = pd.to_datetime(df_orders['order_purchase_timestamp']).dt.date
    
    # Wir filtern nur stornierte Bestellungen
    df_canceled = df_orders[df_orders['order_status'] == 'canceled'].copy()
    
    # Tägliche Stornierungen zählen
    daily_cancels = df_canceled.groupby('purchase_date').size().reset_index(name='cancel_count')
    daily_cancels['purchase_date'] = pd.to_datetime(daily_cancels['purchase_date'])
    daily_cancels.set_index('purchase_date', inplace=True)
    
    # Auffüllen fehlender Tage mit 0
    daily_cancels = daily_cancels.asfreq('D', fill_value=0)
    
    # Fokus auf einen spannenden Zeitraum: Anfang 2018 gab es bei Olist strukturelle Veränderungen
    ts_data = daily_cancels.loc['2018-01-01':'2018-04-30'].copy()

    # -------------------------------------------------------------------
    # 3. CUSUM Algorithmus anwenden
    # -------------------------------------------------------------------
    # Wir nehmen den Januar 2018 als unsere "gesunde" Baseline (Referenzzeitraum)
    baseline_period = ts_data.loc['2018-01-01':'2018-01-31']['cancel_count']
    target_mean = baseline_period.mean()
    target_std = baseline_period.std()
    
    # Parameter für CUSUM einstellen (Abhängig von der Metrik-Skala)
    # allowance = 0.5 Standardabweichungen (erlaubtes Rauschen)
    # threshold = 3 Standardabweichungen (Alarm-Grenze)
    allowance = 0.5 * target_std
    threshold = 3.0 * target_std
    
    # CUSUM berechnen
    cusum_scores, alarm_dates = cusum_detection(ts_data['cancel_count'], 
                                                target_mean=target_mean, 
                                                drift_allowance=allowance, 
                                                threshold=threshold)
    
    ts_data['CUSUM_Score'] = cusum_scores
    
    # Der Change-Point ist der erste Tag, an dem der Alarm ausgelöst wird
    first_change_point = alarm_dates[0] if alarm_dates else None

    # -------------------------------------------------------------------
    # 4. Visualisierung im Dark Mode Design
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
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 12), facecolor='#121212', sharex=True)

    # --- Plot 1: Rohdaten (Tägliche Stornierungen) ---
    ax1.plot(ts_data.index, ts_data['cancel_count'], color='#3498db', marker='o', alpha=0.7, label='Tägliche Stornierungen')
    
    # Baseline einzeichnen
    ax1.axhline(target_mean, color='#2ecc71', linestyle='--', linewidth=2, label=f'Baseline Mean ({target_mean:.1f} / Tag)')
    ax1.axvspan(pd.to_datetime('2018-01-01'), pd.to_datetime('2018-01-31'), color='#2ecc71', alpha=0.1, label='Baseline-Periode (Gesund)')
    
    # Change Point markieren
    if first_change_point:
        ax1.axvline(first_change_point, color='#e74c3c', linestyle=':', linewidth=3, label=f'Regime Shift erkannt ({first_change_point.strftime("%Y-%m-%d")})')

    ax1.set_title("1. Business KPI: Tägliches Storno-Volumen (Rauschende Rohdaten)", fontsize=14, fontweight='bold', color='white')
    ax1.set_ylabel("Anzahl Stornierungen")
    
    legend1 = ax1.legend(loc='upper left', facecolor='#1e1e1e', edgecolor='#333333')
    plt.setp(legend1.get_texts(), color='white')

    # --- Plot 2: Der CUSUM Score (Das Alarm-System) ---
    ax2.plot(ts_data.index, ts_data['CUSUM_Score'], color='#f1c40f', linewidth=3, label='CUSUM Metrik (Kumulierte Abweichung)')
    
    # Threshold Linie
    ax2.axhline(threshold, color='#e74c3c', linestyle='--', linewidth=2, label=f'Alarm Threshold (H = {threshold:.1f})')
    
    # Farbige Füllung, wenn Threshold überschritten wird
    ax2.fill_between(ts_data.index, threshold, ts_data['CUSUM_Score'], 
                     where=(ts_data['CUSUM_Score'] > threshold), color='#e74c3c', alpha=0.3)
    
    if first_change_point:
        ax2.axvline(first_change_point, color='#e74c3c', linestyle=':', linewidth=3)
        ax2.scatter(first_change_point, ts_data.loc[first_change_point, 'CUSUM_Score'], color='#e74c3c', s=150, zorder=5)

    ax2.set_title("2. CUSUM Control Chart: Kumulierte Summe deckt schleichende Trends auf", fontsize=14, fontweight='bold', color='white')
    ax2.set_ylabel("CUSUM Score")
    ax2.set_xlabel("Datum")
    
    legend2 = ax2.legend(loc='upper left', facecolor='#1e1e1e', edgecolor='#333333')
    plt.setp(legend2.get_texts(), color='white')

    # Info-Box
    info_text = (
        "Die CUSUM-Logik im MLOps/Monitoring:\n"
        "------------------------------------\n"
        "Ein einzelner Tag mit 15 Stornos (wie Mitte Februar) ist nur ein Ausreißer.\n"
        "CUSUM vergisst ihn sofort wieder (Score fällt auf 0).\n\n"
        "Aber ab Ende Februar häufen sich leicht erhöhte Storno-Zahlen an.\n"
        "Das menschliche Auge sieht in Plot 1 nur das übliche Rauschen, aber\n"
        "die kumulierte Summe in Plot 2 schießt gnadenlos nach oben und\n"
        f"löst am {first_change_point.strftime('%d.%m.%Y')} den Alarm aus. Die Struktur hat sich geändert!"
    )
    props = dict(boxstyle='round,pad=1', facecolor='#1e1e1e', alpha=0.9, edgecolor='#e74c3c', linewidth=1.5)
    ax2.text(0.4, 0.4, info_text, transform=ax2.transAxes, fontsize=11,
             verticalalignment='center', bbox=props, color='white')

    plt.suptitle("Data Quality & Pipeline Monitoring: Change-Point Detection", 
                 fontsize=18, fontweight='bold', color='white', y=1.02)
    plt.tight_layout()
    
    # Speichern
    plt.savefig(GRAPHIC_PATH, dpi=300, bbox_inches='tight', facecolor=fig.get_facecolor())
    plt.close()
    
    print(f"✅ Grafik erfolgreich gespeichert unter:\n{GRAPHIC_PATH}")

if __name__ == "__main__":
    main()
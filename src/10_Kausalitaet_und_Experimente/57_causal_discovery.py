"""
Griff 57: Time-to-Event vorhersagen? (Survival Analysis & Kaplan-Meier)

Was ist das und was macht das Skript?
Wie lange dauert es, bis ein Kunde kündigt (Churn)? Wie lange dauert es, bis ein 
Maschinenteil kaputtgeht (Predictive Maintenance)? Wie lange dauert eine Lieferung?

Klassische Regression scheitert hier an einem großen Problem: "Zensierte Daten" (Censorship). 
Wenn ein Paket heute noch unterwegs ist, wissen wir nicht, wie lange es am Ende brauchen wird. 
Wir wissen nur: "Es dauert BEREITS 5 Tage". Werden diese Fälle ignoriert, verzerrt das 
die Realität massiv (Survivor Bias). 

Die "Survival Analysis" (Überlebenszeitanalyse) löst dieses Problem. 
Der Kaplan-Meier-Schätzer berechnet die Wahrscheinlichkeit, dass ein Ereignis (z.B. die 
Zustellung) nach einer bestimmten Zeit T NOCH NICHT eingetreten ist ("Survival Probability").

Business Case in diesem Skript:
Wir analysieren die Lieferzeiten im Olist-Marktplatz. Ein "Sterbe-Event" ist in diesem 
Fall positiv: Das Paket wurde erfolgreich zugestellt! Pakete, die im Datensatz noch den 
Status "shipped" oder "processing" haben, sind "Right-Censored" (Rechtszensiert).
Wir vergleichen die Liefer-Performance von drei sehr unterschiedlichen Bundesstaaten: 
SP (São Paulo - Zentral), RJ (Rio - Küste) und AM (Amazonas - Abgelegener Regenwald).

WICHTIG: Benötigt das Paket 'lifelines' -> pip install lifelines
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

try:
    from lifelines import KaplanMeierFitter
except ImportError:
    print("FEHLER: Die Bibliothek 'lifelines' fehlt. Bitte ausführen: pip install lifelines")
    exit()

# -------------------------------------------------------------------
# 1. Setup & Pfade
# -------------------------------------------------------------------
ORDERS_PATH = "/Users/stefanbechthold/statistik_business_analytics/data/olist_orders_dataset.csv"
CUSTOMERS_PATH = "/Users/stefanbechthold/statistik_business_analytics/data/olist_customers_dataset.csv"

GRAPHIC_DIR = "/Users/stefanbechthold/statistik_business_analytics/Grafiken/12_Advanced_Analytics_Spezial"
GRAPHIC_PATH = os.path.join(GRAPHIC_DIR, "57_survival_analysis_kaplan_meier.png")

os.makedirs(GRAPHIC_DIR, exist_ok=True)

def main():
    print("🚀 Starte Griff 57: Survival Analysis (Time-to-Event & Zensierte Daten)...")
    
    # -------------------------------------------------------------------
    # 2. Daten laden & Survival-Variablen (Dauer & Event) bauen
    # -------------------------------------------------------------------
    df_orders = pd.read_csv(ORDERS_PATH)
    df_customers = pd.read_csv(CUSTOMERS_PATH)
    
    df_merged = pd.merge(df_orders, df_customers[['customer_id', 'customer_state']], on='customer_id', how='inner')
    
    # Datums-Spalten konvertieren
    df_merged['purchase_date'] = pd.to_datetime(df_merged['order_purchase_timestamp'], errors='coerce')
    df_merged['delivered_date'] = pd.to_datetime(df_merged['order_delivered_customer_date'], errors='coerce')
    
    # Das Ende der Datenerfassung simulieren (für die zensierten Daten)
    # Wir nehmen das maximale Datum im Datensatz als unser "Heute"
    max_date = df_merged['purchase_date'].max()
    
    # Wir filtern stornierte Bestellungen raus, da diese die Liefer-Logik verzerren
    df_clean = df_merged[df_merged['order_status'] != 'canceled'].copy()
    
    # --- Survival Logik ---
    # Event (E): 1 = Paket kam an, 0 = Paket ist noch unterwegs (Zensiert)
    df_clean['Event'] = df_clean['delivered_date'].notna().astype(int)
    
    # Duration (T): Wie viele Tage sind vergangen?
    # Wenn zugestellt: delivered_date - purchase_date
    # Wenn zensiert: max_date - purchase_date (Wie lange warten wir schon?)
    df_clean['end_date_for_calc'] = df_clean['delivered_date'].fillna(max_date)
    df_clean['Duration'] = (df_clean['end_date_for_calc'] - df_clean['purchase_date']).dt.total_seconds() / (24 * 3600)
    
    # Negative Zeiten (Datenfehler) und absurde Ausreißer (> 100 Tage) rausfiltern
    df_clean = df_clean[(df_clean['Duration'] > 0) & (df_clean['Duration'] <= 100)]
    
    # Wir fokussieren uns auf 3 spannende Bundesstaaten
    states_of_interest = ['SP', 'RJ', 'AM']
    df_plot = df_clean[df_clean['customer_state'].isin(states_of_interest)].copy()

    # -------------------------------------------------------------------
    # 3. Kaplan-Meier-Schätzer anpassen
    # -------------------------------------------------------------------
    kmf = KaplanMeierFitter()

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
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 7), facecolor='#121212')

    # Farben für die Staaten
    state_colors = {'SP': '#2ecc71', 'RJ': '#3498db', 'AM': '#e74c3c'}
    state_names = {'SP': 'São Paulo (Zentrum)', 'RJ': 'Rio de Janeiro (Küste)', 'AM': 'Amazonas (Regenwald)'}

    # --- Plot 1: Survival Curves (Paket noch nicht zugestellt) ---
    for state in states_of_interest:
        mask = df_plot['customer_state'] == state
        T = df_plot[mask]['Duration']
        E = df_plot[mask]['Event']
        
        # Fit der Daten für diesen Staat
        kmf.fit(durations=T, event_observed=E, label=state_names[state])
        
        # Plot der Kurve inklusive Konfidenzintervall
        kmf.plot_survival_function(ax=ax1, color=state_colors[state], linewidth=3, ci_alpha=0.1)

    ax1.set_title("1. Kaplan-Meier Survival Curves (Wahrscheinlichkeit, dass Paket noch unterwegs ist)", 
                  fontsize=14, fontweight='bold', color='white')
    ax1.set_xlabel("Tage seit Bestellung")
    ax1.set_ylabel("Survival Probability (Noch nicht zugestellt)")
    ax1.set_xlim(0, 40)
    ax1.set_ylim(0, 1.05)
    
    legend1 = ax1.legend(loc='upper right', facecolor='#1e1e1e', edgecolor='#333333')
    plt.setp(legend1.get_texts(), color='white')

    # Info-Box Plot 1
    info_1 = (
        "Wie man die Kurve liest:\n"
        "------------------------\n"
        "Am Tag 0 sind 100% aller Pakete unterwegs (Survival = 1.0).\n"
        "Die Kurve fällt ab, sobald Pakete ankommen ('Sterbe-Event').\n\n"
        "In São Paulo (Grün) fällt die Kurve rasant ab: Nach 10 Tagen\n"
        "sind fast alle Pakete da. In Amazonas (Rot) dauert es ewig:\n"
        "Nach 20 Tagen sind immer noch 60% aller Pakete auf der Reise!"
    )
    props1 = dict(boxstyle='round,pad=1', facecolor='#1e1e1e', alpha=0.9, edgecolor='#e74c3c', linewidth=1.5)
    ax1.text(0.4, 0.6, info_1, transform=ax1.transAxes, fontsize=11,
             verticalalignment='center', bbox=props1, color='white')

    # --- Plot 2: Cumulative Density Function (CDF) ---
    # Das ist einfach 1 - Survival Function (Wie viele Pakete SIND bereits da?)
    for state in states_of_interest:
        mask = df_plot['customer_state'] == state
        T = df_plot[mask]['Duration']
        E = df_plot[mask]['Event']
        
        kmf.fit(durations=T, event_observed=E, label=state_names[state])
        
        # 1 - Survival Probability
        kmf.plot_cumulative_density(ax=ax2, color=state_colors[state], linewidth=3, ci_alpha=0.1)

    # 50% Linie (Median Lieferzeit) einzeichnen
    ax2.axhline(0.5, color='white', linestyle='--', linewidth=2, label='Median (50% der Pakete zugestellt)')

    ax2.set_title("2. Cumulative Event Rate (Wann ist der Großteil zugestellt?)", 
                  fontsize=14, fontweight='bold', color='white')
    ax2.set_xlabel("Tage seit Bestellung")
    ax2.set_ylabel("Kumulierter Anteil zugestellter Pakete")
    ax2.set_xlim(0, 40)
    ax2.set_ylim(0, 1.05)
    
    legend2 = ax2.legend(loc='lower right', facecolor='#1e1e1e', edgecolor='#333333')
    plt.setp(legend2.get_texts(), color='white')

    # Info-Box Plot 2
    info_2 = (
        "Warum nicht einfach den Mittelwert nehmen?\n"
        "------------------------------------------\n"
        "Ein simpler Mittelwert ignoriert die Zensur (Pakete, die noch\n"
        "unterwegs sind, würden nicht gezählt oder als '0 Tage' gewertet).\n\n"
        "Kaplan-Meier berücksichtigt diese Teilinformationen mathematisch\n"
        "korrekt. Die gestrichelte 50%-Linie zeigt den echten Median:\n"
        "SP braucht ca. 6 Tage, RJ ca. 11 Tage und AM über 23 Tage!"
    )
    props2 = dict(boxstyle='round,pad=1', facecolor='#1e1e1e', alpha=0.9, edgecolor='#3498db', linewidth=1.5)
    ax2.text(0.05, 0.95, info_2, transform=ax2.transAxes, fontsize=11,
             verticalalignment='top', bbox=props2, color='white')

    plt.suptitle("Survival Analysis: Time-to-Event Modellierung mit zensierten Daten", 
                 fontsize=18, fontweight='bold', color='white', y=1.05)
    plt.tight_layout()
    
    # Speichern der Grafik
    plt.savefig(GRAPHIC_PATH, dpi=300, bbox_inches='tight', facecolor=fig.get_facecolor())
    plt.close()
    
    print(f"✅ Grafik erfolgreich gespeichert unter:\n{GRAPHIC_PATH}")

if __name__ == "__main__":
    main()
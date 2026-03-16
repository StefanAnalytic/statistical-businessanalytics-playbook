"""
Griff 23: Müsst ihr die Test-Größe berechnen? (Power Analysis & Sample Size)

Was ist das und was macht das Skript?
Bevor man einen A/B-Test startet (z. B. ein neues Checkout-Design testet), MUSS man wissen, 
wie viele Datenpunkte (Kunden/Bestellungen) man braucht. Wenn man den Test zu früh stoppt, 
ist das Ergebnis statistischer Müll (Underpowered). Wenn man ihn zu lange laufen lässt, 
verschwendet man Zeit und Geld.

Die "Power Analysis" (Trennschärfe-Analyse) balanciert 4 magische Metriken:
1. Alpha (Signifikanzniveau): Meist 5% (0.05). Die Chance, einen Effekt zu sehen, der gar nicht da ist (False Positive).
2. Power (Trennschärfe): Meist 80% (0.80). Die Wahrscheinlichkeit, einen ECHTEN Effekt auch wirklich zu finden.
3. Effect Size (Effektstärke): Wie stark wird sich die Metrik ändern? (z.B. +5 BRL mehr Umsatz).
4. Sample Size (Stichprobengröße): Wie viele Kunden brauche ich pro Variante?

Business Case in diesem Skript:
Wir wollen eine neue Marketing-Kampagne testen, von der wir hoffen, dass sie den durchschnittlichen 
Bestellwert (Average Order Value - AOV) um 5 BRL erhöht. Wir nutzen die historischen Olist-Zahlungsdaten, 
um die natürliche Streuung (Standardabweichung) des AOVs zu berechnen. 
Danach berechnen wir exakt, wie viele Kunden wir für Variante A (Kontrolle) und B (Test) brauchen!
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Statsmodels hat ein eigenes Modul für Power Analysis
from statsmodels.stats.power import TTestIndPower

# -------------------------------------------------------------------
# 1. Setup & Pfade
# -------------------------------------------------------------------
# Wir wechseln in den Ordner "03_Hypothesentests_und_Inferenz" (Da A/B Tests dorthin gehören)
PAYMENTS_PATH = "/Users/stefanbechthold/statistik_business_analytics/data/olist_order_payments_dataset.csv"

GRAPHIC_DIR = "/Users/stefanbechthold/statistik_business_analytics/Grafiken/03_Hypothesentests_und_Inferenz"
GRAPHIC_PATH = os.path.join(GRAPHIC_DIR, "23_power_analysis_sample_size.png")

os.makedirs(GRAPHIC_DIR, exist_ok=True)

def main():
    print("🚀 Starte Griff 23: Power Analysis (Sample Size Berechnung)...")
    
    # -------------------------------------------------------------------
    # 2. Historische Daten laden (Baseline bestimmen)
    # -------------------------------------------------------------------
    print(f"Lade historische Zahlungsdaten von: {PAYMENTS_PATH}")
    df_payments = pd.read_csv(PAYMENTS_PATH)
    
    # Wir aggregieren den Zahlungswert pro Bestellung
    order_values = df_payments.groupby('order_id')['payment_value'].sum()
    
    # Ausreißer entfernen (> 1000 BRL), da sie die Standardabweichung künstlich aufblähen 
    # und A/B-Tests für den "normalen" Kunden unmöglich zu berechnen machen.
    baseline_data = order_values[order_values < 1000]
    
    baseline_mean = baseline_data.mean()
    baseline_std = baseline_data.std()
    
    print(f"\n--- Baseline Metriken (Historisch) ---")
    print(f"Aktueller Durchschnitt (AOV): {baseline_mean:.2f} BRL")
    print(f"Standardabweichung (Streuung): {baseline_std:.2f} BRL")

    # -------------------------------------------------------------------
    # 3. Effektstärke (Cohen's d) definieren
    # -------------------------------------------------------------------
    # Wie viel BRL Erhöhung wollen wir mit dem A/B Test mindestens messen können?
    # MDE = Minimum Detectable Effect
    mde_brl = 5.0  
    
    # Cohen's d ist die standardisierte Effektstärke: (Erwartete Differenz) / (Standardabweichung)
    # Es macht den absoluten BRL-Wert unabhängig von der Währung vergleichbar.
    effect_size_cohen = mde_brl / baseline_std
    
    print(f"\n--- Test-Design Parameter ---")
    print(f"Zieldifferenz (MDE): +{mde_brl:.2f} BRL")
    print(f"Standardisierte Effektstärke (Cohen's d): {effect_size_cohen:.4f}")

    # -------------------------------------------------------------------
    # 4. Power Analysis durchführen
    # -------------------------------------------------------------------
    print("\nBerechne benötigte Stichprobengröße (Sample Size)...")
    
    # Wir initialisieren das Power-Analyse Objekt für unabhängige t-Tests (Variante A vs B)
    power_analysis = TTestIndPower()
    
    # Parameter für den Test:
    alpha = 0.05  # 5% False Positive Rate (Standard)
    power = 0.80  # 80% True Positive Rate (Standard)
    
    # Wir übergeben 3 der 4 Parameter. Der Wert, der auf 'None' steht, wird berechnet!
    required_n = power_analysis.solve_power(effect_size=effect_size_cohen, 
                                            alpha=alpha, 
                                            power=power, 
                                            ratio=1.0, # 1.0 bedeutet Variante A und B sind gleich groß (50/50 Split)
                                            nobs1=None) 
    
    required_n = int(np.ceil(required_n)) # Aufrunden auf ganze Kunden
    
    print(f"Signifikanzniveau (Alpha): {alpha*100}%")
    print(f"Trennschärfe (Power): {power*100}%")
    print(f"-> BENÖTIGTE KUNDEN PRO VARIANTE: {required_n:,}")
    print(f"-> GESAMTE TEST-GRÖSSE (A + B): {(required_n * 2):,}")

    # -------------------------------------------------------------------
    # 5. Wunderschöne Visualisierung (Power Curves)
    # -------------------------------------------------------------------
    print("\nGeneriere Visualisierungs-Grid (Power Curves)...")
    sns.set_theme(style="whitegrid")
    
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))

    # --- Plot 1: Standard Power Curve (Sample Size vs. Power) ---
    # Wir plotten, wie die Power steigt, wenn wir mehr Kunden in den Test aufnehmen.
    # Wir berechnen dies für 3 verschiedene Szenarien (MDEs).
    
    sample_sizes = np.arange(1000, 30000, 500)
    
    mde_scenarios = [3.0, 5.0, 7.0] # In BRL
    colors = ['#e74c3c', '#2980b9', '#2ecc71']
    
    for mde, color in zip(mde_scenarios, colors):
        d_val = mde / baseline_std
        # Berechne die Power für jede Sample Size bei gegebenem Effekt
        powers = [power_analysis.solve_power(effect_size=d_val, nobs1=n, alpha=alpha, ratio=1.0) for n in sample_sizes]
        axes[0].plot(sample_sizes, powers, color=color, linewidth=3, label=f'+{mde} BRL (Cohen\'s d: {d_val:.3f})')
        
    axes[0].axhline(0.80, color='black', linestyle='--', linewidth=2, label='Ziel-Power (80%)')
    
    # Vertikale Linie für unser berechnetes N (beim 5 BRL Szenario) einzeichnen
    axes[0].axvline(required_n, color='gray', linestyle=':', linewidth=2)
    axes[0].scatter(required_n, 0.80, color='#2c3e50', s=100, zorder=5) # Markierungspunkt
    
    axes[0].set_title("1. Power Curve: Wie viele Kunden brauche ich?", fontsize=14, fontweight="bold")
    axes[0].set_xlabel("Sample Size (Anzahl Kunden pro Variante)")
    axes[0].set_ylabel("Statistical Power (Wahrscheinlichkeit, den Effekt zu finden)")
    axes[0].legend(title="Minimum Detectable Effect (MDE)")
    
    # Y-Achse als Prozent formatieren
    axes[0].set_ylim(0.0, 1.05)
    vals = axes[0].get_yticks()
    axes[0].set_yticklabels(['{:,.0%}'.format(x) for x in vals])

    # --- Plot 2: Effektstärke vs. Sample Size (bei konstanter 80% Power) ---
    # Zeigt den Trade-off: Wenn der Effekt klein ist, explodiert die benötigte Sample Size exponentiell!
    
    effect_sizes_to_plot = np.linspace(2.0, 15.0, 50) # BRL Effekte
    cohens_ds = effect_sizes_to_plot / baseline_std
    
    # Berechne die benötigte N für jede Effektstärke
    ns_required = [power_analysis.solve_power(effect_size=d, alpha=alpha, power=0.80, ratio=1.0, nobs1=None) for d in cohens_ds]
    
    axes[1].plot(effect_sizes_to_plot, ns_required, color='#8e44ad', linewidth=4)
    axes[1].fill_between(effect_sizes_to_plot, ns_required, color='#8e44ad', alpha=0.1)
    
    # Unseren spezifischen Fall einzeichnen
    axes[1].scatter(mde_brl, required_n, color='#e74c3c', s=150, zorder=5, marker='*')
    axes[1].plot([mde_brl, mde_brl], [0, required_n], color='#e74c3c', linestyle='--')
    axes[1].plot([0, mde_brl], [required_n, required_n], color='#e74c3c', linestyle='--')
    
    axes[1].set_title("2. Trade-Off: Kleinerer Effekt = Exponentiell mehr Daten", fontsize=14, fontweight="bold")
    axes[1].set_xlabel("Sichtbarer Effekt in BRL (Minimum Detectable Effect)")
    axes[1].set_ylabel("Benötigte Kunden (Pro Variante)")
    
    axes[1].set_xlim(2, 15)
    axes[1].set_ylim(0, max(ns_required) * 1.1)

    # Info-Box in Plot 2
    info_text = (
        f"A/B-Test Planung:\n"
        f"------------------\n"
        f"Baseline AOV: {baseline_mean:.1f} BRL\n"
        f"Streuung (StdDev): {baseline_std:.1f} BRL\n\n"
        f"Um eine Erhöhung von +{mde_brl:.1f} BRL mit\n"
        f"80% Sicherheit (Power) zu beweisen,\n"
        f"müssen exakt {required_n:,} Kunden pro\n"
        f"Variante durch den Test laufen."
    )
    props = dict(boxstyle='round,pad=0.8', facecolor='white', alpha=0.9, edgecolor='#8e44ad', linewidth=2)
    axes[1].text(0.5, 0.85, info_text, transform=axes[1].transAxes, fontsize=11,
                 verticalalignment='top', horizontalalignment='left', bbox=props, color='#2c3e50')

    # Layout optimieren und speichern
    plt.suptitle("Experiment Design: Power Analysis & Sample Size Berechnung", 
                 fontsize=18, fontweight="bold", y=1.05)
    plt.tight_layout()
    sns.despine()
    
    plt.savefig(GRAPHIC_PATH, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✅ Grafik erfolgreich gespeichert unter:\n{GRAPHIC_PATH}")

if __name__ == "__main__":
    main()
"""
Griff 6: Welche Verteilung passt zu meinen Daten? (Distributions-Fitting & QQ-Plots)

Was macht dieses Skript?
Es berechnet die reale Lieferzeit (in Tagen) aus dem Olist-Dataset. Danach testet das Skript, 
welche mathematische Wahrscheinlichkeitsverteilung (Normal, Log-Normal oder Gamma) am besten 
zu diesen echten Business-Daten passt. 
Es erstellt ein Histogramm mit den überlagerten Idealkurven und sogenannte QQ-Plots 
(Quantile-Quantile Plots), um die Passgenauigkeit visuell zu prüfen.

Wofür ist das im Business gut?
Wenn wir wissen, dass unsere Lieferzeiten einer Gamma-Verteilung folgen, können wir exakte 
Wahrscheinlichkeiten berechnen (z.B. "Mit 95%iger Wahrscheinlichkeit ist das Paket in unter 
12 Tagen beim Kunden"). Viele statistische Tests (wie der t-Test in Griff 9) setzen eine 
Normalverteilung voraus. Hier prüfen wir, ob diese Annahme überhaupt gerechtfertigt ist.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats

# -------------------------------------------------------------------
# 1. Setup & Pfade
# -------------------------------------------------------------------
DATA_PATH = "/Users/stefanbechthold/statistik_business_analytics/data/olist_orders_dataset.csv"
GRAPHIC_DIR = "/Users/stefanbechthold/statistik_business_analytics/Grafiken/02_Verteilungen_und_Sampling"
GRAPHIC_PATH = os.path.join(GRAPHIC_DIR, "06_distributions_fitting.png")

os.makedirs(GRAPHIC_DIR, exist_ok=True)

def main():
    print("🚀 Starte Griff 6: Distributions-Fitting & QQ-Plots...")
    
    # -------------------------------------------------------------------
    # 2. Daten laden & Feature Engineering (Lieferzeit berechnen)
    # -------------------------------------------------------------------
    print(f"Lade Daten von: {DATA_PATH}")
    df = pd.read_csv(DATA_PATH)
    
    # Konvertiere die Zeitstempel-Strings in echte datetime-Objekte
    df['purchase_time'] = pd.to_datetime(df['order_purchase_timestamp'])
    df['delivery_time'] = pd.to_datetime(df['order_delivered_customer_date'])
    
    # Berechne die Lieferdauer in Tagen
    df['delivery_days'] = (df['delivery_time'] - df['purchase_time']).dt.total_seconds() / (24 * 3600)
    
    # Bereinigung: Wir droppen fehlende Werte und ignorieren fehlerhafte negative/Null-Zeiten
    # sowie extreme Ausreißer (> 60 Tage), um das Fitting auf das Kerngeschäft zu fokussieren.
    delivery_data = df.loc[(df['delivery_days'] > 0) & (df['delivery_days'] <= 60), 'delivery_days'].dropna().values
    
    print(f"Analysiere {len(delivery_data)} gültige Lieferzeiten (0 bis 60 Tage).")

    # -------------------------------------------------------------------
    # 3. Fitting der Verteilungen (Mathematische Parameter schätzen)
    # -------------------------------------------------------------------
    print("Fitte Normal-, Log-Normal- und Gamma-Verteilungen an die Daten...")
    
    # 1. Normalverteilung (Glockenkurve)
    # loc = Mittelwert (Mean), scale = Standardabweichung (Std)
    norm_params = stats.norm.fit(delivery_data)
    
    # 2. Log-Normalverteilung (Perfekt für Werte, die bei 0 anfangen und einen langen rechten Schwanz haben)
    # s = Shape-Parameter
    lognorm_params = stats.lognorm.fit(delivery_data, floc=0)
    
    # 3. Gamma-Verteilung (Sehr flexibel für Wartezeiten/Lieferzeiten)
    # a = Shape, loc = Shift, scale = Skalierung
    gamma_params = stats.gamma.fit(delivery_data, floc=0)

    # -------------------------------------------------------------------
    # 4. Professionelle Visualisierung (PDFs & QQ-Plots)
    # -------------------------------------------------------------------
    print("Generiere Visualisierungs-Grid (1 Histogramm + 3 QQ-Plots)...")
    
    sns.set_theme(style="whitegrid")
    # Ein 2x2 Grid. Oben links und rechts verschmelzen wir für das Histogramm.
    fig = plt.figure(figsize=(16, 12))
    ax_hist = plt.subplot2grid((2, 3), (0, 0), colspan=3)
    ax_qq_norm = plt.subplot2grid((2, 3), (1, 0))
    ax_qq_lognorm = plt.subplot2grid((2, 3), (1, 1))
    ax_qq_gamma = plt.subplot2grid((2, 3), (1, 2))

    # --- Plot A: Histogramm mit gefitteten Dichtefunktionen (PDFs) ---
    # Zeichne das Histogramm der echten Daten als Dichte (stat='density')
    sns.histplot(delivery_data, bins=60, stat='density', color='lightgray', 
                 edgecolor='black', alpha=0.6, ax=ax_hist, label="Echte Daten (Lieferzeit)")
    
    # Erstelle eine x-Achse für die perfekten Kurven
    x_vals = np.linspace(0, 60, 500)
    
    # Berechne die Wahrscheinlichkeitsdichte (PDF) für jede gefittete Verteilung
    pdf_norm = stats.norm.pdf(x_vals, *norm_params)
    pdf_lognorm = stats.lognorm.pdf(x_vals, *lognorm_params)
    pdf_gamma = stats.gamma.pdf(x_vals, *gamma_params)
    
    # Kurven einzeichnen
    ax_hist.plot(x_vals, pdf_norm, color='#e74c3c', lw=3, label="Normalverteilung")
    ax_hist.plot(x_vals, pdf_lognorm, color='#2980b9', lw=3, label="Log-Normalverteilung")
    ax_hist.plot(x_vals, pdf_gamma, color='#27ae60', lw=3, label="Gamma-Verteilung")
    
    ax_hist.set_title("Lieferzeiten (Olist) vs. Theoretische Verteilungen", fontsize=16, fontweight="bold")
    ax_hist.set_xlabel("Lieferzeit in Tagen")
    ax_hist.set_ylabel("Dichte (Wahrscheinlichkeit)")
    ax_hist.legend(fontsize=12)

    # --- Plot B, C, D: QQ-Plots ---
    # Ein QQ-Plot vergleicht die Quantile der echten Daten mit den Quantilen der theoretischen Verteilung.
    # Liegen die roten Punkte exakt auf der diagonalen Linie, ist der Fit perfekt!
    
    # 1. QQ-Plot: Normalverteilung
    stats.probplot(delivery_data, dist="norm", sparams=(norm_params[0], norm_params[1]), plot=ax_qq_norm)
    ax_qq_norm.set_title("QQ-Plot: Normalverteilung", fontweight="bold")
    ax_qq_norm.get_lines()[0].set_markerfacecolor('#e74c3c')
    ax_qq_norm.get_lines()[0].set_markeredgecolor('white')
    
    # 2. QQ-Plot: Log-Normalverteilung
    stats.probplot(delivery_data, dist="lognorm", sparams=(lognorm_params[0], lognorm_params[1], lognorm_params[2]), plot=ax_qq_lognorm)
    ax_qq_lognorm.set_title("QQ-Plot: Log-Normalverteilung", fontweight="bold")
    ax_qq_lognorm.get_lines()[0].set_markerfacecolor('#2980b9')
    ax_qq_lognorm.get_lines()[0].set_markeredgecolor('white')

    # 3. QQ-Plot: Gamma-Verteilung
    stats.probplot(delivery_data, dist="gamma", sparams=(gamma_params[0], gamma_params[1], gamma_params[2]), plot=ax_qq_gamma)
    ax_qq_gamma.set_title("QQ-Plot: Gamma-Verteilung", fontweight="bold")
    ax_qq_gamma.get_lines()[0].set_markerfacecolor('#27ae60')
    ax_qq_gamma.get_lines()[0].set_markeredgecolor('white')

    # Layout optimieren und speichern
    plt.tight_layout()
    sns.despine()
    
    # --- Business Insights ---
    print("\n--- Business Insights (Basierend auf Log-Normal) ---")
    p50 = stats.lognorm.ppf(0.50, *lognorm_params)
    p90 = stats.lognorm.ppf(0.90, *lognorm_params)
    p95 = stats.lognorm.ppf(0.95, *lognorm_params)

    print(f"💡 Zu 50% (Median) ist das Paket in unter {p50:.1f} Tagen beim Kunden.")
    print(f"🚀 Zu 90% (Sicherheits-Level) ist das Paket in unter {p90:.1f} Tagen beim Kunden.")
    print(f"🛡️  Zu 95% (Garantie-Versprechen) ist das Paket in unter {p95:.1f} Tagen beim Kunden.")
    print("----------------------------------------------------\n")

    plt.savefig(GRAPHIC_PATH, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✅ Grafik erfolgreich gespeichert unter:\n{GRAPHIC_PATH}")

if __name__ == "__main__":
    main()
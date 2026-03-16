"""
Griff 10: Was tun, wenn die KPI nicht normalverteilt ist? (Nonparametrische Tests)

Was ist das und was macht das Skript?
Der t-Test (aus Griff 9) ist super, aber er hat eine Schwäche: Er vergleicht Mittelwerte.
Wenn deine Daten extrem schief sind (wie Umsätze, wo wenige "Whales" den Schnitt ruinieren), 
wird der Mittelwert verzerrt. Hier betritt der Mann-Whitney U Test (auch Wilcoxon-Rangsummentest) die Bühne!

Dieser Test ist "non-parametrisch". Er ignoriert die absolute Höhe des Umsatzes und schaut
sich stattdessen die *Ränge* an. (Einfach gesagt: Er sortiert alle Umsätze von Platz 1 bis 10.000 
und vergleicht, ob Gruppe A im Schnitt höhere Platzierungen hat als Gruppe B).

Business Case in diesem Skript:
Wir verknüpfen den Umsatz mit den Kundenbewertungen! Geben extrem unzufriedene Kunden (1 Stern) 
eigentlich mehr oder weniger Geld aus als extrem glückliche Kunden (5 Sterne)? 
Da Umsatz extrem schief verteilt ist, nutzen wir den robusten Mann-Whitney U Test.
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
PAYMENTS_PATH = "/Users/stefanbechthold/statistik_business_analytics/data/olist_order_payments_dataset.csv"
REVIEWS_PATH = "/Users/stefanbechthold/statistik_business_analytics/data/olist_order_reviews_dataset.csv"
GRAPHIC_DIR = "/Users/stefanbechthold/statistik_business_analytics/Grafiken/03_Hypothesentests_und_Inferenz"
GRAPHIC_PATH = os.path.join(GRAPHIC_DIR, "10_mann_whitney_u_test.png")

os.makedirs(GRAPHIC_DIR, exist_ok=True)

def main():
    print("🚀 Starte Griff 10: Non-Parametrische Tests (Mann-Whitney U)...")
    
    # -------------------------------------------------------------------
    # 2. Daten laden, aggregieren & verknüpfen (Mergen)
    # -------------------------------------------------------------------
    print("Lade Zahlungs- und Bewertungsdaten...")
    df_payments = pd.read_csv(PAYMENTS_PATH)
    df_reviews = pd.read_csv(REVIEWS_PATH)
    
    # 1. Zahlungen pro Bestellung aufsummieren (falls jemand mit 2 Karten zahlt)
    order_spend = df_payments.groupby('order_id')['payment_value'].sum().reset_index()
    
    # 2. Nur die erste Bewertung pro Bestellung nehmen (Duplikate entfernen)
    order_reviews = df_reviews.drop_duplicates(subset=['order_id'])[['order_id', 'review_score']]
    
    # 3. Mergen: Umsatz und Bewertung zusammenbringen
    df_merged = pd.merge(order_spend, order_reviews, on='order_id', how='inner')
    
    # -------------------------------------------------------------------
    # 3. Gruppen definieren (1 Stern vs. 5 Sterne)
    # -------------------------------------------------------------------
    group_1_star = df_merged[df_merged['review_score'] == 1]['payment_value'].values
    group_5_star = df_merged[df_merged['review_score'] == 5]['payment_value'].values
    
    median_1 = np.median(group_1_star)
    median_5 = np.median(group_5_star)
    
    print(f"Stichprobe 1 Stern:  {len(group_1_star)} Bestellungen (Median: {median_1:.2f} BRL)")
    print(f"Stichprobe 5 Sterne: {len(group_5_star)} Bestellungen (Median: {median_5:.2f} BRL)")

    # -------------------------------------------------------------------
    # 4. Mann-Whitney U Test durchführen
    # -------------------------------------------------------------------
    print("\nFühre Mann-Whitney U Test durch (Rangbasiert, robust gegen Ausreißer)...")
    
    # alternative='two-sided' prüft, ob es überhaupt einen Unterschied gibt (egal in welche Richtung)
    u_stat, p_value = stats.mannwhitneyu(group_1_star, group_5_star, alternative='two-sided')
    
    print(f"U-Statistik: {u_stat:.2f}")
    print(f"P-Wert:      {p_value:.4e}")
    
    alpha = 0.05
    if p_value < alpha:
        interpretation = "Signifikant!\n(Die Umsatzverteilungen unterscheiden sich echt)"
        color_sig = "#27ae60" # Grün
    else:
        interpretation = "Nicht signifikant.\n(Kein verlässlicher Unterschied)"
        color_sig = "#e74c3c" # Rot

    # -------------------------------------------------------------------
    # 5. Wunderschöne Visualisierung (Boxplot + ECDF)
    # -------------------------------------------------------------------
    print("Generiere Visualisierungs-Grid (Boxplot & ECDF)...")
    sns.set_theme(style="whitegrid")
    
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    
    # DataFrame für Seaborn vorbereiten
    plot_df = pd.DataFrame({
        'Bewertung': ['1 Stern (Wütend)'] * len(group_1_star) + ['5 Sterne (Glücklich)'] * len(group_5_star),
        'Umsatz': np.concatenate([group_1_star, group_5_star])
    })

    # --- Plot 1: Boxplot (Zoom auf die Masse der Daten) ---
    # Boxplots basieren auf dem Median und Quartilen -> Perfekt für non-parametrische Daten!
    sns.boxplot(data=plot_df, x='Bewertung', y='Umsatz', 
                palette=['#e74c3c', '#2ecc71'], ax=axes[0], showfliers=False) # Ausreißer ausblenden für Zoom
    
    axes[0].set_title("Vergleich der Mediane (Ausreißer ausgeblendet)", fontsize=14, fontweight="bold")
    axes[0].set_ylabel("Bestellwert (BRL)")
    axes[0].set_xlabel("")
    
    # Mediane als Text in den Plot schreiben
    axes[0].text(0, median_1 + 10, f"Median: {median_1:.2f}", ha='center', color='white', weight='bold', 
                 bbox=dict(facecolor='black', alpha=0.6, edgecolor='none', boxstyle='round,pad=0.3'))
    axes[0].text(1, median_5 + 10, f"Median: {median_5:.2f}", ha='center', color='white', weight='bold', 
                 bbox=dict(facecolor='black', alpha=0.6, edgecolor='none', boxstyle='round,pad=0.3'))

    # --- Plot 2: ECDF (Empirical Cumulative Distribution Function) ---
    # Die ECDF zeigt uns die komplette Verteilung ohne Informationsverlust durch Bins (wie beim Histogramm).
    # Lesebeispiel: Bei y=0.8 (80%) sehen wir, unter welchem Betrag 80% der Bestellungen liegen.
    sns.ecdfplot(data=plot_df, x='Umsatz', hue='Bewertung', 
                 palette=['#e74c3c', '#2ecc71'], ax=axes[1], linewidth=3)
    
    axes[1].set_xlim(0, 500) # Fokus auf die relevantesten 90% der Daten
    axes[1].set_title("Kumulierte Verteilung (ECDF) bis 500 BRL", fontsize=14, fontweight="bold")
    axes[1].set_ylabel("Anteil der Bestellungen (Kumuliert)")
    axes[1].set_xlabel("Bestellwert (BRL)")
    
    # Textbox mit den statistischen Ergebnissen
    stats_text = (
        f"Mann-Whitney U Test:\n"
        f"--------------------\n"
        f"U-Stat: {u_stat:,.0f}\n"
        f"p-Wert: {p_value:.5f}\n"
        f"Fazit: {interpretation}"
    )
    
    props = dict(boxstyle='round,pad=0.8', facecolor='white', alpha=0.9, edgecolor=color_sig, linewidth=2)
    axes[1].text(0.55, 0.25, stats_text, transform=axes[1].transAxes, fontsize=12,
                 verticalalignment='top', bbox=props, color='#2c3e50')

    # Layout optimieren und speichern
    plt.suptitle("Umsatz-Analyse nach Kundenzufriedenheit (1 Stern vs. 5 Sterne)", 
                 fontsize=18, fontweight="bold", y=1.05)
    plt.tight_layout()
    sns.despine()
    
    plt.savefig(GRAPHIC_PATH, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✅ Grafik erfolgreich gespeichert unter:\n{GRAPHIC_PATH}")

if __name__ == "__main__":
    main()
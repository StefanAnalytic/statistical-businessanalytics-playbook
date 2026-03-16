"""
Griff 37: Viele Hypothesen testen? (Multiple Testing: Bonferroni & Benjamini-Hochberg)

Was ist das und was macht das Skript?
Wenn wir einen einzelnen statistischen Test mit einem Signifikanzniveau (Alpha) von 5% durchführen, 
akzeptieren wir eine 5%ige Chance auf einen False Positive (wir finden einen Effekt, der nur Zufall ist). 
Wenn wir jedoch 50 Tests gleichzeitig durchführen (z.B. A/B-Tests für 50 verschiedene Produktkategorien), 
steigt die Wahrscheinlichkeit, MINDESTENS einen False Positive zu finden, dramatisch an (Alpha-Inflation).

Um diese künstlichen "Schein-Erfolge" zu verhindern, müssen wir unsere p-Werte korrigieren:
1. Bonferroni-Korrektur: Sehr konservativ. Teilt Alpha durch die Anzahl der Tests (FWER - Family-Wise Error Rate).
   Nachteil: Sie ist oft zu streng und "tötet" echte Effekte (reduziert die statistische Power).
2. Benjamini-Hochberg (FDR - False Discovery Rate): Der moderne Standard. Kontrolliert nicht die 
   Wahrscheinlichkeit EINES False Positives, sondern den Anteil der False Positives unter allen 
   signifikanten Ergebnissen. Viel praxisnäher für Business-Anwendungen.

Business Case in diesem Skript:
Wir prüfen für die Top 50 Produktkategorien im Olist-Datensatz, ob ihre durchschnittliche 
Kundenbewertung signifikant vom globalen Durchschnitt abweicht. Wir vergleichen die naiven 
p-Werte mit den korrigierten Werten nach Bonferroni und FDR, um zu sehen, wie viele 
"Erfolge" einer strengen Prüfung standhalten.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import ttest_1samp
from statsmodels.stats.multitest import multipletests

# -------------------------------------------------------------------
# 1. Setup & Pfade
# -------------------------------------------------------------------
REVIEWS_PATH = "/Users/stefanbechthold/statistik_business_analytics/data/olist_order_reviews_dataset.csv"
ITEMS_PATH = "/Users/stefanbechthold/statistik_business_analytics/data/olist_order_items_dataset.csv"
PRODUCTS_PATH = "/Users/stefanbechthold/statistik_business_analytics/data/olist_products_dataset.csv"
TRANSLATION_PATH = "/Users/stefanbechthold/statistik_business_analytics/data/product_category_name_translation.csv"

GRAPHIC_DIR = "/Users/stefanbechthold/statistik_business_analytics/Grafiken/03_Hypothesentests_und_Inferenz"
GRAPHIC_PATH = os.path.join(GRAPHIC_DIR, "37_multiple_testing_correction.png")

os.makedirs(GRAPHIC_DIR, exist_ok=True)

def main():
    # -------------------------------------------------------------------
    # 2. Daten laden & Review-Scores pro Kategorie aggregieren
    # -------------------------------------------------------------------
    df_reviews = pd.read_csv(REVIEWS_PATH)
    df_items = pd.read_csv(ITEMS_PATH)
    df_products = pd.read_csv(PRODUCTS_PATH)
    df_translation = pd.read_csv(TRANSLATION_PATH)
    
    # Eindeutige Reviews pro Bestellung
    reviews_unique = df_reviews.drop_duplicates(subset=['order_id'])[['order_id', 'review_score']]
    items_unique = df_items.drop_duplicates(subset=['order_id'])[['order_id', 'product_id']]
    
    # Kategorien übersetzen
    df_products = pd.merge(df_products, df_translation, on='product_category_name', how='left')
    
    # Master-Tabelle
    df_merged = pd.merge(reviews_unique, items_unique, on='order_id', how='inner')
    df_merged = pd.merge(df_merged, df_products[['product_id', 'product_category_name_english']], on='product_id', how='inner')
    df_clean = df_merged.dropna(subset=['review_score', 'product_category_name_english']).copy()
    
    # Globaler Durchschnitt aller Bewertungen
    global_mean = df_clean['review_score'].mean()
    
    # Wir nehmen die Top 50 Kategorien (mit den meisten Bestellungen), um ausreichend Daten pro Test zu haben
    top_50_categories = df_clean['product_category_name_english'].value_counts().head(50).index

    # -------------------------------------------------------------------
    # 3. Multiple Hypothesentests durchführen
    # -------------------------------------------------------------------
    categories = []
    p_values = []
    means = []
    
    # Für jede Kategorie: 1-Sample t-Test gegen den globalen Durchschnitt
    for cat in top_50_categories:
        cat_data = df_clean[df_clean['product_category_name_english'] == cat]['review_score']
        
        # Test: Ist der Mittelwert dieser Kategorie signifikant anders als der globale Mittelwert?
        stat, pval = ttest_1samp(cat_data, popmean=global_mean)
        
        categories.append(cat)
        p_values.append(pval)
        means.append(cat_data.mean())

    df_tests = pd.DataFrame({
        'Category': categories,
        'Mean': means,
        'p_value_naive': p_values
    })

    # -------------------------------------------------------------------
    # 4. p-Werte korrigieren (Bonferroni & Benjamini-Hochberg FDR)
    # -------------------------------------------------------------------
    alpha = 0.05
    
    # 1. Naive Signifikanz
    df_tests['sig_naive'] = df_tests['p_value_naive'] < alpha
    
    # 2. Bonferroni Korrektur
    reject_bonf, pvals_bonf, _, _ = multipletests(df_tests['p_value_naive'], alpha=alpha, method='bonferroni')
    df_tests['p_value_bonferroni'] = pvals_bonf
    df_tests['sig_bonferroni'] = reject_bonf
    
    # 3. Benjamini-Hochberg (FDR) Korrektur
    reject_fdr, pvals_fdr, _, _ = multipletests(df_tests['p_value_naive'], alpha=alpha, method='fdr_bh')
    df_tests['p_value_fdr'] = pvals_fdr
    df_tests['sig_fdr'] = reject_fdr

    # Zusammenfassung der "Entdeckungen"
    naive_count = df_tests['sig_naive'].sum()
    fdr_count = df_tests['sig_fdr'].sum()
    bonf_count = df_tests['sig_bonferroni'].sum()

    # Wir sortieren den DataFrame nach naiven p-Werten für die Visualisierung
    df_tests = df_tests.sort_values('p_value_naive').reset_index(drop=True)

    # -------------------------------------------------------------------
    # 5. Visualisierung im Dark Mode Design
    # -------------------------------------------------------------------
    plt.style.use('dark_background')
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7), facecolor='#121212')
    ax1.set_facecolor('#121212')
    ax2.set_facecolor('#121212')

    # --- Plot 1: p-Wert Verteilung & Schwellenwerte ---
    # Wir plotten die naiven p-Werte log-skaliert, da kleine p-Werte entscheidend sind
    ax1.plot(range(len(df_tests)), df_tests['p_value_naive'], marker='o', color='#3498db', linewidth=2, label='Beobachtete p-Werte')
    
    # Naives Alpha (0.05)
    ax1.axhline(alpha, color='#e74c3c', linestyle='--', linewidth=2, label=f'Naives Alpha ({alpha})')
    
    # Bonferroni Schwelle (0.05 / 50)
    bonf_threshold = alpha / len(top_50_categories)
    ax1.axhline(bonf_threshold, color='#f1c40f', linestyle=':', linewidth=2, label=f'Bonferroni Schwelle ({bonf_threshold:.4f})')
    
    ax1.set_yscale('log')
    ax1.set_title("1. p-Werte der 50 Tests (Log-Skala)", fontsize=14, fontweight='bold', color='white')
    ax1.set_xlabel("Rang des Tests (sortiert nach Signifikanz)")
    ax1.set_ylabel("Naiv berechneter p-Wert (log)")
    
    legend1 = ax1.legend(loc='lower right', facecolor='#1e1e1e', edgecolor='#333333')
    plt.setp(legend1.get_texts(), color='white')

    # --- Plot 2: Bar Chart der signifikanten Ergebnisse ---
    labels = ['Naiv\n(Oft False Positives)', 'FDR / Benjamini-Hochberg\n(Guter Kompromiss)', 'Bonferroni\n(Zu streng)']
    counts = [naive_count, fdr_count, bonf_count]
    colors = ['#e74c3c', '#2ecc71', '#f1c40f']
    
    bars = ax2.bar(labels, counts, color=colors, edgecolor='none')
    
    ax2.set_title("2. Anzahl der als 'signifikant' eingestuften Kategorien", fontsize=14, fontweight='bold', color='white')
    ax2.set_ylabel("Anzahl signifikanter Abweichungen")
    ax2.set_ylim(0, max(counts) * 1.2)
    
    for bar in bars:
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height + 0.5, 
                 f"{int(height)}", ha='center', va='bottom', color='white', fontweight='bold', fontsize=14)

    # Info-Box in Plot 2
    info_text = (
        "Das Multiple Testing Problem:\n"
        "------------------------------\n"
        f"Testen wir 50 Kategorien naiv mit Alpha=0.05,\n"
        f"finden wir {naive_count} Abweichungen. Darunter sind fast\n"
        f"garantiert False Positives (reiner Zufall).\n\n"
        f"Bonferroni ist extrem streng und lässt nur noch {bonf_count}\n"
        f"Kategorien zu (verliert statistische Power).\n"
        f"FDR (Benjamini-Hochberg) filtert intelligent und\n"
        f"liefert {fdr_count} verlässliche Business-Erkenntnisse."
    )
    props = dict(boxstyle='round,pad=1', facecolor='#1e1e1e', alpha=0.9, edgecolor='#2ecc71', linewidth=1.5)
    ax2.text(0.05, 0.95, info_text, transform=ax2.transAxes, fontsize=11,
             verticalalignment='top', bbox=props, color='#e0e0e0')

    plt.suptitle("Statistical Inference: Kontrolle der False Discovery Rate bei Multiplen Tests", fontsize=18, fontweight='bold', color='white', y=1.05)
    plt.tight_layout()
    
    # Speichern der Grafik
    plt.savefig(GRAPHIC_PATH, dpi=300, bbox_inches='tight', facecolor=fig.get_facecolor())
    plt.close()

if __name__ == "__main__":
    main()
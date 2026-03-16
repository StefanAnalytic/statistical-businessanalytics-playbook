"""
Griff 35: Daten mit Gruppen/Hierarchien? (Mixed Effects / Hierarchical Models)

Was ist das und was macht das Skript?
Klassische Regressionen (OLS) gehen davon aus, dass alle Datenpunkte unabhängig voneinander sind.
Im E-Commerce ist das fast nie der Fall. Daten sind oft in "Gruppen" verschachtelt (Nesting).

Ein perfektes Beispiel: Produktbewertungen (Review Scores) und Produktkategorien.
Wir wollen wissen, ob eine längere Produktbeschreibung zu besseren Bewertungen führt (Fixed Effect).
Aber: Eine Kategorie wie "Möbel" hat vielleicht von Haus aus strengere Bewertungen (schwieriger 
Versand, Aufbau) als die Kategorie "Bücher". 

Wenn wir alle Daten in einen Topf werfen (OLS), ignorieren wir diese Kategorie-spezifischen 
Unterschiede. Ein Mixed Effects Model löst das:
1. Fixed Effects: Der globale, allgemeine Effekt der Beschreibungslänge auf die Bewertung.
2. Random Effects: Erlaubt jeder Produktkategorie einen eigenen "Startwert" (Random Intercept).

Business Case in diesem Skript (Neuer Fokus!):
Wir analysieren den Einfluss der Produktbeschreibungs-Länge (Wörter/Zeichen) auf die 
Kundenbewertung (1-5 Sterne). Wir gruppieren die Daten hierarchisch nach Produktkategorie 
(z.B. Health/Beauty, Bed/Bath, Sports), um den echten Effekt der Textlänge von der 
grundsätzlichen Zufriedenheit in der jeweiligen Kategorie zu trennen.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.formula.api as smf

# -------------------------------------------------------------------
# 1. Setup & Pfade
# -------------------------------------------------------------------
REVIEWS_PATH = "/Users/stefanbechthold/statistik_business_analytics/data/olist_order_reviews_dataset.csv"
ITEMS_PATH = "/Users/stefanbechthold/statistik_business_analytics/data/olist_order_items_dataset.csv"
PRODUCTS_PATH = "/Users/stefanbechthold/statistik_business_analytics/data/olist_products_dataset.csv"
TRANSLATION_PATH = "/Users/stefanbechthold/statistik_business_analytics/data/product_category_name_translation.csv"

GRAPHIC_DIR = "/Users/stefanbechthold/statistik_business_analytics/Grafiken/17_Mixed_Effects"
GRAPHIC_PATH = os.path.join(GRAPHIC_DIR, "35_mixed_effects_reviews_categories.png")

os.makedirs(GRAPHIC_DIR, exist_ok=True)

def main():
    # -------------------------------------------------------------------
    # 2. Daten laden & Hierarchischen Datensatz aufbauen
    # -------------------------------------------------------------------
    df_reviews = pd.read_csv(REVIEWS_PATH)
    df_items = pd.read_csv(ITEMS_PATH)
    df_products = pd.read_csv(PRODUCTS_PATH)
    df_translation = pd.read_csv(TRANSLATION_PATH)
    
    # 1. Wir brauchen nur die initiale Bewertung pro Bestellung
    reviews_unique = df_reviews.drop_duplicates(subset=['order_id'])[['order_id', 'review_score']]
    
    # 2. Produkt-ID zur Bestellung mappen (wir nehmen das erste Produkt der Bestellung)
    items_unique = df_items.drop_duplicates(subset=['order_id'])[['order_id', 'product_id']]
    
    # 3. Kategorienamen übersetzen
    df_products = pd.merge(df_products, df_translation, on='product_category_name', how='left')
    
    # 4. Master-Tabelle bauen
    df_merged = pd.merge(reviews_unique, items_unique, on='order_id', how='inner')
    df_merged = pd.merge(df_merged, df_products[['product_id', 'product_category_name_english', 'product_description_lenght']], on='product_id', how='inner')
    
    # Spaltennamen bereinigen (Tippfehler im Original-Datensatz korrigieren)
    df_merged.rename(columns={'product_description_lenght': 'description_length', 
                              'product_category_name_english': 'category'}, inplace=True)
    df_merged.dropna(subset=['review_score', 'description_length', 'category'], inplace=True)
    
    # Um stabile Gruppen zu haben, nehmen wir die Top 8 meistverkauften Kategorien
    top_categories = df_merged['category'].value_counts().nlargest(8).index
    df_clean = df_merged[df_merged['category'].isin(top_categories)].copy()
    
    # Skalierung der Beschreibungslänge (in 100er Zeichen), damit die Koeffizienten lesbar bleiben
    df_clean['desc_len_100'] = df_clean['description_length'] / 100.0
    
    # Für LML-Konvergenz (Mixed Models) ziehen wir ein repräsentatives Sample
    df_sample = df_clean.sample(n=4000, random_state=42)

    # -------------------------------------------------------------------
    # 3. Modelle trainieren (OLS vs. Mixed Effects)
    # -------------------------------------------------------------------
    formula = "review_score ~ desc_len_100"
    
    # 1. Naives OLS (Ignoriert die Kategorien-Hierarchie)
    ols_model = smf.ols(formula, data=df_sample).fit()
    
    # 2. Mixed Effects Model (Random Intercept für jede Produktkategorie)
    # groups=df_sample['category'] definiert die Hierarchie-Ebene (Nesting)
    mixed_model = smf.mixedlm(formula, data=df_sample, groups=df_sample['category']).fit()
    
    # Extrahieren der Kategorie-Abweichungen (Random Effects)
    random_effects = mixed_model.random_effects

    # -------------------------------------------------------------------
    # 4. Visualisierung im Dark Mode Design
    # -------------------------------------------------------------------
    plt.style.use('dark_background')
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 7), facecolor='#121212')
    ax1.set_facecolor('#121212')
    ax2.set_facecolor('#121212')

    # X-Werte für die Regressionsgeraden
    x_pred = np.linspace(df_sample['desc_len_100'].min(), df_sample['desc_len_100'].max(), 50)
    
    # --- Plot 1: Globale vs. Gruppenspezifische Trends ---
    # Streudiagramm (Transparenz hoch, da Review Scores diskret 1-5 sind + Jittering für Sichtbarkeit)
    jittered_y = df_sample['review_score'] + np.random.normal(0, 0.15, size=len(df_sample))
    
    sns.scatterplot(x=df_sample['desc_len_100'], y=jittered_y, 
                    hue=df_sample['category'], palette='tab10', 
                    alpha=0.15, edgecolor='none', s=20, ax=ax1, legend=False)
    
    # Globale OLS Linie (Fixed Effect im naiven Modell)
    y_ols = ols_model.params['Intercept'] + ols_model.params['desc_len_100'] * x_pred
    ax1.plot(x_pred, y_ols, color='#ffffff', linewidth=4, linestyle='--', label='Naives OLS (Ignoriert Kategorien)')
    
    # Gruppenspezifische Linien (Mixed Model)
    fixed_intercept = mixed_model.params['Intercept']
    fixed_slope = mixed_model.params['desc_len_100']
    
    colors = sns.color_palette('tab10', n_colors=8)
    for idx, (category, re) in enumerate(random_effects.items()):
        # Das Intercept verschiebt sich pro Kategorie (Random Intercept)
        cat_intercept = fixed_intercept + re['Group']
        y_mixed = cat_intercept + fixed_slope * x_pred
        ax1.plot(x_pred, y_mixed, color=colors[idx], linewidth=2.5, alpha=0.9, label=f'Kat: {category}')

    ax1.set_title("1. Random Intercepts: Basis-Zufriedenheit pro Kategorie", fontsize=14, fontweight='bold', color='white')
    ax1.set_xlabel("Länge der Produktbeschreibung (in 100 Zeichen)")
    ax1.set_ylabel("Review Score (1 bis 5 Sterne)")
    ax1.set_ylim(0.5, 5.5)
    
    legend = ax1.legend(loc='lower right', facecolor='#1e1e1e', edgecolor='#333333', fontsize=9)
    plt.setp(legend.get_texts(), color='white')

    # --- Plot 2: Caterpillar Plot (Abweichungen der Random Effects) ---
    # Zeigt, welche Kategorie über- oder unterdurchschnittliche Basis-Bewertungen hat
    categories = list(random_effects.keys())
    re_values = [re['Group'] for re in random_effects.values()]
    
    # Sortieren für visuelle Klarheit
    sorted_indices = np.argsort(re_values)
    cat_sorted = [categories[i] for i in sorted_indices]
    re_values_sorted = [re_values[i] for i in sorted_indices]
    
    ax2.axvline(0, color='white', linestyle='--', linewidth=2, alpha=0.5)
    
    for i, (cat, val) in enumerate(zip(cat_sorted, re_values_sorted)):
        color = '#2ecc71' if val > 0 else '#e74c3c'
        ax2.plot(val, i, marker='o', markersize=10, color=color)
        ax2.hlines(y=i, xmin=0, xmax=val, color=color, linewidth=3)
        
    ax2.set_yticks(range(len(cat_sorted)))
    ax2.set_yticklabels(cat_sorted, fontsize=11, fontweight='bold')
    
    ax2.set_title("2. Kategorie-Effekte (Abweichung vom globalen Sterne-Durchschnitt)", fontsize=14, fontweight='bold', color='white')
    ax2.set_xlabel("Abweichung in Sternen (Random Intercept)")

    # Info-Box
    info_text = (
        "Business Insight (Hierarchische Modelle):\n"
        "-----------------------------------------\n"
        "Wenn wir alle Daten mischen (OLS), übersehen wir Struktur.\n"
        "Das Mixed Model deckt auf:\n"
        "1. Längere Beschreibungen heben die Bewertung leicht an (Steigung).\n"
        "2. Aber der Startpunkt ist völlig anders! Kunden vergeben für\n"
        "   'Gesundheit/Beauty' viel leichter gute Bewertungen (Grün)\n"
        "   als für 'Bett/Bad/Tisch' (Rot), unabhängig vom Text."
    )
    props = dict(boxstyle='round,pad=1', facecolor='#1e1e1e', alpha=0.9, edgecolor='#9b59b6', linewidth=1.5)
    ax2.text(0.02, 0.95, info_text, transform=ax2.transAxes, fontsize=11,
             verticalalignment='top', bbox=props, color='#e0e0e0')

    plt.suptitle("Hierarchical Models: Genestete Datenstrukturen (Kategorien vs. Reviews)", fontsize=18, fontweight='bold', color='white', y=1.05)
    plt.tight_layout()
    
    # Speichern
    plt.savefig(GRAPHIC_PATH, dpi=300, bbox_inches='tight', facecolor=fig.get_facecolor())
    plt.close()

if __name__ == "__main__":
    main()
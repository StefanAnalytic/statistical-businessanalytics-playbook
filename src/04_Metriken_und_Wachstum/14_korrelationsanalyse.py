"""
Griff 14: Wie stark ist der Zusammenhang? (Korrelation: Pearson, Spearman, Kendall, Partielle Korr.)

Was ist das und was macht das Skript?
Korrelation misst, wie stark zwei numerische Variablen zusammenhängen. Aber nicht alle 
Zusammenhänge sind gleich! Dieses Skript berechnet vier verschiedene Arten der Korrelation 
anhand echter Produktdaten (Preis, Versandkosten, Gewicht, Anzahl der Fotos):

1. Pearson-Korrelation: Der Standard. Misst rein LINEARE Zusammenhänge. Anfällig für Ausreißer.
2. Spearman-Korrelation: Misst MONOTONE Zusammenhänge (Wenn X steigt, steigt Y, egal ob als Kurve 
   oder Linie). Arbeitet mit Rängen statt echten Werten. Extrem robust gegen Ausreißer!
3. Kendall-Tau: Ähnlich wie Spearman (rangbasiert), aber noch robuster bei kleinen Datenmengen 
   oder vielen exakt gleichen Werten (Ties).
4. Partielle Korrelation: Die "Detektiv"-Korrelation. Misst den wahren Zusammenhang zwischen 
   X und Y, während der verzerrende Einfluss einer dritten Variable (Z) herausgerechnet wird.

Business Case in diesem Skript:
Wir untersuchen den Zusammenhang zwischen 'price' (Produktpreis) und 'freight_value' (Versandkosten).
Scheinbar gibt es eine starke Korrelation! Aber Moment... schwere Produkte ('product_weight_g') 
sind oft teurer UND kosten mehr Versand. Ist der Zusammenhang zwischen Preis und Versand 
vielleicht nur eine Illusion, getrieben vom Gewicht? Die Partielle Korrelation deckt es auf!
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats
from sklearn.linear_model import LinearRegression

# -------------------------------------------------------------------
# 1. Setup & Pfade
# -------------------------------------------------------------------
# Wir wechseln in den Ordner "04_Metriken_und_Wachstum"
ITEMS_PATH = "/Users/stefanbechthold/statistik_business_analytics/data/olist_order_items_dataset.csv"
PRODUCTS_PATH = "/Users/stefanbechthold/statistik_business_analytics/data/olist_products_dataset.csv"

GRAPHIC_DIR = "/Users/stefanbechthold/statistik_business_analytics/Grafiken/04_Metriken_und_Wachstum"
GRAPHIC_PATH = os.path.join(GRAPHIC_DIR, "14_korrelationen_vergleich.png")

os.makedirs(GRAPHIC_DIR, exist_ok=True)

def main():
    print("🚀 Starte Griff 14: Korrelationen (Pearson, Spearman, Kendall & Partiell)...")
    
    # -------------------------------------------------------------------
    # 2. Daten laden, bereinigen & mergen
    # -------------------------------------------------------------------
    print("Lade Produkt- und Bestelldaten...")
    df_items = pd.read_csv(ITEMS_PATH)
    df_products = pd.read_csv(PRODUCTS_PATH)
    
    # Verbinde die Verkaufsdaten (Preis, Versand) mit den Produktmerkmalen (Gewicht, Fotos)
    df_merged = pd.merge(df_items, df_products, on='product_id', how='inner')
    
    # Wähle die relevanten numerischen Features
    cols = ['price', 'freight_value', 'product_weight_g', 'product_photos_qty']
    df_analysis = df_merged[cols].dropna().copy()
    
    # Um die Berechnungszeit im Rahmen zu halten und Overplotting im Scatterplot 
    # zu vermeiden, ziehen wir ein repräsentatives Sample von 5.000 Zeilen.
    df_sample = df_analysis.sample(n=5000, random_state=42)
    
    print(f"Analysiere {len(df_sample)} Produkte...")

    # -------------------------------------------------------------------
    # 3. Klassische Korrelationen berechnen (Pearson, Spearman, Kendall)
    # -------------------------------------------------------------------
    print("\nBerechne Korrelationsmatrizen...")
    
    # Pandas macht das extrem einfach über die .corr() Methode
    corr_pearson = df_sample.corr(method='pearson')
    corr_spearman = df_sample.corr(method='spearman')
    corr_kendall = df_sample.corr(method='kendall')
    
    print("\nPearson Korrelation (Preis vs. Versandkosten):  {:.3f}".format(corr_pearson.loc['price', 'freight_value']))
    print("Spearman Korrelation (Preis vs. Versandkosten): {:.3f}".format(corr_spearman.loc['price', 'freight_value']))

    # -------------------------------------------------------------------
    # 4. Partielle Korrelation berechnen (Der Einfluss-Entferner)
    # -------------------------------------------------------------------
    print("\nBerechne Partielle Korrelation (Preis vs. Versand, kontrolliert für Gewicht)...")
    
    # Wie rechnet man den Einfluss des Gewichts (Z) aus Preis (X) und Versand (Y) heraus?
    # Schritt 1: Wir sagen den Preis durch das Gewicht vorher (Lineare Regression) und behalten nur den Fehler (Residuum).
    # Das Residuum ist der "reine" Preis, der absolut nichts mehr mit dem Gewicht zu tun hat.
    # Schritt 2: Dasselbe machen wir für die Versandkosten.
    # Schritt 3: Wir berechnen die normale Pearson-Korrelation zwischen diesen beiden Residuen!
    
    X_weight = df_sample[['product_weight_g']]
    y_price = df_sample['price']
    y_freight = df_sample['freight_value']
    
    # Regression 1: Preis durch Gewicht erklären
    model_price = LinearRegression().fit(X_weight, y_price)
    residuals_price = y_price - model_price.predict(X_weight)
    
    # Regression 2: Versand durch Gewicht erklären
    model_freight = LinearRegression().fit(X_weight, y_freight)
    residuals_freight = y_freight - model_freight.predict(X_weight)
    
    # Partielle Korrelation ist die Pearson-Korrelation der Residuen
    partial_corr, p_val_partial = stats.pearsonr(residuals_price, residuals_freight)
    
    print(f"Partielle Korrelation: {partial_corr:.3f} (p-Wert: {p_val_partial:.4e})")
    print(f"-> Sobald wir das Gewicht herausrechnen, fällt die Korrelation von {corr_pearson.loc['price', 'freight_value']:.3f} auf {partial_corr:.3f}!")

    # -------------------------------------------------------------------
    # 5. Wunderschöne Visualisierung (Heatmaps & Scatter)
    # -------------------------------------------------------------------
    print("\nGeneriere Visualisierungs-Grid...")
    sns.set_theme(style="whitegrid")
    
    fig = plt.figure(figsize=(18, 10))
    
    # Grid Layout: Oben zwei Heatmaps (Pearson vs Spearman), Unten ein Scatterplot für die Partielle Korrelation
    ax_pearson = plt.subplot2grid((2, 3), (0, 0))
    ax_spearman = plt.subplot2grid((2, 3), (0, 1))
    ax_kendall = plt.subplot2grid((2, 3), (0, 2))
    ax_scatter = plt.subplot2grid((2, 3), (1, 0), colspan=3)
    
    # --- Plot 1: Pearson Heatmap ---
    sns.heatmap(corr_pearson, annot=True, fmt=".2f", cmap="coolwarm", center=0, 
                cbar=False, square=True, ax=ax_pearson, annot_kws={"size": 11})
    ax_pearson.set_title("1. Pearson (Linear)", fontweight="bold")
    ax_pearson.set_xticklabels(['Preis', 'Versand', 'Gewicht', 'Fotos'], rotation=45)
    ax_pearson.set_yticklabels(['Preis', 'Versand', 'Gewicht', 'Fotos'], rotation=0)

    # --- Plot 2: Spearman Heatmap ---
    sns.heatmap(corr_spearman, annot=True, fmt=".2f", cmap="coolwarm", center=0, 
                cbar=False, square=True, ax=ax_spearman, annot_kws={"size": 11})
    ax_spearman.set_title("2. Spearman (Rangbasiert, Robust)", fontweight="bold")
    ax_spearman.set_xticklabels(['Preis', 'Versand', 'Gewicht', 'Fotos'], rotation=45)
    ax_spearman.set_yticklabels([])

    # --- Plot 3: Kendall Heatmap ---
    sns.heatmap(corr_kendall, annot=True, fmt=".2f", cmap="coolwarm", center=0, 
                cbar=True, square=True, ax=ax_kendall, annot_kws={"size": 11})
    ax_kendall.set_title("3. Kendall-Tau (Konkordanz)", fontweight="bold")
    ax_kendall.set_xticklabels(['Preis', 'Versand', 'Gewicht', 'Fotos'], rotation=45)
    ax_kendall.set_yticklabels([])

    # --- Plot 4: Partielle Korrelation (Scatterplot der Residuen) ---
    # Wir zeigen den echten Zusammenhang zwischen Preis und Versandkosten OHNE den Einfluss des Gewichts.
    sns.regplot(x=residuals_price, y=residuals_freight, ax=ax_scatter, 
                scatter_kws={'alpha': 0.3, 'color': '#3498db', 's': 15}, 
                line_kws={'color': '#e74c3c', 'linewidth': 3})
    
    ax_scatter.set_title("4. Partielle Korrelation: Preis vs. Versandkosten (Einfluss von 'Gewicht' herausgerechnet)", 
                         fontsize=14, fontweight="bold")
    ax_scatter.set_xlabel("Preis (Residuen: um Gewicht bereinigt)")
    ax_scatter.set_ylabel("Versandkosten (Residuen: um Gewicht bereinigt)")
    
    # Wir zoomen etwas heran, um extreme Ausreißer in den Residuen abzuschneiden
    ax_scatter.set_xlim(-200, 500)
    ax_scatter.set_ylim(-30, 50)
    
    # Info-Box in den Scatterplot
    info_text = (
        f"Pearson (Original): {corr_pearson.loc['price', 'freight_value']:.2f}\n"
        f"Partiell (Bereinigt): {partial_corr:.2f}\n\n"
        f"Fazit: Ein großer Teil der ursprünglichen\n"
        f"Korrelation war nur eine Illusion, da\n"
        f"schwere Produkte automatisch teurer sind."
    )
    props = dict(boxstyle='round,pad=0.8', facecolor='white', alpha=0.9, edgecolor='gray', linewidth=1.5)
    ax_scatter.text(0.98, 0.05, info_text, transform=ax_scatter.transAxes, fontsize=11,
                    verticalalignment='bottom', horizontalalignment='right', bbox=props)

    # Layout optimieren und speichern
    plt.suptitle("Korrelationsanalyse: Wie Variablen wirklich zusammenhängen", 
                 fontsize=18, fontweight="bold", y=1.05)
    plt.tight_layout()
    sns.despine()
    
    plt.savefig(GRAPHIC_PATH, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✅ Grafik erfolgreich gespeichert unter:\n{GRAPHIC_PATH}")

if __name__ == "__main__":
    main()
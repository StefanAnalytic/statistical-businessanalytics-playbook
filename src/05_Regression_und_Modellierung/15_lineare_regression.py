"""
Griff 15: Welche Faktoren erklären unsere Zielgröße? (Lineare Regression & Diagnostics)

Was ist das und was macht das Skript?
Die Lineare Regression zieht nicht einfach nur eine "Trendlinie" durch eine Punktewolke. 
Sie ist das wichtigste Werkzeug, um zu verstehen, WIE stark verschiedene Faktoren (Features) 
eine Zielgröße (Target) beeinflussen. 

Business Case in diesem Skript:
Wir wollen den Produktpreis ('price') anhand der physischen Eigenschaften der Produkte erklären: 
Gewicht ('product_weight_g'), Volumen (Länge x Breite x Höhe) und Anzahl der Fotos ('product_photos_qty').

Aber ein Modell zu trainieren reicht nicht! Wir müssen prüfen, ob das Modell "gesund" ist. 
Dafür machen wir eine "Regressions-Diagnostik" (Die 4 wichtigsten Checks):
1. R-Squared: Wie viel Prozent der Preisschwankungen können wir durch unsere Variablen erklären?
2. Multikollinearität (VIF): Sind unsere Variablen zu ähnlich? (z.B. Gewicht und Volumen).
3. Heteroskedastizität: Macht unser Modell bei teuren Produkten größere Fehler als bei billigen?
4. Cook's Distance: Gibt es einzelne Ausreißer-Produkte, die das ganze Modell verzerren?
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Statsmodels ist für statistische Regressionen (Erklärung) viel besser als Sklearn (Vorhersage),
# da es uns direkt eine wunderbare, detaillierte Zusammenfassung (Summary) liefert.
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor

# -------------------------------------------------------------------
# 1. Setup & Pfade
# -------------------------------------------------------------------
# Wir wechseln nun in den Ordner "05_Regression_und_Modellierung"
ITEMS_PATH = "/Users/stefanbechthold/statistik_business_analytics/data/olist_order_items_dataset.csv"
PRODUCTS_PATH = "/Users/stefanbechthold/statistik_business_analytics/data/olist_products_dataset.csv"

GRAPHIC_DIR = "/Users/stefanbechthold/statistik_business_analytics/Grafiken/05_Regression_und_Modellierung"
GRAPHIC_PATH = os.path.join(GRAPHIC_DIR, "15_regression_diagnostics.png")

os.makedirs(GRAPHIC_DIR, exist_ok=True)

def main():
    print("🚀 Starte Griff 15: Lineare Regression & Model Diagnostics...")
    
    # -------------------------------------------------------------------
    # 2. Daten laden & Feature Engineering (Volumen berechnen)
    # -------------------------------------------------------------------
    print("Lade Produkt- und Bestelldaten...")
    df_items = pd.read_csv(ITEMS_PATH)
    df_products = pd.read_csv(PRODUCTS_PATH)
    
    df_merged = pd.merge(df_items, df_products, on='product_id', how='inner')
    
    # Wir berechnen das Volumen des Produkts in cm³
    df_merged['product_volume_cm3'] = (
        df_merged['product_length_cm'] * df_merged['product_width_cm'] * df_merged['product_height_cm']
    )
    
    # Relevante Spalten auswählen und fehlende Werte entfernen
    cols = ['price', 'product_weight_g', 'product_volume_cm3', 'product_photos_qty']
    df_model = df_merged[cols].dropna().copy()
    
    # Um extreme Ausreißer zu vermeiden (die eine einfache lineare Regression zerstören würden),
    # filtern wir auf "normale" Produkte: Preis unter 1000 BRL, Gewicht unter 20kg.
    df_model = df_model[(df_model['price'] < 1000) & (df_model['product_weight_g'] < 20000)]
    
    # Wir ziehen uns ein Zufallssample, damit der Plot nicht überladen wird (Overplotting)
    df_sample = df_model.sample(n=3000, random_state=42)
    
    print(f"Trainiere Modell mit {len(df_sample)} Datenpunkten...")

    # -------------------------------------------------------------------
    # 3. Das Lineare Regressions-Modell bauen (OLS - Ordinary Least Squares)
    # -------------------------------------------------------------------
    # Zielvariable (Y)
    y = df_sample['price']
    
    # Erklärende Variablen (X)
    X = df_sample[['product_weight_g', 'product_volume_cm3', 'product_photos_qty']]
    
    # WICHTIG: Statsmodels erfordert, dass wir manuell eine Konstante (Achsenabschnitt/Intercept) hinzufügen.
    # Ohne Konstante würde die Regressionsgerade gezwungen werden, exakt durch den Nullpunkt (0,0) zu gehen.
    X_with_const = sm.add_constant(X)
    
    # Modell anpassen (Trainieren)
    model = sm.OLS(y, X_with_const).fit()
    
    # Die magische Summary ausgeben (Zeigt R-Squared, p-Werte der Features, etc.)
    print("\n" + "="*80)
    print("REGRESSIONS-ZUSAMMENFASSUNG (Statsmodels Summary)")
    print("="*80)
    print(model.summary())
    print("="*80 + "\n")

    # -------------------------------------------------------------------
    # 4. Multikollinearität prüfen (Variance Inflation Factor - VIF)
    # -------------------------------------------------------------------
    # Wenn zwei X-Variablen stark korrelieren (z.B. Gewicht und Volumen), 
    # weiß das Modell nicht, wem es den "Erfolg" zuschreiben soll. Das macht die Koeffizienten instabil.
    # Ein VIF über 5 (manchmal 10) ist ein rotes Tuch!
    
    print("Prüfe Multikollinearität (VIF):")
    vif_data = pd.DataFrame()
    vif_data["Feature"] = X_with_const.columns
    vif_data["VIF"] = [variance_inflation_factor(X_with_const.values, i) for i in range(X_with_const.shape[1])]
    print(vif_data[1:]) # Wir ignorieren den VIF der Konstante (Index 0)
    print("-" * 50)

    # -------------------------------------------------------------------
    # 5. Diagnostics-Werte für die Plots extrahieren
    # -------------------------------------------------------------------
    predictions = model.predict(X_with_const)
    residuals = model.resid # Die Fehler (Tatsächlicher Preis minus Vorhergesagter Preis)
    
    # Cook's Distance berechnen (Misst den Einfluss jedes einzelnen Datenpunkts auf das Modell)
    influence = model.get_influence()
    cooks_d = influence.cooks_distance[0]

    # -------------------------------------------------------------------
    # 6. Wunderschöne Diagnostik-Visualisierung (2x2 Grid)
    # -------------------------------------------------------------------
    print("Generiere Visualisierungs-Grid (Diagnostics)...")
    sns.set_theme(style="whitegrid")
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    # --- Plot 1: Tatsächlich vs. Vorhergesagt (Actual vs Predicted) ---
    axes[0, 0].scatter(predictions, y, alpha=0.3, color='#3498db', s=20)
    # Ideallinie (wenn das Modell perfekt wäre, lägen alle Punkte auf dieser Linie)
    min_val = min(predictions.min(), y.min())
    max_val = max(predictions.max(), y.max())
    axes[0, 0].plot([min_val, max_val], [min_val, max_val], color='#e74c3c', linewidth=2, linestyle='--')
    
    axes[0, 0].set_title("1. Modellgüte: Vorhersage vs. Echte Preise", fontweight="bold")
    axes[0, 0].set_xlabel("Vorhergesagter Preis (Fitted Values)")
    axes[0, 0].set_ylabel("Tatsächlicher Preis")

    # --- Plot 2: Residuen vs. Vorhersage (Heteroskedastizität) ---
    # Hier prüfen wir, ob die Fehler bei teuren Produkten größer werden.
    # Wir wollen im Idealfall eine völlig zufällige "Wolke" um die Nulllinie sehen.
    # Ein Trichter-Muster (Funnel) = Heteroskedastizität (Schlecht!)
    axes[0, 1].scatter(predictions, residuals, alpha=0.3, color='#9b59b6', s=20)
    axes[0, 1].axhline(0, color='black', linewidth=2, linestyle='--')
    
    # Eine LOWESS-Kurve (Geglätteter Trend) zeigt an, ob die Fehler systematisch in eine Richtung driften
    sns.regplot(x=predictions, y=residuals, scatter=False, ax=axes[0, 1], lowess=True, color='#e74c3c')
    
    axes[0, 1].set_title("2. Fehlerverteilung (Residuen vs. Fitted)", fontweight="bold")
    axes[0, 1].set_xlabel("Vorhergesagter Preis (Fitted Values)")
    axes[0, 1].set_ylabel("Modell-Fehler (Residuen)")

    # --- Plot 3: QQ-Plot der Residuen ---
    # Prüft, ob unsere Fehler (Residuen) normalverteilt sind. 
    # Wenn die blauen Punkte stark von der roten Linie abweichen, haben wir ein Problem.
    sm.qqplot(residuals, line='45', fit=True, ax=axes[1, 0], color='#2ecc71', alpha=0.4)
    axes[1, 0].set_title("3. Normalität der Fehler (QQ-Plot)", fontweight="bold")
    # statsmodels hardcodet oft die farbe rot für die linie, wir machen sie deutlicher:
    axes[1, 0].get_lines()[1].set_color('#e74c3c')
    axes[1, 0].get_lines()[1].set_linewidth(2)

    # --- Plot 4: Cook's Distance (Ausreißer-Einfluss) ---
    # Zeigt, welche Datenpunkte das Modell massiv verzerren.
    # Faustregel: Cook's D > 4/n (oder > 1) ist kritisch.
    axes[1, 1].stem(np.arange(len(cooks_d)), cooks_d, markerfmt=",", basefmt=" ", 
                    linefmt="-", label='Cooks Distance')
    
    # Kritische Grenze einzeichnen (4 / n)
    threshold = 4 / len(df_sample)
    axes[1, 1].axhline(threshold, color='#e74c3c', linestyle='--', label=f'Threshold ({threshold:.4f})')
    
    axes[1, 1].set_title("4. Einflussreiche Ausreißer (Cook's Distance)", fontweight="bold")
    axes[1, 1].set_xlabel("Datenpunkt Index")
    axes[1, 1].set_ylabel("Cook's Distance")
    axes[1, 1].legend()

    # Layout optimieren und speichern
    plt.suptitle("Linear Regression Diagnostics: Preisvorhersage anhand Produkteigenschaften", 
                 fontsize=18, fontweight="bold", y=1.03)
    plt.tight_layout()
    sns.despine()
    
    plt.savefig(GRAPHIC_PATH, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✅ Grafik erfolgreich gespeichert unter:\n{GRAPHIC_PATH}")

if __name__ == "__main__":
    main()
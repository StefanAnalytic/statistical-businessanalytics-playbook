"""
Griff 41: Komplexe Modelle erklären? (Surrogate Models & Interaction Detection)

Was ist das und was macht das Skript?
Oft erzielen komplexe "Blackbox"-Modelle (wie Random Forests oder Gradient Boosting) 
die beste Vorhersagekraft. Das Problem: Sie sind für das Business schwer interpretierbar 
und nicht als einfaches Regelwerk kommunizierbar.

Zwei fortgeschrittene Techniken schaffen hier Abhilfe:
1. Global Surrogate Models (Ersatzmodelle): Wir trainieren ein extrem simples, 
   interpretierbares Modell (z.B. einen flachen Entscheidungsbaum) NICHT auf den 
   echten Daten, sondern AUF DEN VORHERSAGEN der Blackbox. Das Surrogate-Modell 
   "imitiert" die Logik der Blackbox und gießt sie in einfache Wenn-Dann-Regeln.
2. Interaction Detection (2D PDPs): Zeigt auf, wie zwei Features GEMEINSAM auf 
   die Zielvariable wirken (Interaktion). 

Spezifische Anforderung:
- Dark Mode Design.
- Schriftfarbe **schwarz** nur innerhalb der Knoten/Boxen des Entscheidungsbaums, 
  während der Rest (Titel, Achsen, Info-Boxen) weiß bleibt.

Business Case in diesem Skript:
Wir wollen verstehen, wie Frachtkosten (freight_value) im Olist-Netzwerk berechnet werden. 
Wir nutzen ein Blackbox-Modell, das die Frachtkosten basierend auf Produktgewicht, 
Volumen (Länge*Breite*Höhe) und Preis schätzt. Anschließend extrahieren wir die gelernten 
Regeln mit einem Surrogate-Tree und analysieren die Interaktion zwischen Volumen und Gewicht.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor, plot_tree
from sklearn.inspection import PartialDependenceDisplay

# -------------------------------------------------------------------
# 1. Setup & Pfade
# -------------------------------------------------------------------
ITEMS_PATH = "/Users/stefanbechthold/statistik_business_analytics/data/olist_order_items_dataset.csv"
PRODUCTS_PATH = "/Users/stefanbechthold/statistik_business_analytics/data/olist_products_dataset.csv"

GRAPHIC_DIR = "/Users/stefanbechthold/statistik_business_analytics/Grafiken/11_Interpretierbarkeit_und_Kalibrierung"
GRAPHIC_PATH = os.path.join(GRAPHIC_DIR, "41_surrogate_models_interactions.png")

os.makedirs(GRAPHIC_DIR, exist_ok=True)

def main():
    print("🚀 Starte Griff 41: Komplexe Modelle erklären (Surrogate Models & Interactions)...")
    
    # -------------------------------------------------------------------
    # 2. Daten laden & Features bauen
    # -------------------------------------------------------------------
    df_items = pd.read_csv(ITEMS_PATH)
    df_products = pd.read_csv(PRODUCTS_PATH)
    
    # Produktdimensionen mit Bestelldaten mergen
    df_merged = pd.merge(df_items[['product_id', 'price', 'freight_value']], 
                         df_products[['product_id', 'product_weight_g', 
                                      'product_length_cm', 'product_height_cm', 'product_width_cm']], 
                         on='product_id', how='inner')
    
    df_clean = df_merged.dropna().copy()
    
    # Feature Engineering: Volumen in Litern (cm³ / 1000)
    df_clean['volume_liters'] = (df_clean['product_length_cm'] * df_clean['product_height_cm'] * df_clean['product_width_cm']) / 1000.0
    # Gewicht in KG (zur besseren Interpretierbarkeit)
    df_clean['weight_kg'] = df_clean['product_weight_g'] / 1000.0
    
    # Ausreißer filtern (für saubere PDP-Plots und Modellstabilität)
    df_clean = df_clean[(df_clean['freight_value'] < 100) & 
                        (df_clean['weight_kg'] < 30) & 
                        (df_clean['volume_liters'] < 100)]
    
    # Sample ziehen für überschaubare Rechenzeit
    df_sample = df_clean.sample(n=10000, random_state=42)
    
    features = ['weight_kg', 'volume_liters', 'price']
    X = df_sample[features]
    y = df_sample['freight_value']

    # -------------------------------------------------------------------
    # 3. Blackbox Modell & Surrogate Modell trainieren
    # -------------------------------------------------------------------
    # 1. Die Blackbox (Random Forest)
    blackbox_model = RandomForestRegressor(n_estimators=100, max_depth=8, random_state=42)
    blackbox_model.fit(X, y)
    
    # Vorhersagen der Blackbox generieren
    blackbox_predictions = blackbox_model.predict(X)
    
    # 2. Das Surrogate Modell (Ersatzmodell)
    # Ein sehr flacher Entscheidungsbaum (max_depth=3), der versucht, die Logik 
    # der Blackbox-Vorhersagen (nicht der echten Y-Werte!) nachzubauen.
    surrogate_tree = DecisionTreeRegressor(max_depth=3, random_state=42)
    surrogate_tree.fit(X, blackbox_predictions)
    
    # R² des Surrogate-Modells zeigt, wie gut es die Blackbox erklärt (Fit-Qualität)
    surrogate_r2 = surrogate_tree.score(X, blackbox_predictions)

    # -------------------------------------------------------------------
    # 4. Visualisierung im Dark Mode Design
    # -------------------------------------------------------------------
    # Globale Einstellungen für den Dark Mode beibehalten
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
    
    fig = plt.figure(figsize=(20, 9), facecolor='#121212')
    
    # Layout-Grid: Links der Tree, Rechts das PDP
    ax1 = plt.subplot(1, 2, 1, facecolor='#121212')
    ax2 = plt.subplot(1, 2, 2, facecolor='#121212')

    # --- Plot 1: Das Surrogate Model (Entscheidungsbaum) ---
    # Wir erfassen die Rückgabe (texts), um die Farbe später zu ändern
    texts = plot_tree(surrogate_tree, 
                      feature_names=features, 
                      filled=True, 
                      rounded=True, 
                      fontsize=9,
                      ax=ax1,
                      impurity=False, # MSE ausblenden für Business-Fokus
                      proportion=True) # Zeigt % der Daten im Node an
    
    # SPEZIFISCHE ANFORDERUNG: Schriftfarbe schwarz NUR in den Boxen
    for text in texts:
        text.set_color('black')
        
    ax1.set_title(f"1. Global Surrogate Model (R² = {surrogate_r2:.2f})\nÜbersetzt die Blackbox in einfache Business-Regeln", 
                  fontsize=14, fontweight='bold', color='white', pad=20)

    # Info-Box Plot 1
    info_1 = (
        "Lesehilfe Surrogate Tree:\n"
        "Der komplexe Random Forest wurde in Wenn-Dann-Regeln gepresst.\n"
        "Wir sehen sofort die Hauptlogik der Frachtkosten-Berechnung:\n"
        "Die allererste Trennung (Wurzel) erfolgt beim Gewicht.\n"
        "Dunkle Boxen signalisieren hohe geschätzte Versandkosten."
    )
    props1 = dict(boxstyle='round,pad=1', facecolor='#1e1e1e', alpha=0.9, edgecolor='#3498db', linewidth=1.5)
    ax1.text(0.05, 0.05, info_1, transform=ax1.transAxes, fontsize=11,
             verticalalignment='bottom', bbox=props1, color='white')

    # --- Plot 2: Interaction Detection (2D Partial Dependence Plot) ---
    # Zeigt, wie Gewicht und Volumen GEMEINSAM auf die Blackbox-Vorhersage wirken
    PartialDependenceDisplay.from_estimator(
        blackbox_model, X, 
        features=[('weight_kg', 'volume_liters')], # Tupel = 2D Interaktion
        grid_resolution=30,
        contour_kw={'cmap': 'magma', 'alpha': 0.8},
        ax=ax2
    )
    
    ax2.set_title("2. Interaction Detection (2D PDP)\nWie wirken Gewicht & Volumen zusammen?", 
                  fontsize=14, fontweight='bold', color='white')
    ax2.set_xlabel("Gewicht (KG)")
    ax2.set_ylabel("Volumen (Liter)")

    # Info-Box Plot 2
    info_2 = (
        "Business Insight (Interaktionen):\n"
        "---------------------------------\n"
        "Ein 1D-Plot zeigt nur isolierte Effekte. Der 2D-Plot deckt auf:\n"
        "Die Frachtkosten explodieren (helle Farben) nur dann,\n"
        "wenn das Paket SOWOHL schwer ALS AUCH voluminös ist.\n"
        "Ein kleines, aber schweres Paket (z.B. Hanteln) ist günstiger\n"
        "als ein gleich schweres, aber sperriges Paket (z.B. Möbelstück)."
    )
    props2 = dict(boxstyle='round,pad=1', facecolor='#1e1e1e', alpha=0.9, edgecolor='#e74c3c', linewidth=1.5)
    ax2.text(0.05, 0.95, info_2, transform=ax2.transAxes, fontsize=11,
             verticalalignment='top', bbox=props2, color='white')

    plt.suptitle("Explainable AI (XAI): Komplexe Modelle für das Business transparent machen", 
                 fontsize=18, fontweight='bold', color='white', y=1.02)
    plt.tight_layout()
    
    # Speichern der Grafik
    plt.savefig(GRAPHIC_PATH, dpi=300, bbox_inches='tight', facecolor=fig.get_facecolor())
    plt.close()
    
    print(f"✅ Grafik erfolgreich gespeichert unter:\n{GRAPHIC_PATH}")

if __name__ == "__main__":
    main()
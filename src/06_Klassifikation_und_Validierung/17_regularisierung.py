"""
Griff 17: Overfitting kontrollieren (Regularisierung: Ridge, Lasso, ElasticNet)

Was ist das und was macht das Skript?
Wenn wir sehr viele Einflussfaktoren (Features) haben, neigt eine normale Lineare Regression 
dazu, den Trainingsdaten "zu stark zu vertrauen". Sie lernt die echten Muster, aber auch 
das zufällige Rauschen auswendig. Das nennt man Overfitting. Bei neuen Daten performt 
das Modell dann miserabel.

Die Lösung ist "Regularisierung". Wir bestrafen das Modell für zu komplexe Vorhersagen 
(zu große Koeffizienten).
1. Ridge (L2-Penalty): Drückt alle Koeffizienten sanft in Richtung null. Perfekt, wenn 
   viele Variablen stark miteinander korrelieren (Multikollinearität).
2. Lasso (L1-Penalty): Die "Machete". Drückt unwichtige Koeffizienten auf exakt 0.0! 
   Dadurch macht Lasso automatisch "Feature Selection" (Variablenauswahl).

Business Case in diesem Skript:
Wir wollen den Produktpreis ('price') vorhersagen. Um absichtlich eine "High-Dimensionality" 
(viele Variablen) zu erzeugen, nutzen wir nicht nur Gewicht und Volumen, sondern machen 
aus der Produktkategorie ('product_category_name') über 70 einzelne Dummy-Variablen (1/0). 
Wir vergleichen, wie OLS (Standard), Ridge und Lasso mit dieser Flut an Informationen umgehen.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, RidgeCV, LassoCV
from sklearn.metrics import mean_squared_error, r2_score

# -------------------------------------------------------------------
# 1. Setup & Pfade
# -------------------------------------------------------------------
# Wir sind im Ordner "05_Regression_und_Modellierung"
ITEMS_PATH = "/Users/stefanbechthold/statistik_business_analytics/data/olist_order_items_dataset.csv"
PRODUCTS_PATH = "/Users/stefanbechthold/statistik_business_analytics/data/olist_products_dataset.csv"

GRAPHIC_DIR = "/Users/stefanbechthold/statistik_business_analytics/Grafiken/05_Regression_und_Modellierung"
GRAPHIC_PATH = os.path.join(GRAPHIC_DIR, "17_regularisierung_lasso_ridge.png")

os.makedirs(GRAPHIC_DIR, exist_ok=True)

def main():
    print("🚀 Starte Griff 17: Regularisierung (Ridge & Lasso)...")
    
    # -------------------------------------------------------------------
    # 2. Daten laden & High-Dimensional Feature Engineering
    # -------------------------------------------------------------------
    print("Lade Produktdaten und baue High-Dimensional Dataset...")
    df_items = pd.read_csv(ITEMS_PATH)
    df_products = pd.read_csv(PRODUCTS_PATH)
    
    # Mergen und unwichtige Zeilen werfen
    df_merged = pd.merge(df_items, df_products, on='product_id', how='inner')
    df_clean = df_merged.dropna(subset=['price', 'product_weight_g', 'product_length_cm', 'product_category_name']).copy()
    
    # Extremwerte abschneiden für saubere Modellierung
    df_clean = df_clean[(df_clean['price'] < 1000) & (df_clean['product_weight_g'] < 20000)]
    
    # Wir ziehen ein Sample von 5000 Produkten
    df_sample = df_clean.sample(n=5000, random_state=42)
    
    # Feature 1: Volumen berechnen
    df_sample['volume_cm3'] = df_sample['product_length_cm'] * df_sample['product_width_cm'] * df_sample['product_height_cm']
    
    # Feature 2: Dummy-Variablen für Kategorien (One-Hot-Encoding)
    # Das macht aus EINER Spalte (Kategorie) plötzlich ~70 Spalten (ist_auto, ist_moebel, etc.)
    # Genau hier glänzt Regularisierung!
    df_dummies = pd.get_dummies(df_sample['product_category_name'], drop_first=True)
    
    # X (Features) und y (Target) zusammenbauen
    X = pd.concat([df_sample[['product_weight_g', 'volume_cm3', 'freight_value']], df_dummies], axis=1)
    y = df_sample['price']
    
    print(f"Dataset Form: {X.shape[0]} Zeilen und {X.shape[1]} Features (Spalten)!")

    # -------------------------------------------------------------------
    # 3. Train/Test Split & Standardisierung (Extrem wichtig!)
    # -------------------------------------------------------------------
    print("Mache Train/Test Split und Skalierung...")
    # Wir trainieren auf 80% der Daten und testen auf 20% unbekannten Daten
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # WICHTIG: Lasso und Ridge verlangen zwingend, dass alle Variablen auf der gleichen 
    # Skala sind (Standardisierung). Sonst bestraft das Modell "große Zahlen" unfair hart.
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # -------------------------------------------------------------------
    # 4. Modelle trainieren (OLS, Ridge, Lasso)
    # -------------------------------------------------------------------
    print("Trainiere OLS, Ridge und Lasso...")
    
    # 1. Standard Lineare Regression (ohne Strafe)
    lin_reg = LinearRegression()
    lin_reg.fit(X_train_scaled, y_train)
    
    # 2. Ridge Regression (L2)
    # RidgeCV testet automatisch verschiedene Straf-Stärken (Alphas) durch Cross-Validation
    ridge_reg = RidgeCV(alphas=np.logspace(-3, 3, 20), cv=5)
    ridge_reg.fit(X_train_scaled, y_train)
    
    # 3. Lasso Regression (L1)
    lasso_reg = LassoCV(alphas=np.logspace(-3, 3, 20), cv=5, max_iter=10000)
    lasso_reg.fit(X_train_scaled, y_train)
    
    # Auswertung auf dem Testset
    models = {
        'OLS (Keine Regularisierung)': lin_reg,
        f'Ridge (L2, α={ridge_reg.alpha_:.1f})': ridge_reg,
        f'Lasso (L1, α={lasso_reg.alpha_:.1f})': lasso_reg
    }
    
    print("\nTest-Performance (R² Score - Je näher an 1, desto besser):")
    for name, model in models.items():
        y_pred = model.predict(X_test_scaled)
        r2 = r2_score(y_test, y_pred)
        print(f"{name}: R² = {r2:.4f}")

    # -------------------------------------------------------------------
    # 5. Wunderschöne Visualisierung (Koeffizienten-Vergleich)
    # -------------------------------------------------------------------
    print("\nGeneriere Visualisierung (Koeffizienten-Schrumpfung)...")
    sns.set_theme(style="whitegrid")
    
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))

    # Wir bauen ein DataFrame aus den Modell-Gewichten (Koeffizienten)
    coef_df = pd.DataFrame({
        'Feature_Index': range(X.shape[1]),
        'OLS': lin_reg.coef_,
        'Ridge': ridge_reg.coef_,
        'Lasso': lasso_reg.coef_
    })

    # --- Plot 1: Verteilung der Koeffizienten-Größen ---
    # Wir zeigen, wie stark die Modelle die Gewichte bestrafen
    
    axes[0].plot(coef_df['Feature_Index'], coef_df['OLS'], 'o', alpha=0.5, color='#34495e', label='OLS (Wild & Groß)')
    axes[0].plot(coef_df['Feature_Index'], coef_df['Ridge'], 's', alpha=0.7, color='#3498db', label='Ridge (Gedämpft)')
    axes[0].plot(coef_df['Feature_Index'], coef_df['Lasso'], '^', alpha=0.9, color='#e74c3c', label='Lasso (Viele auf 0!)')
    
    axes[0].axhline(0, color='black', linestyle='--', linewidth=1)
    axes[0].set_title("1. Die Wirkung: Wie Regularisierung Gewichte drückt", fontsize=14, fontweight="bold")
    axes[0].set_xlabel("Feature Index (ca. 70 Kategorien + Gewicht/Volumen)")
    axes[0].set_ylabel("Koeffizienten-Größe (Einfluss im Modell)")
    axes[0].legend(loc='upper right')
    
    # Y-Achse etwas einzoomen, da OLS oft extreme Ausreißer hat
    y_limit = max(abs(ridge_reg.coef_).max(), abs(lasso_reg.coef_).max()) * 1.5
    axes[0].set_ylim(-y_limit, y_limit)

    # --- Plot 2: Feature Selection durch Lasso ---
    # Lasso zwingt unwichtige Variablen exakt auf 0. Wir visualisieren das als Bar-Chart (Anzahl Nicht-Null Features).
    
    non_zero_ols = np.sum(lin_reg.coef_ != 0)
    non_zero_ridge = np.sum(np.abs(ridge_reg.coef_) > 1e-5) # Ridge drückt nahe 0, aber selten exakt 0
    non_zero_lasso = np.sum(lasso_reg.coef_ != 0)
    
    bars = axes[1].bar(['OLS', 'Ridge', 'Lasso'], 
                       [non_zero_ols, non_zero_ridge, non_zero_lasso], 
                       color=['#34495e', '#3498db', '#e74c3c'], alpha=0.85, edgecolor='black', width=0.5)
    
    axes[1].set_title("2. Feature Selection: Wie viele Features überleben?", fontsize=14, fontweight="bold")
    axes[1].set_ylabel("Anzahl der Features mit Einfluss (> 0)")
    
    # Zahlen in die Balken schreiben
    for bar in bars:
        height = bar.get_height()
        axes[1].text(bar.get_x() + bar.get_width()/2., height/2,
                     f"{int(height)} von {X.shape[1]}",
                     ha='center', va='center', color='white', fontweight='bold', fontsize=12)

    # Info-Box
    info_text = (
        "Erkenntnis:\n"
        "Während OLS und Ridge alle ~70 Features nutzen,\n"
        "erkennt Lasso durch die L1-Strafe, dass die meisten\n"
        "Produktkategorien keinen echten Vorhersagewert\n"
        "für den Preis haben und wirft sie komplett raus (=0).\n"
        "Das macht das Lasso-Modell viel schlanker und\n"
        "resistenter gegen Overfitting!"
    )
    props = dict(boxstyle='round,pad=0.8', facecolor='white', alpha=0.9, edgecolor='#e74c3c', linewidth=2)
    axes[1].text(0.5, 0.95, info_text, transform=axes[1].transAxes, fontsize=11,
                 verticalalignment='top', horizontalalignment='center', bbox=props, color='#2c3e50')

    # Layout optimieren und speichern
    plt.suptitle("Ridge vs. Lasso: Kampf gegen das Overfitting", 
                 fontsize=18, fontweight="bold", y=1.05)
    plt.tight_layout()
    sns.despine()
    
    plt.savefig(GRAPHIC_PATH, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✅ Grafik erfolgreich gespeichert unter:\n{GRAPHIC_PATH}")

if __name__ == "__main__":
    main()
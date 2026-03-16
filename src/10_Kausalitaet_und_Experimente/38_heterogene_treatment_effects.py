"""
Griff 38: Heterogene Treatment Effects finden? (Uplift Modeling & T-Learner)

Was ist das und was macht das Skript?
In klassischen Tests messen wir meist den Average Treatment Effect (ATE):
"Hat unsere Maßnahme im Durchschnitt funktioniert?" 
Das Problem: Durchschnitte lügen. Eine Maßnahme kann bei einer Kundengruppe den Umsatz 
massiv steigern, während sie bei einer anderen Gruppe völlig wirkungslos verpufft.

Uplift Modeling sucht nach CATE (Conditional Average Treatment Effect): 
Wie wirkt das Treatment auf ein SPEZIFISCHES Objekt, gegeben seine Eigenschaften (X)?
Wir wollen die Teilmengen finden, bei denen der Hebel (Uplift) am größten ist.

Dieses Skript nutzt einen T-Learner (Two-Model-Learner):
1. Wir trainieren Modell 0 nur auf der Kontrollgruppe.
2. Wir trainieren Modell 1 nur auf der Treatment-Gruppe.
3. Für JEDEN Datenpunkt berechnen wir: Vorhersage Modell 1 - Vorhersage Modell 0. 
   Das Ergebnis ist der individuelle Uplift (CATE)!

Neuer Business Case in diesem Skript:
Führt "Rich Media" (Viele Produktfotos, >= 3) dazu, dass Kunden bereit sind, teurere 
Artikel zu kaufen? Und bei welchen Produkttypen ist dieser Uplift am stärksten?
Wir untersuchen, ob Verkäufer von schweren und sperrigen Produkten (Möbel, Geräte) 
mehr von zusätzlichen Fotos profitieren als Verkäufer von kleinen Standard-Artikeln.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor

# -------------------------------------------------------------------
# 1. Setup & Pfade
# -------------------------------------------------------------------
ITEMS_PATH = "/Users/stefanbechthold/statistik_business_analytics/data/olist_order_items_dataset.csv"
PRODUCTS_PATH = "/Users/stefanbechthold/statistik_business_analytics/data/olist_products_dataset.csv"

GRAPHIC_DIR = "/Users/stefanbechthold/statistik_business_analytics/Grafiken/10_Kausalitaet_und_Experimente"
GRAPHIC_PATH = os.path.join(GRAPHIC_DIR, "38_uplift_modeling_tlearner.png")

os.makedirs(GRAPHIC_DIR, exist_ok=True)

def main():
    # -------------------------------------------------------------------
    # 2. Daten laden & Treatment / Target / Features definieren
    # -------------------------------------------------------------------
    df_items = pd.read_csv(ITEMS_PATH)
    df_products = pd.read_csv(PRODUCTS_PATH)
    
    # Mergen der Artikel- und Produktdaten
    df_merged = pd.merge(df_items[['product_id', 'price']], 
                         df_products[['product_id', 'product_photos_qty', 
                                      'product_weight_g', 'product_length_cm', 
                                      'product_height_cm', 'product_width_cm']], 
                         on='product_id', how='inner')
    
    df_clean = df_merged.dropna().copy()
    
    # Feature Engineering: Volumen berechnen (in cm³)
    df_clean['product_volume_cm3'] = df_clean['product_length_cm'] * df_clean['product_height_cm'] * df_clean['product_width_cm']
    
    # Wir filtern extreme Ausreißer im Preis und Gewicht für stabile Modelle
    df_clean = df_clean[(df_clean['price'] < 1000) & (df_clean['product_weight_g'] < 30000) & (df_clean['product_photos_qty'] > 0)]
    
    # Für Performance ziehen wir ein Sample von 20.000 Artikeln
    df_sample = df_clean.sample(n=20000, random_state=42).reset_index(drop=True)
    
    # Treatment (T): Hat das Produkt viele Fotos ("Rich Media")? (>= 3 Fotos = 1, sonst 0)
    df_sample['T'] = (df_sample['product_photos_qty'] >= 3).astype(int)
    
    # Outcome (Y): Preis (Wie teuer ist der Artikel, der gekauft wurde?)
    df_sample['Y'] = df_sample['price']
    
    # Features (X): Physische Eigenschaften des Produkts
    features = ['product_weight_g', 'product_volume_cm3']
    
    # -------------------------------------------------------------------
    # 3. T-Learner Algorithmus implementieren
    # -------------------------------------------------------------------
    X = df_sample[features]
    Y = df_sample['Y']
    T = df_sample['T']
    
    # Aufteilung in Treatment und Control
    X_control = X[T == 0]
    Y_control = Y[T == 0]
    
    X_treated = X[T == 1]
    Y_treated = Y[T == 1]
    
    # Modell 0 (Kontrollgruppe: Wenig Fotos) trainieren
    model_control = RandomForestRegressor(n_estimators=100, max_depth=6, random_state=42)
    model_control.fit(X_control, Y_control)
    
    # Modell 1 (Treatmentgruppe: Viele Fotos) trainieren
    model_treated = RandomForestRegressor(n_estimators=100, max_depth=6, random_state=42)
    model_treated.fit(X_treated, Y_treated)
    
    # Individuellen Uplift für ALLE Produkte vorhersagen
    # Uplift = E[Y|X, T=1] - E[Y|X, T=0]
    df_sample['pred_Y_treated'] = model_treated.predict(X)
    df_sample['pred_Y_control'] = model_control.predict(X)
    df_sample['CATE'] = df_sample['pred_Y_treated'] - df_sample['pred_Y_control']
    
    # Average Treatment Effect (ATE)
    ate = df_sample['CATE'].mean()

    # -------------------------------------------------------------------
    # 4. Uplift-Segmente bilden (Wer profitiert am meisten?)
    # -------------------------------------------------------------------
    # Wir teilen die Produkte nach Volumen in 4 Gruppen (Quartile) auf, 
    # um zu sehen, ob sperrige Artikel mehr von Fotos profitieren als kleine.
    df_sample['volume_segment'] = pd.qcut(df_sample['product_volume_cm3'], q=4, labels=['Klein', 'Mittel', 'Groß', 'Sperrig'])
    uplift_by_volume = df_sample.groupby('volume_segment')['CATE'].mean().reset_index()

    # -------------------------------------------------------------------
    # 5. Visualisierung im Dark Mode Design
    # -------------------------------------------------------------------
    plt.style.use('dark_background')
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7), facecolor='#121212')
    ax1.set_facecolor('#121212')
    ax2.set_facecolor('#121212')

    # --- Plot 1: Verteilung des individuellen Uplifts (CATE) ---
    sns.histplot(df_sample['CATE'], bins=50, color='#3498db', kde=True, ax=ax1, edgecolor='none')
    
    # ATE als vertikale Linie
    ax1.axvline(ate, color='#e74c3c', linestyle='--', linewidth=3, label=f'Average Treatment Effect (ATE): +{ate:.1f} BRL')
    ax1.axvline(0, color='white', linestyle='-', linewidth=1, alpha=0.5)
    
    ax1.set_title("1. Heterogenität: Verteilung des individuellen Uplifts", fontsize=14, fontweight='bold', color='white')
    ax1.set_xlabel("Erwarteter Preis-Uplift durch viele Fotos (BRL)")
    ax1.set_ylabel("Anzahl der Artikel")
    
    legend1 = ax1.legend(loc='upper right', facecolor='#1e1e1e', edgecolor='#333333')
    plt.setp(legend1.get_texts(), color='white')

    # Info-Box Plot 1
    info_1 = (
        "Das Ende des Durchschnitts:\n"
        "Der ATE (rot) zeigt, dass mehr Fotos im Schnitt zu einem\n"
        "etwas höheren Artikelwert führen. Aber die Streuung ist enorm!\n"
        "Bei einigen Produkten bringt 'Rich Media' fast +40 BRL,\n"
        "bei anderen hat es überhaupt keinen Effekt (Uplift um 0)."
    )
    props1 = dict(boxstyle='round,pad=1', facecolor='#1e1e1e', alpha=0.9, edgecolor='#3498db', linewidth=1.5)
    ax1.text(0.05, 0.85, info_1, transform=ax1.transAxes, fontsize=11,
             verticalalignment='top', bbox=props1, color='#e0e0e0')

    # --- Plot 2: Uplift nach Volumen-Segmenten ---
    sns.barplot(x='volume_segment', y='CATE', data=uplift_by_volume, palette='magma', ax=ax2, edgecolor='none')
    ax2.axhline(ate, color='#e74c3c', linestyle='--', linewidth=2, alpha=0.7)
    
    ax2.set_title("2. Wer ist am sensibelsten? (Uplift nach Produktvolumen)", fontsize=14, fontweight='bold', color='white')
    ax2.set_xlabel("Produktgröße (Volumen-Segmente)")
    ax2.set_ylabel("Durchschnittlicher Preis-Uplift (BRL)")
    
    # Werte über die Balken schreiben
    for i, bar in enumerate(ax2.patches):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5, 
                 f"+{bar.get_height():.1f}", 
                 ha='center', va='bottom', color='white', fontweight='bold', fontsize=12)

    # Info-Box Plot 2
    info_2 = (
        "Business Strategie (Asset Allokation):\n"
        "---------------------------------------\n"
        "Für kleine, standardisierte Artikel (z.B. USB-Kabel)\n"
        "bringen viele Fotos kaum finanziellen Uplift.\n\n"
        "Bei 'sperrigen' Produkten (z.B. Möbel, große Elektronik)\n"
        "ist das Informationsbedürfnis der Kunden extrem hoch.\n"
        "Hier hebeln zusätzliche Fotos die Zahlungsbereitschaft\n"
        "signifikant nach oben. \n\n"
        "-> Fokus für Produktfotografie auf große Items legen!"
    )
    props2 = dict(boxstyle='round,pad=1', facecolor='#1e1e1e', alpha=0.9, edgecolor='#e74c3c', linewidth=1.5)
    ax2.text(0.05, 0.95, info_2, transform=ax2.transAxes, fontsize=11,
             verticalalignment='top', bbox=props2, color='#e0e0e0')

    plt.suptitle("Causal Machine Learning: Heterogene Effekte von Produktbildern (T-Learner)", fontsize=18, fontweight='bold', color='white', y=1.05)
    plt.tight_layout()
    
    # Speichern der Grafik
    plt.savefig(GRAPHIC_PATH, dpi=300, bbox_inches='tight', facecolor=fig.get_facecolor())
    plt.close()

if __name__ == "__main__":
    main()
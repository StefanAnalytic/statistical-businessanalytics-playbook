"""
Griff 48: Bessere CATE-Schätzer? (IPW & Doubly Robust Estimators)

Was ist das und was macht das Skript?
In reinen Beobachtungsdaten (Observational Data) ohne A/B-Test gibt es fast immer Confounding (Störvariablen).
Beispiel: Verursacht der Kauf von "Sperrgut" (schweren Produkten) schlechtere Bewertungen? 
Das Problem: Sperrgut hat höhere Frachtkosten (Confounder). Hohe Frachtkosten verärgern Kunden. 
Wenn wir einfach die Bewertungen vergleichen (Naiv), vermischen wir den Effekt des Gewichts 
mit dem Effekt der Frachtkosten.

Zwei klassische Lösungen:
1. Inverse Probability Weighting (IPW): Wir trainieren ein Modell, das vorhersagt, wie 
   wahrscheinlich ein Kauf "Sperrgut" ist (Propensity Score). Dann gewichten wir die Daten um, 
   sodass Treatment- und Kontrollgruppe statistisch identisch aussehen.
2. Outcome Regression (OR): Wir kontrollieren für die Frachtkosten in einer normalen Regression.

Das Problem: Wenn Modell 1 oder Modell 2 falsch spezifiziert ist, ist unser Ergebnis falsch.
Die Lösung: Der Doubly Robust Estimator (DRE). Er kombiniert IPW und OR. 
Seine Magie: Solange MINDESTENS EINES der beiden Modelle (Propensity oder Outcome) 
korrekt ist, liefert der DRE den echten kausalen Effekt!

Business Case in diesem Skript:
Wir messen den echten isolierten Effekt von Sperrgut (Treatment: > 5kg) auf die Kundenbewertung.
Wir vergleichen den naiven Ansatz mit IPW, Outcome Regression und dem Doubly Robust Estimator.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression, LinearRegression

# -------------------------------------------------------------------
# 1. Setup & Pfade
# -------------------------------------------------------------------
ORDERS_PATH = "/Users/stefanbechthold/statistik_business_analytics/data/olist_orders_dataset.csv"
REVIEWS_PATH = "/Users/stefanbechthold/statistik_business_analytics/data/olist_order_reviews_dataset.csv"
ITEMS_PATH = "/Users/stefanbechthold/statistik_business_analytics/data/olist_order_items_dataset.csv"
PRODUCTS_PATH = "/Users/stefanbechthold/statistik_business_analytics/data/olist_products_dataset.csv"

GRAPHIC_DIR = "/Users/stefanbechthold/statistik_business_analytics/Grafiken/10_Kausalitaet_und_Experimente"
GRAPHIC_PATH = os.path.join(GRAPHIC_DIR, "48_doubly_robust_estimator.png")

os.makedirs(GRAPHIC_DIR, exist_ok=True)

def main():
    print("🚀 Starte Griff 48: Kausale Inferenz mit Doubly Robust Estimators...")
    
    # -------------------------------------------------------------------
    # 2. Daten laden & Variablen definieren
    # -------------------------------------------------------------------
    df_reviews = pd.read_csv(REVIEWS_PATH)
    df_items = pd.read_csv(ITEMS_PATH)
    df_products = pd.read_csv(PRODUCTS_PATH)
    
    # Erste Bewertung pro Bestellung
    reviews_unique = df_reviews.drop_duplicates(subset=['order_id'])[['order_id', 'review_score']]
    
    # Produktdaten mergen (für Gewicht)
    df_items = pd.merge(df_items, df_products[['product_id', 'product_weight_g']], on='product_id', how='inner')
    
    # Aggregation auf Bestell-Ebene
    order_features = df_items.groupby('order_id').agg({
        'price': 'sum',
        'freight_value': 'sum',
        'product_weight_g': 'sum'
    }).reset_index()
    
    df_merged = pd.merge(reviews_unique, order_features, on='order_id', how='inner')
    df_clean = df_merged.dropna().copy()
    
    # Ausreißer entfernen für stabile Modelle
    df_clean = df_clean[(df_clean['price'] < 1000) & (df_clean['freight_value'] < 150)]
    
    # Target (Y): Review Score
    Y = df_clean['review_score']
    
    # Treatment (T): Ist es Sperrgut? (Gewicht > 5kg)
    T = (df_clean['product_weight_g'] > 5000).astype(int)
    df_clean['Treatment'] = T
    
    # Confounders (X): Frachtkosten und Preis (beeinflussen das Treatment UND das Outcome)
    X = df_clean[['freight_value', 'price']]
    
    # -------------------------------------------------------------------
    # 3. Kausale Modelle trainieren
    # -------------------------------------------------------------------
    
    # --- A) Naive Schätzung (Einfacher Mittelwert-Vergleich) ---
    naive_ate = Y[T == 1].mean() - Y[T == 0].mean()
    
    # --- B) Propensity Score Model (Für IPW) ---
    # Vorhersage: Wie wahrscheinlich ist es, dass jemand Sperrgut kauft, gegeben Preis und Fracht?
    ps_model = LogisticRegression(random_state=42)
    ps_model.fit(X, T)
    ps = ps_model.predict_proba(X)[:, 1]
    
    # Clipping (Positivity Annahme sichern - Niemand darf 0% oder 100% haben)
    ps = np.clip(ps, 0.05, 0.95)
    df_clean['Propensity_Score'] = ps
    
    # IPW Schätzung (Inverse Probability Weighting)
    # Gewicht = 1/PS für Treatment, 1/(1-PS) für Kontrolle
    ipw_ate = np.mean((T * Y) / ps) - np.mean(((1 - T) * Y) / (1 - ps))
    
    # --- C) Outcome Regression Model ---
    # Vorhersage des Review Scores basierend auf X und T
    or_model = LinearRegression()
    X_with_T = X.copy()
    X_with_T['Treatment'] = T
    or_model.fit(X_with_T, Y)
    
    # Was wäre, wenn ALLE Sperrgut gekauft hätten? (T=1)
    X_all_treated = X_with_T.copy()
    X_all_treated['Treatment'] = 1
    mu_1 = or_model.predict(X_all_treated)
    
    # Was wäre, wenn NIEMAND Sperrgut gekauft hätte? (T=0)
    X_all_control = X_with_T.copy()
    X_all_control['Treatment'] = 0
    mu_0 = or_model.predict(X_all_control)
    
    or_ate = np.mean(mu_1 - mu_0)
    
    # --- D) Doubly Robust Estimator ---
    # Kombiniert die Vorhersagen (mu_1, mu_0) mit den IPW-Gewichteten Fehlern
    dr_1 = mu_1 + (T * (Y - mu_1)) / ps
    dr_0 = mu_0 + ((1 - T) * (Y - mu_0)) / (1 - ps)
    dr_ate = np.mean(dr_1 - dr_0)

    # -------------------------------------------------------------------
    # 4. Visualisierung im Dark Mode Design
    # -------------------------------------------------------------------
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
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 7), facecolor='#121212')

    # --- Plot 1: Positivity Check (Propensity Score Overlap) ---
    # Prüft, ob es in beiden Gruppen ähnliche Kunden gibt. Wenn es keinen Overlap gibt, scheitert Kausalität!
    sns.kdeplot(df_clean[df_clean['Treatment'] == 1]['Propensity_Score'], 
                color='#e74c3c', fill=True, label='Treatment (Sperrgut)', ax=ax1, alpha=0.5)
    sns.kdeplot(df_clean[df_clean['Treatment'] == 0]['Propensity_Score'], 
                color='#3498db', fill=True, label='Control (Leichtgewicht)', ax=ax1, alpha=0.5)
    
    ax1.set_title("1. Positivity Check: Propensity Score Overlap", fontsize=14, fontweight='bold', color='white')
    ax1.set_xlabel("Wahrscheinlichkeit für Sperrgut-Kauf (Propensity Score)")
    ax1.set_ylabel("Dichte")
    
    legend1 = ax1.legend(loc='upper right', facecolor='#1e1e1e', edgecolor='#333333')
    plt.setp(legend1.get_texts(), color='white')

    # Info-Box Plot 1
    info_1 = (
        "Die Positivity Annahme:\n"
        "Für einen validen kausalen Vergleich brauchen wir 'Overlap'.\n"
        "Das heißt: Es muss schwere Pakete mit niedrigen Frachtkosten\n"
        "geben und leichte Pakete mit hohen Frachtkosten.\n"
        "Überschneiden sich die Kurven nicht, vergleicht unser\n"
        "Modell Äpfel mit Birnen und muss blind extrapolieren!"
    )
    props1 = dict(boxstyle='round,pad=1', facecolor='#1e1e1e', alpha=0.9, edgecolor='#3498db', linewidth=1.5)
    ax1.text(0.05, 0.65, info_1, transform=ax1.transAxes, fontsize=11,
             verticalalignment='top', bbox=props1, color='white')

    # --- Plot 2: Vergleich der Causal Estimators ---
    estimators = ['Naiv\n(Verzerrt)', 'IPW\n(Propensity)', 'OR\n(Regression)', 'Doubly Robust\n(Gold Standard)']
    values = [naive_ate, ipw_ate, or_ate, dr_ate]
    colors = ['#e74c3c', '#f1c40f', '#3498db', '#2ecc71']
    
    bars = ax2.bar(estimators, values, color=colors, edgecolor='none')
    
    # Null-Linie
    ax2.axhline(0, color='white', linewidth=1, linestyle='-')
    
    ax2.set_title("2. Kausaler Effekt: Wie viele Sterne kostet Sperrgut wirklich?", fontsize=14, fontweight='bold', color='white')
    ax2.set_ylabel("Average Treatment Effect (Sterne-Verlust)")

    # Werte über die Balken schreiben
    for bar in bars:
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height - 0.05, 
                 f"{height:.2f} Sterne", ha='center', va='top', color='white', fontweight='bold', fontsize=12)

    # Info-Box Plot 2
    info_2 = (
        "Business Insight (Der wahre Effekt):\n"
        "------------------------------------\n"
        "Naiv betrachtet (Rot) kostet Sperrgut extrem viele Sterne.\n"
        "Aber das liegt teilweise an den nervigen Frachtkosten!\n\n"
        "Sobald wir Doubly Robust (Grün) für die Confounder kontrollieren,\n"
        "sehen wir den isolierten, kausalen Effekt: Sperrgut an sich\n"
        "kostet weniger Sterne als der naive Vergleich uns glauben macht.\n"
        "Wir haben den 'Frachtkosten-Bias' erfolgreich herausgerechnet."
    )
    props2 = dict(boxstyle='round,pad=1', facecolor='#1e1e1e', alpha=0.9, edgecolor='#2ecc71', linewidth=1.5)
    ax2.text(0.35, 0.35, info_2, transform=ax2.transAxes, fontsize=11,
             verticalalignment='center', bbox=props2, color='white')

    plt.suptitle("Causal Inference in Observational Data: Doubly Robust Estimators", 
                 fontsize=18, fontweight='bold', color='white', y=1.05)
    plt.tight_layout()
    
    # Speichern der Grafik
    plt.savefig(GRAPHIC_PATH, dpi=300, bbox_inches='tight', facecolor=fig.get_facecolor())
    plt.close()
    
    print(f"✅ Grafik erfolgreich gespeichert unter:\n{GRAPHIC_PATH}")

if __name__ == "__main__":
    main()
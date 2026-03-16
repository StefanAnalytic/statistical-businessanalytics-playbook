"""
Griff 39: Wahrscheinlichkeitsmodelle kalibrieren? (Calibration Plots & Brier Score)

Was ist das und was macht das Skript?
Wenn ein Machine-Learning-Modell (z.B. ein Random Forest) eine "80%ige Wahrscheinlichkeit" 
vorhersagt, bedeutet das oft nicht, dass das Ereignis in der Realität auch in 80 von 100 
Fällen eintritt. Besonders baumbasierte Modelle sind oft unkalibriert: Sie sind zwar gut 
im Sortieren (Ranking), drücken aber Wahrscheinlichkeiten systematisch in Richtung 0.5 
oder extrem an die Ränder.

In der Business-Realität (z.B. Risiko-Scoring, Churn-Prediction) brauchen wir aber ECHTE 
Wahrscheinlichkeiten, um Erwartungswerte (z.B. Kosten = Wahrscheinlichkeit * Warenwert) 
finanziell korrekt zu berechnen. 

Lösung: Wir prüfen die Kalibrierung mit einem Reliability Diagram (Calibration Plot) und 
dem Brier Score (MSE für Wahrscheinlichkeiten, je niedriger desto besser). Wenn das Modell 
lügt, zwingen wir es mit einer "Isotonic Regression", echte Wahrscheinlichkeiten zu lernen.

Business Case in diesem Skript (Neuer Datenfokus!):
Logistik-Risikomanagement: Wir sagen voraus, ob ein Paket verspätet beim Kunden ankommt 
(Late Delivery). Wir nutzen einen neuen Feature-Mix: Die Distanz-Kategorie 
(Cross-State vs. Same-State Lieferung) verknüpft mit dem echten physischen Volumen 
(Länge * Breite * Höhe) und Gewicht des Produkts.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.calibration import calibration_curve, CalibratedClassifierCV
from sklearn.metrics import brier_score_loss

# -------------------------------------------------------------------
# 1. Setup & Pfade
# -------------------------------------------------------------------
ORDERS_PATH = "/Users/stefanbechthold/statistik_business_analytics/data/olist_orders_dataset.csv"
ITEMS_PATH = "/Users/stefanbechthold/statistik_business_analytics/data/olist_order_items_dataset.csv"
CUSTOMERS_PATH = "/Users/stefanbechthold/statistik_business_analytics/data/olist_customers_dataset.csv"
SELLERS_PATH = "/Users/stefanbechthold/statistik_business_analytics/data/olist_sellers_dataset.csv"
PRODUCTS_PATH = "/Users/stefanbechthold/statistik_business_analytics/data/olist_products_dataset.csv"

GRAPHIC_DIR = "/Users/stefanbechthold/statistik_business_analytics/Grafiken/11_Interpretierbarkeit_und_Kalibrierung"
GRAPHIC_PATH = os.path.join(GRAPHIC_DIR, "39_probability_calibration.png")

os.makedirs(GRAPHIC_DIR, exist_ok=True)

def main():
    # -------------------------------------------------------------------
    # 2. Daten laden & Logistik-Features generieren
    # -------------------------------------------------------------------
    df_orders = pd.read_csv(ORDERS_PATH)
    df_items = pd.read_csv(ITEMS_PATH)
    df_customers = pd.read_csv(CUSTOMERS_PATH)
    df_sellers = pd.read_csv(SELLERS_PATH)
    df_products = pd.read_csv(PRODUCTS_PATH)
    
    # Target (Y): Kam das Paket nach dem geschätzten Lieferdatum an? (1 = Ja, 0 = Nein)
    df_orders['delivered_time'] = pd.to_datetime(df_orders['order_delivered_customer_date'], errors='coerce')
    df_orders['estimated_time'] = pd.to_datetime(df_orders['order_estimated_delivery_date'], errors='coerce')
    df_orders['is_late'] = (df_orders['delivered_time'] > df_orders['estimated_time']).astype(int)
    
    df_orders = df_orders.dropna(subset=['delivered_time', 'estimated_time']).copy()

    # Features: Kunde und Verkäufer (Cross-State Logistik)
    order_cust = pd.merge(df_orders[['order_id', 'customer_id', 'is_late']], 
                          df_customers[['customer_id', 'customer_state']], on='customer_id')
    
    # Items mit Verkäufern und Produkten verknüpfen (Volumen in cm³ berechnen)
    df_items = pd.merge(df_items, df_sellers[['seller_id', 'seller_state']], on='seller_id')
    df_products['product_volume_cm3'] = df_products['product_length_cm'] * df_products['product_height_cm'] * df_products['product_width_cm']
    df_items = pd.merge(df_items, df_products[['product_id', 'product_weight_g', 'product_volume_cm3']], on='product_id')
    
    # Aggregation auf Bestell-Ebene (Gewicht und Volumen aufsummieren)
    order_features = df_items.groupby('order_id').agg({
        'seller_state': 'first', # Annahme: Wir betrachten den Hauptverkäufer
        'product_weight_g': 'sum',
        'product_volume_cm3': 'sum'
    }).reset_index()
    
    df_merged = pd.merge(order_cust, order_features, on='order_id', how='inner')
    df_clean = df_merged.dropna().copy()
    
    # Logistik-Feature: Geht das Paket über Bundesstaats-Grenzen hinweg? (1 = Ja, 0 = Nein)
    df_clean['is_cross_state'] = (df_clean['customer_state'] != df_clean['seller_state']).astype(int)
    
    # Sample ziehen für schnelle Ausführung im Script
    df_sample = df_clean.sample(n=20000, random_state=42)
    
    X = df_sample[['is_cross_state', 'product_weight_g', 'product_volume_cm3']]
    y = df_sample['is_late']

    # -------------------------------------------------------------------
    # 3. Modellierung: Unkalibriert vs. Kalibriert
    # -------------------------------------------------------------------
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
    
    # Modell 1: Standard Random Forest (oft schlechte Kalibrierung)
    rf_uncal = RandomForestClassifier(n_estimators=100, max_depth=6, random_state=42)
    rf_uncal.fit(X_train, y_train)
    probs_uncal = rf_uncal.predict_proba(X_test)[:, 1]
    
    # Modell 2: Kalibrierter Random Forest (Isotonic Regression)
    # Nutzt Cross-Validation, um die ausgegebenen Probabilities an die Realität anzupassen
    rf_cal = CalibratedClassifierCV(rf_uncal, method='isotonic', cv=3)
    rf_cal.fit(X_train, y_train)
    probs_cal = rf_cal.predict_proba(X_test)[:, 1]

    # Brier Score berechnen (MSE für Wahrscheinlichkeiten, näher an 0 ist besser)
    brier_uncal = brier_score_loss(y_test, probs_uncal)
    brier_cal = brier_score_loss(y_test, probs_cal)

    # -------------------------------------------------------------------
    # 4. Calibration Curve berechnen
    # -------------------------------------------------------------------
    # Gruppiert Vorhersagen in 10 Bins (z.B. 10%-20%) und vergleicht mit der echten Trefferquote
    frac_true_uncal, prob_pred_uncal = calibration_curve(y_test, probs_uncal, n_bins=10)
    frac_true_cal, prob_pred_cal = calibration_curve(y_test, probs_cal, n_bins=10)

    # -------------------------------------------------------------------
    # 5. Visualisierung im Dark Mode Design
    # -------------------------------------------------------------------
    plt.style.use('dark_background')
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7), facecolor='#121212')
    ax1.set_facecolor('#121212')
    ax2.set_facecolor('#121212')

    # --- Plot 1: Calibration Curve (Reliability Diagram) ---
    # Die perfekte Linie (Diagonale) - Wenn das Modell 30% sagt, trifft es zu 30% ein
    ax1.plot([0, 1], [0, 1], "k:", label="Perfekt kalibriert (Wahrheit)", color="white", linewidth=2)
    
    # Unkalibriertes Modell
    ax1.plot(prob_pred_uncal, frac_true_uncal, "s-", color="#e74c3c", linewidth=2, 
             label=f"Random Forest (Brier: {brier_uncal:.4f})")
    
    # Kalibriertes Modell
    ax1.plot(prob_pred_cal, frac_true_cal, "o-", color="#2ecc71", linewidth=2, 
             label=f"RF + Isotonic (Brier: {brier_cal:.4f})")
    
    ax1.set_title("1. Calibration Curve (Lügt das Modell?)", fontsize=14, fontweight='bold', color='white')
    ax1.set_xlabel("Vom Modell vorhergesagte Wahrscheinlichkeit (Late Delivery)")
    ax1.set_ylabel("Tatsächlicher Anteil (Realität)")
    
    legend1 = ax1.legend(loc='lower right', facecolor='#1e1e1e', edgecolor='#333333')
    plt.setp(legend1.get_texts(), color='white')

    # Info-Box Plot 1
    info_1 = (
        "Die Diagnose:\n"
        "Der unkalibrierte RF (rot) ist zu selbstbewusst. Wenn er\n"
        "z.B. 40% Verspätungs-Risiko meldet, sind in der Realität\n"
        "vielleicht nur 15% der Pakete verspätet (Linie sackt ab).\n"
        "Die Isotonic Regression (grün) zieht die Kurve wieder\n"
        "näher an die perfekte Diagonale."
    )
    props1 = dict(boxstyle='round,pad=1', facecolor='#1e1e1e', alpha=0.9, edgecolor='#e74c3c', linewidth=1.5)
    ax1.text(0.05, 0.95, info_1, transform=ax1.transAxes, fontsize=11,
             verticalalignment='top', bbox=props1, color='#e0e0e0')

    # --- Plot 2: Histogramm der vorhergesagten Wahrscheinlichkeiten ---
    sns.histplot(probs_uncal, bins=20, color='#e74c3c', alpha=0.5, label='Unkalibriert', ax=ax2, edgecolor='none')
    sns.histplot(probs_cal, bins=20, color='#2ecc71', alpha=0.5, label='Kalibriert', ax=ax2, edgecolor='none')
    
    ax2.set_title("2. Verteilung der Wahrscheinlichkeiten", fontsize=14, fontweight='bold', color='white')
    ax2.set_xlabel("Wahrscheinlichkeit für 'Verspätete Lieferung'")
    ax2.set_ylabel("Anzahl der Bestellungen im Test-Set")
    
    legend2 = ax2.legend(loc='upper right', facecolor='#1e1e1e', edgecolor='#333333')
    plt.setp(legend2.get_texts(), color='white')

    # Info-Box Plot 2
    info_2 = (
        "Business Impact (Risk Mitigation):\n"
        "------------------------------------\n"
        "Nur mit exakt kalibrierten Modellen können wir\n"
        "Schwellenwerte für Automatisierungen setzen.\n"
        "Bsp: 'Wenn P(Verspätung) > 80%, sende proaktiv\n"
        "einen 5€ Gutschein.' Wenn die Wahrscheinlichkeit\n"
        "falsch ist, verbrennen wir massiv Geld!"
    )
    props2 = dict(boxstyle='round,pad=1', facecolor='#1e1e1e', alpha=0.9, edgecolor='#2ecc71', linewidth=1.5)
    ax2.text(0.4, 0.6, info_2, transform=ax2.transAxes, fontsize=11,
             verticalalignment='center', bbox=props2, color='#e0e0e0')

    plt.suptitle("Model Calibration: Von Ranking-Scores zu echten Wahrscheinlichkeiten", fontsize=18, fontweight='bold', color='white', y=1.05)
    plt.tight_layout()
    
    # Speichern der Grafik
    plt.savefig(GRAPHIC_PATH, dpi=300, bbox_inches='tight', facecolor=fig.get_facecolor())
    plt.close()

if __name__ == "__main__":
    main()
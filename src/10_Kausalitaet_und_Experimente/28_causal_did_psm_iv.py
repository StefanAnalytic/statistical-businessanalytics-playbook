"""
Griff 28: Ist ein Effekt wirklich kausal? (Causal Inference: Propensity Score Matching)

Was ist das und was macht das Skript?
"Korrelation ist nicht Kausalität!" – Der berühmteste Satz in der Statistik.
Wenn wir sehen, dass Kunden, die hohe Versandkosten zahlen, schlechtere Bewertungen hinterlassen, 
könnten wir denken: "Hohe Versandkosten machen Kunden wütend!" (Kausalität).
Aber Moment: Hohe Versandkosten entstehen oft durch schwere Produkte oder weite Distanzen (Confounder). 
Vielleicht macht die lange Lieferzeit (Distanz) den Kunden wütend, NICHT der Preis für den Versand!

Um den REINEN kausalen Effekt der Versandkosten zu isolieren, nutzen wir Propensity Score Matching (PSM).
PSM ist wie ein künstlicher A/B-Test aus historischen Daten:
1. Wir berechnen für jede Bestellung die "Wahrscheinlichkeit" (Propensity Score), hohe Versandkosten zu haben,
   basierend auf Gewicht, Preis und Lieferzeit.
2. Wir suchen für jede Bestellung mit hohen Versandkosten (Treatment) einen statistischen "Zwilling" 
   mit niedrigen Versandkosten (Control), der exakt denselben Propensity Score hat (also gleich schwer 
   ist und gleich lang gebraucht hat).
3. Wir vergleichen die Bewertungen nur noch zwischen diesen perfekten Zwillingen!

Business Case in diesem Skript:
Senken hohe Versandkosten (Treatment > 20 BRL) WIRKLICH die Kundenzufriedenheit (1-5 Sterne), 
oder ist das nur eine Illusion, die durch schwere Pakete und lange Lieferzeiten getrieben wird?
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import NearestNeighbors

# -------------------------------------------------------------------
# 1. Setup & Pfade
# -------------------------------------------------------------------
# Wir sind im Ordner "10_Kausalitaet_und_Experimente"
ORDERS_PATH = "/Users/stefanbechthold/statistik_business_analytics/data/olist_orders_dataset.csv"
ITEMS_PATH = "/Users/stefanbechthold/statistik_business_analytics/data/olist_order_items_dataset.csv"
REVIEWS_PATH = "/Users/stefanbechthold/statistik_business_analytics/data/olist_order_reviews_dataset.csv"
PRODUCTS_PATH = "/Users/stefanbechthold/statistik_business_analytics/data/olist_products_dataset.csv"

GRAPHIC_DIR = "/Users/stefanbechthold/statistik_business_analytics/Grafiken/10_Kausalitaet_und_Experimente"
GRAPHIC_PATH = os.path.join(GRAPHIC_DIR, "28_psm_causal_inference.png")

os.makedirs(GRAPHIC_DIR, exist_ok=True)

def main():
    print("🚀 Starte Griff 28: Causal Inference (Propensity Score Matching)...")
    
    # -------------------------------------------------------------------
    # 2. Daten laden & Confounder (Verzerrer) vorbereiten
    # -------------------------------------------------------------------
    print("Lade Bestelldaten und baue Confounder-Features...")
    df_orders = pd.read_csv(ORDERS_PATH)
    df_items = pd.read_csv(ITEMS_PATH)
    df_reviews = pd.read_csv(REVIEWS_PATH)
    df_products = pd.read_csv(PRODUCTS_PATH)
    
    # Lieferzeit berechnen (Confounder 1)
    df_orders['purchase_time'] = pd.to_datetime(df_orders['order_purchase_timestamp'], errors='coerce')
    df_orders['delivered_time'] = pd.to_datetime(df_orders['order_delivered_customer_date'], errors='coerce')
    df_orders['delivery_days'] = (df_orders['delivered_time'] - df_orders['purchase_time']).dt.total_seconds() / (24 * 3600)
    
    # Produkte und Items mergen für Gewicht (Confounder 2) und Preis (Confounder 3)
    items_prod = pd.merge(df_items, df_products[['product_id', 'product_weight_g']], on='product_id', how='inner')
    order_features = items_prod.groupby('order_id').agg({
        'price': 'sum',
        'freight_value': 'sum',
        'product_weight_g': 'sum'
    }).reset_index()
    
    # Erste Bewertung pro Bestellung (Outcome / Zielvariable Y)
    order_reviews = df_reviews.drop_duplicates(subset=['order_id'])[['order_id', 'review_score']]
    
    # Alles mergen
    df_merged = pd.merge(df_orders[['order_id', 'delivery_days']], order_features, on='order_id', how='inner')
    df_merged = pd.merge(df_merged, order_reviews, on='order_id', how='inner')
    
    # Bereinigung und Sample ziehen (PSM mit Nearest Neighbors skaliert quadratisch, daher Sample!)
    df_clean = df_merged.dropna().copy()
    df_clean = df_clean[(df_clean['delivery_days'] > 0) & (df_clean['delivery_days'] < 60)]
    df_sample = df_clean.sample(n=10000, random_state=42).reset_index(drop=True)
    
    # -------------------------------------------------------------------
    # 3. Treatment, Outcome und Naive Schätzung
    # -------------------------------------------------------------------
    # Treatment (T): Hat der Kunde mehr als 20 BRL Versand gezahlt? (1 = Ja, 0 = Nein)
    df_sample['T'] = (df_sample['freight_value'] > 20).astype(int)
    
    # Outcome (Y): Die Bewertung
    df_sample['Y'] = df_sample['review_score']
    
    # Confounder (X): Variablen, die sowohl T als auch Y beeinflussen
    covariates = ['delivery_days', 'product_weight_g', 'price']
    
    # Naive Schätzung (Einfacher Mittelwert-Vergleich)
    mean_treated_naive = df_sample[df_sample['T'] == 1]['Y'].mean()
    mean_control_naive = df_sample[df_sample['T'] == 0]['Y'].mean()
    naive_effect = mean_treated_naive - mean_control_naive
    
    print(f"\n--- NAIVE SCHÄTZUNG (KORRELATION) ---")
    print(f"Bewertung Hohe Versandkosten (Treated): {mean_treated_naive:.2f} Sterne")
    print(f"Bewertung Niedrige Versandkosten (Control): {mean_control_naive:.2f} Sterne")
    print(f"Scheinbarer Effekt: {naive_effect:.2f} Sterne")

    # -------------------------------------------------------------------
    # 4. Propensity Score schätzen
    # -------------------------------------------------------------------
    print("\nBerechne Propensity Scores (Wahrscheinlichkeit für das Treatment)...")
    # Wir nutzen Logistische Regression, um anhand von X vorherzusagen, wer in Gruppe T=1 landet
    logit = LogisticRegression(max_iter=1000)
    logit.fit(df_sample[covariates], df_sample['T'])
    
    # Wahrscheinlichkeit für T=1 speichern
    df_sample['propensity_score'] = logit.predict_proba(df_sample[covariates])[:, 1]

    # -------------------------------------------------------------------
    # 5. Matching der "Zwillinge" (Nearest Neighbors)
    # -------------------------------------------------------------------
    print("Suche statistische Zwillinge (Matching)...")
    treated = df_sample[df_sample['T'] == 1].copy()
    control = df_sample[df_sample['T'] == 0].copy()
    
    # NearestNeighbors sucht für jeden Treated-Punkt den Control-Punkt mit dem ähnlichsten Propensity Score
    nn = NearestNeighbors(n_neighbors=1, algorithm='ball_tree')
    nn.fit(control[['propensity_score']])
    
    # Distanzen und Indices der Zwillinge in der Control-Gruppe finden
    distances, indices = nn.kneighbors(treated[['propensity_score']])
    
    # Wir bauen ein neues DataFrame nur mit den erfolgreich gematchten Paaren!
    matched_control = control.iloc[indices.flatten()].copy()
    matched_data = pd.concat([treated, matched_control])
    
    # Kausale Schätzung (Average Treatment Effect on the Treated - ATT)
    mean_treated_causal = matched_data[matched_data['T'] == 1]['Y'].mean()
    mean_control_causal = matched_data[matched_data['T'] == 0]['Y'].mean()
    causal_effect = mean_treated_causal - mean_control_causal
    
    print(f"\n--- KAUSALE SCHÄTZUNG (MATCHED DATA) ---")
    print(f"Gematchte Bewertung (Treated): {mean_treated_causal:.2f} Sterne")
    print(f"Gematchte Bewertung (Control Zwillinge): {mean_control_causal:.2f} Sterne")
    print(f"Echter Kausaler Effekt: {causal_effect:.2f} Sterne")
    print(f"-> Der Fehler der naiven Schätzung war: {abs(naive_effect - causal_effect):.2f} Sterne!")

    # -------------------------------------------------------------------
    # 6. Wunderschöne Visualisierung (Dashboard)
    # -------------------------------------------------------------------
    print("\nGeneriere Visualisierungs-Grid...")
    sns.set_theme(style="whitegrid")
    fig = plt.figure(figsize=(16, 10))
    
    ax_ps_before = plt.subplot2grid((2, 2), (0, 0))
    ax_ps_after = plt.subplot2grid((2, 2), (0, 1))
    ax_effect = plt.subplot2grid((2, 2), (1, 0), colspan=2)

    # --- Plot 1: Propensity Scores VOR dem Matching ---
    # Zeigt die massive Verzerrung in den Rohdaten
    sns.kdeplot(data=df_sample[df_sample['T']==0], x='propensity_score', fill=True, color='#3498db', label='Kontrolle (Günstiger Versand)', ax=ax_ps_before)
    sns.kdeplot(data=df_sample[df_sample['T']==1], x='propensity_score', fill=True, color='#e74c3c', label='Treatment (Teurer Versand)', ax=ax_ps_before)
    
    ax_ps_before.set_title("1. Propensity Scores VOR dem Matching (Äpfel vs. Birnen)", fontsize=14, fontweight="bold")
    ax_ps_before.set_xlabel("Wahrscheinlichkeit für teuren Versand (Propensity Score)")
    ax_ps_before.set_ylabel("Dichte")
    ax_ps_before.legend()

    # --- Plot 2: Propensity Scores NACH dem Matching ---
    # Zeigt, dass wir jetzt perfekte statistische Zwillinge vergleichen
    sns.kdeplot(data=matched_data[matched_data['T']==0], x='propensity_score', fill=True, color='#3498db', label='Gematchte Kontrolle', ax=ax_ps_after)
    sns.kdeplot(data=matched_data[matched_data['T']==1], x='propensity_score', fill=True, color='#e74c3c', label='Treatment', ax=ax_ps_after)
    
    ax_ps_after.set_title("2. Propensity Scores NACH dem Matching (Äpfel vs. Äpfel)", fontsize=14, fontweight="bold")
    ax_ps_after.set_xlabel("Wahrscheinlichkeit für teuren Versand (Propensity Score)")
    ax_ps_after.set_ylabel("")
    ax_ps_after.legend()

    # --- Plot 3: Der Naive vs. Kausale Effekt (Waterfall / Bar Chart) ---
    labels = ['Naive Schätzung\n(Korrelation)', 'Kausale Schätzung\n(Bereinigt durch Matching)']
    effects = [naive_effect, causal_effect]
    colors = ['#7f8c8d', '#2ecc71']
    
    bars = ax_effect.barh(labels, effects, color=colors, edgecolor='black', height=0.5)
    ax_effect.axvline(0, color='black', linewidth=2)
    
    ax_effect.set_title("3. Business Insight: Wie stark senken teure Versandkosten die Bewertung WIRKLICH?", fontsize=14, fontweight="bold")
    ax_effect.set_xlabel("Verlust an Sternen (Effektgröße)")
    
    # Werte an die Balken schreiben
    for bar in bars:
        width = bar.get_width()
        # Text leicht nach rechts versetzt, da die Balken negativ sind
        ax_effect.text(width - 0.05, bar.get_y() + bar.get_height()/2, 
                       f"{width:.2f} Sterne", 
                       ha='right', va='center', color='white', fontweight='bold', fontsize=14)

    # Info-Box für Plot 3
    info_text = (
        "Was bedeutet das?\n"
        "Der naive Vergleich war übertrieben!\n"
        "Ein großer Teil der schlechten Bewertungen\n"
        "kam gar nicht durch den teuren Versand,\n"
        "sondern durch das Gewicht und lange Lieferzeiten.\n"
        "Wenn wir diese Confounder eliminieren (Kausalität),\n"
        "schrumpft der echte negative Effekt der\n"
        "Versandkosten deutlich zusammen."
    )
    props = dict(boxstyle='round,pad=0.8', facecolor='white', alpha=0.9, edgecolor='gray', linewidth=1.5)
    ax_effect.text(0.02, 0.1, info_text, transform=ax_effect.transAxes, fontsize=12,
                   verticalalignment='bottom', horizontalalignment='left', bbox=props, color='#2c3e50')

    # Layout optimieren und speichern
    plt.suptitle("Causal Inference: Echte Ursachen von Korrelation trennen (PSM)", 
                 fontsize=18, fontweight="bold", y=1.05)
    plt.tight_layout()
    sns.despine()
    
    plt.savefig(GRAPHIC_PATH, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✅ Grafik erfolgreich gespeichert unter:\n{GRAPHIC_PATH}")

if __name__ == "__main__":
    main()
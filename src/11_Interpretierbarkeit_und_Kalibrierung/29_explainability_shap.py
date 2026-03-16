"""
Griff 29: Wie erkläre ich mein Modell? (Explainability: SHAP, Permutation, PDP)

Was ist das und was macht das Skript?
Wir arbeiten im Bereich "Credit Scoring" (Kreditwürdigkeit / Ratenzahlungs-Wahrscheinlichkeit).
Ein komplexes Random Forest Modell (Blackbox) entscheidet, ob ein Kunde Ratenzahlung 
in Anspruch nimmt. In der Finanzwelt (und durch die DSGVO) MÜSSEN wir diese 
Entscheidungen erklären können (Transparenz & Debugging).

Die 3 Säulen der Erklärbarkeit in diesem Skript:
1. Permutation Importance: Was passiert mit der Modell-Genauigkeit, wenn wir eine 
   Spalte zufällig mischen (shuffeln)? Fällt die Genauigkeit, war das Feature wichtig.
2. PDP (Partial Dependence Plot): Wie genau verändert ein einzelnes Feature (z.B. der Preis) 
   die Vorhersage? Steigt die Wahrscheinlichkeit für Raten linear oder exponentiell?
3. SHAP (SHapley Additive exPlanations): Der Goldstandard. Spieltheoretische Verteilung des 
   "Erfolgs" auf die Features. 

Business Case & Wichtige Warnung:
-> SHAP ist extrem rechenintensiv ("teuer"). Wir nutzen maximale Rechenleistung (n_jobs=-1).
-> Transparenz ist nicht Kausalität! SHAP zeigt uns, WIE das Modell entscheidet, nicht 
   zwingend, wie die Welt kausal funktioniert.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.inspection import permutation_importance, PartialDependenceDisplay

try:
    import shap
except ImportError:
    print("FEHLER: Die Bibliothek 'shap' fehlt. Bitte ausführen: pip install shap")
    exit()

# -------------------------------------------------------------------
# 1. Setup & Pfade
# -------------------------------------------------------------------
PAYMENTS_PATH = "/Users/stefanbechthold/statistik_business_analytics/data/olist_order_payments_dataset.csv"
ITEMS_PATH = "/Users/stefanbechthold/statistik_business_analytics/data/olist_order_items_dataset.csv"
ORDERS_PATH = "/Users/stefanbechthold/statistik_business_analytics/data/olist_orders_dataset.csv"

GRAPHIC_DIR = "/Users/stefanbechthold/statistik_business_analytics/Grafiken/13_Explainability"
GRAPHIC_PATH = os.path.join(GRAPHIC_DIR, "29_explainability_shap_pdp_darkmode.png")

os.makedirs(GRAPHIC_DIR, exist_ok=True)

def main():
    print("🚀 Starte Griff 29: Model Explainability (SHAP, Permutation, PDP)...")
    
    # -------------------------------------------------------------------
    # 2. Daten laden & Imbalanced Dataset für Credit Scoring aufbauen
    # -------------------------------------------------------------------
    print("Lade Daten und berechne Features...")
    df_payments = pd.read_csv(PAYMENTS_PATH)
    df_items = pd.read_csv(ITEMS_PATH)
    df_orders = pd.read_csv(ORDERS_PATH)
    
    # Target: Ratenzahlung genutzt? (1 = Ja, 0 = Nein)
    cc_payments = df_payments[df_payments['payment_type'] == 'credit_card'].copy()
    order_pay = cc_payments.groupby('order_id').agg({
        'payment_value': 'sum',
        'payment_installments': 'max'
    }).reset_index()
    order_pay['is_installment'] = (order_pay['payment_installments'] > 1).astype(int)
    
    # Feature: Lieferzeit (Tage)
    df_orders['purchase_time'] = pd.to_datetime(df_orders['order_purchase_timestamp'], errors='coerce')
    df_orders['estimated_time'] = pd.to_datetime(df_orders['order_estimated_delivery_date'], errors='coerce')
    df_orders['delivery_promise_days'] = (df_orders['estimated_time'] - df_orders['purchase_time']).dt.total_seconds() / (24 * 3600)
    
    # Feature: Bestellkosten (Preis & Versand)
    order_costs = df_items.groupby('order_id').agg({'price': 'sum', 'freight_value': 'sum'}).reset_index()
    
    # Mergen
    df_merged = pd.merge(order_pay, order_costs, on='order_id', how='inner')
    df_merged = pd.merge(df_merged, df_orders[['order_id', 'delivery_promise_days']], on='order_id', how='inner')
    
    df_clean = df_merged.dropna().copy()
    
    # Für SHAP ziehen wir ein Sample von 10.000 (SHAP ist extrem rechenintensiv)
    df_sample = df_clean.sample(n=10000, random_state=42)
    
    X = df_sample[['payment_value', 'freight_value', 'price', 'delivery_promise_days']]
    y = df_sample['is_installment']
    
    print(f"Datensatz bereit: {len(X)} Bestellungen für die Modell-Erklärung.")

    # -------------------------------------------------------------------
    # 3. Modell trainieren (Die Blackbox)
    # -------------------------------------------------------------------
    print("\nTrainiere Random Forest (Max Compute n_jobs=-1)...")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    rf = RandomForestClassifier(n_estimators=300, max_depth=8, n_jobs=-1, random_state=42)
    rf.fit(X_train, y_train)

    # -------------------------------------------------------------------
    # 4. Explainability-Methoden anwenden
    # -------------------------------------------------------------------
    print("\nBerechne Explainability-Metriken...")
    
    # A) Permutation Importance
    print("-> 1/3: Permutation Importance...")
    perm_importance = permutation_importance(rf, X_test, y_test, n_repeats=10, random_state=42, n_jobs=-1)
    
    # B) SHAP (TreeExplainer für Bäume)
    print("-> 2/3: SHAP Values (Das dauert kurz)...")
    explainer = shap.TreeExplainer(rf)
    shap_values = explainer.shap_values(X_test)
    
    # SHAP Versions-Check für Random Forest: 
    # Ältere Versionen = Liste, neuere Versionen = 3D-Array (Samples, Features, Classes)
    if isinstance(shap_values, list):
        shap_target = shap_values[1]  # Klasse 1 extrahieren (alt)
    elif len(np.shape(shap_values)) == 3:
        shap_target = shap_values[:, :, 1]  # Klasse 1 extrahieren (neu)
    else:
        shap_target = shap_values
        
    # -------------------------------------------------------------------
    # 5. Wunderschöne Visualisierung (3-teiliges Dashboard im Darkmode)
    # -------------------------------------------------------------------
    print("\nGeneriere Visualisierungs-Grid im Darkmode...")
    
    # Darkmode Styling
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
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    # Features und Importances vorbereiten - JETZT ALS NUMPY ARRAY
    features = np.array(X.columns)
    
    # --- Plot 1: Permutation Importance ---
    perm_sorted_idx = perm_importance.importances_mean.argsort()
    axes[0].barh(features[perm_sorted_idx], 
                 perm_importance.importances_mean[perm_sorted_idx], 
                 color="#3498db", edgecolor="white")
    
    axes[0].set_title("1. Permutation Importance", fontsize=14, fontweight="bold", color="white")
    axes[0].set_xlabel("Abfall der Accuracy beim Shuffeln")
    axes[0].set_ylabel("Features")

    # --- Plot 2: Partial Dependence Plot (PDP) für 'payment_value' ---
    # Zeigt den isolierten marginalen Effekt des wichtigsten Features
    print("-> 3/3: Partial Dependence Plot...")
    PartialDependenceDisplay.from_estimator(
        estimator=rf, 
        X=X_test, 
        features=['payment_value'], 
        ax=axes[1],
        line_kw={"color": "#e74c3c", "linewidth": 3}
    )
    axes[1].set_title("2. PDP: Effekt des Bestellwerts", fontsize=14, fontweight="bold", color="white")
    axes[1].set_ylabel("Wahrscheinlichkeit (Ratenzahlung)")
    
    # Info-Box im PDP
    axes[1].text(0.5, 0.1, "PDP zeigt, ab welchem Betrag\ndie Raten-Chance explodiert.", 
                 transform=axes[1].transAxes, fontsize=11, ha='center',
                 bbox=dict(facecolor='#1e1e1e', alpha=0.9, edgecolor='#e74c3c', linewidth=1.5))

    # --- Plot 3: SHAP Feature Importance ---
    # Durchschnittliche absolute SHAP-Werte pro Feature
    shap_mean_abs = np.abs(shap_target).mean(axis=0)
    shap_sorted_idx = shap_mean_abs.argsort()
    
    axes[2].barh(features[shap_sorted_idx], 
                 shap_mean_abs[shap_sorted_idx], 
                 color="#9b59b6", edgecolor="white")
    
    axes[2].set_title("3. SHAP Importance (Goldstandard)", fontsize=14, fontweight="bold", color="white")
    axes[2].set_xlabel("Mittlerer Einfluss auf die Modell-Ausgabe")

    # Layout optimieren und speichern
    plt.suptitle("Transparenz & Debugging im Credit Scoring (Blackbox Explainability)", 
                 fontsize=18, fontweight="bold", color="white", y=1.05)
    plt.tight_layout()
    
    # Hintergrund beim Speichern erzwingen
    plt.savefig(GRAPHIC_PATH, dpi=300, bbox_inches='tight', facecolor=fig.get_facecolor())
    plt.close()
    
    print(f"✅ Grafik erfolgreich gespeichert unter:\n{GRAPHIC_PATH}")

if __name__ == "__main__":
    main()
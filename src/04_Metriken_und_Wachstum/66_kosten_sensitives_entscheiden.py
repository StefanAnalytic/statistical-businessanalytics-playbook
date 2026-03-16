"""
Griff 66: Kosten-sensitives Entscheiden (Profit Calibration & Decision Curves)

Was ist das und was macht das Skript?
Data Scientists optimieren Modelle oft auf Metriken wie "Accuracy" (Genauigkeit) oder "AUC". 
Das Business denkt aber in Euro! Ein False Positive (Fehlalarm) und ein False Negative 
(Übersehener Fall) kosten in der Realität fast nie gleich viel.

Beispiel proaktiver Kundenservice (Late Delivery):
- False Positive: Wir denken, das Paket kommt zu spät und geben dem Kunden vorsorglich 
  einen 5€ Gutschein. Das Paket kommt pünktlich. Wir haben 5€ verschwendet.
- False Negative: Wir denken, das Paket kommt pünktlich. Es kommt zu spät. Der Kunde 
  ist wütend und kauft nie wieder bei uns. Wir verlieren 50€ (Customer Lifetime Value).

Wenn ein Fehler 10x teurer ist als der andere, dürfen wir unser Modell nicht bei 
der Standard-Wahrscheinlichkeit von 50% (0.5) abschneiden lassen. Wir müssen den 
Threshold (Schwellenwert) so kalibrieren, dass die Gesamtkosten minimiert werden.

Business Case in diesem Skript:
Wir trainieren ein Modell, das verspätete Lieferungen vorhersagt. Anstatt auf Accuracy 
zu schauen, weisen wir jedem Fehler im Business-Kontext harte Euro-Werte zu. 
Wir berechnen eine "Decision Curve" (Entscheidungskurve), die uns genau zeigt, bei 
welcher Wahrscheinlichkeit wir den 5€ Gutschein rausschicken müssen, um den 
maximalen Profit (bzw. die minimalen Kosten) für Olist herauszuholen.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix

# -------------------------------------------------------------------
# 1. Setup & Pfade
# -------------------------------------------------------------------
ORDERS_PATH = "/Users/stefanbechthold/statistik_business_analytics/data/olist_orders_dataset.csv"
ITEMS_PATH = "/Users/stefanbechthold/statistik_business_analytics/data/olist_order_items_dataset.csv"
PRODUCTS_PATH = "/Users/stefanbechthold/statistik_business_analytics/data/olist_products_dataset.csv"

GRAPHIC_DIR = "/Users/stefanbechthold/statistik_business_analytics/Grafiken/06_Klassifikation_und_Validierung"
GRAPHIC_PATH = os.path.join(GRAPHIC_DIR, "60_cost_sensitive_decision_curve.png")

os.makedirs(GRAPHIC_DIR, exist_ok=True)

def main():
    print("🚀 Starte Griff 60: Kosten-sensitives Entscheiden & Profit Calibration...")
    
    # -------------------------------------------------------------------
    # 2. Daten laden & Feature Engineering (Schneller Durchlauf)
    # -------------------------------------------------------------------
    df_orders = pd.read_csv(ORDERS_PATH)
    df_items = pd.read_csv(ITEMS_PATH)
    df_products = pd.read_csv(PRODUCTS_PATH)
    
    # Target: Verspätete Lieferung? (1 = Ja, 0 = Nein)
    df_orders['delivered'] = pd.to_datetime(df_orders['order_delivered_customer_date'], errors='coerce')
    df_orders['estimated'] = pd.to_datetime(df_orders['order_estimated_delivery_date'], errors='coerce')
    df_orders['is_late'] = (df_orders['delivered'] > df_orders['estimated']).astype(int)
    
    # Features mergen
    df_items = pd.merge(df_items, df_products[['product_id', 'product_weight_g']], on='product_id', how='left')
    order_features = df_items.groupby('order_id').agg({
        'price': 'sum',
        'freight_value': 'sum',
        'product_weight_g': 'sum'
    }).reset_index()
    
    df_merged = pd.merge(df_orders[['order_id', 'is_late']], order_features, on='order_id', how='inner')
    df_clean = df_merged.dropna().copy()
    
    # Sample ziehen
    df_sample = df_clean.sample(n=15000, random_state=42).reset_index(drop=True)
    
    X = df_sample[['price', 'freight_value', 'product_weight_g']]
    y = df_sample['is_late']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

    # -------------------------------------------------------------------
    # 3. Modell trainieren & Wahrscheinlichkeiten vorhersagen
    # -------------------------------------------------------------------
    model = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42)
    model.fit(X_train, y_train)
    
    # Wir brauchen die rohen Wahrscheinlichkeiten (Predict Proba), nicht die harten 0/1 Labels!
    y_probs = model.predict_proba(X_test)[:, 1]

    # -------------------------------------------------------------------
    # 4. Die Business-Kostenmatrix definieren
    # -------------------------------------------------------------------
    # Was kostet uns welche Entscheidung in Euro (BRL)?
    cost_TP = 5.0   # Wir geben 5€ Gutschein. Kunde ist besänftigt. (Kosten = 5€)
    cost_FP = 5.0   # Wir geben 5€ Gutschein umsonst (Paket war pünktlich). (Kosten = 5€)
    cost_TN = 0.0   # Wir tun nichts, Paket ist pünktlich. Alles super. (Kosten = 0€)
    cost_FN = 50.0  # Wir tun nichts, Paket ist zu spät. Kunde kündigt! (Kosten = 50€ CLV Verlust)
    
    # Wir iterieren über 100 mögliche Schwellenwerte (Thresholds von 0.01 bis 0.99)
    thresholds = np.linspace(0.01, 0.99, 100)
    total_costs = []
    
    for t in thresholds:
        # Harte Vorhersage basierend auf aktuellem Threshold
        y_pred_t = (y_probs >= t).astype(int)
        
        # Wahrheitsmatrix (Confusion Matrix) berechnen
        tn, fp, fn, tp = confusion_matrix(y_test, y_pred_t).ravel()
        
        # Totale Business-Kosten für dieses Setup berechnen
        cost = (tp * cost_TP) + (fp * cost_FP) + (fn * cost_FN) + (tn * cost_TN)
        
        # Durchschnittliche Kosten pro Bestellung
        avg_cost = cost / len(y_test)
        total_costs.append(avg_cost)

    # Den absolut besten Threshold finden (der Punkt mit den geringsten Kosten)
    best_idx = np.argmin(total_costs)
    optimal_threshold = thresholds[best_idx]
    min_cost = total_costs[best_idx]
    
    # Kosten beim Standard-Threshold (0.5) zum Vergleich
    default_idx = np.abs(thresholds - 0.5).argmin()
    default_cost = total_costs[default_idx]

    # -------------------------------------------------------------------
    # 5. Visualisierung im Dark Mode Design
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

    # --- Plot 1: Die Kostenkurve (Decision Curve) ---
    ax1.plot(thresholds, total_costs, color='#3498db', linewidth=3, label='Erwartete Kosten pro Bestellung')
    
    # Optimalen Punkt markieren
    ax1.scatter(optimal_threshold, min_cost, color='#2ecc71', s=150, zorder=5)
    ax1.axvline(optimal_threshold, color='#2ecc71', linestyle='--', linewidth=2, 
                label=f'Optimaler Threshold ({optimal_threshold:.2f}) -> {min_cost:.2f} BRL')
    
    # Naiven Standard-Punkt (0.5) markieren
    ax1.scatter(0.5, default_cost, color='#e74c3c', s=150, zorder=5)
    ax1.axvline(0.5, color='#e74c3c', linestyle=':', linewidth=2, 
                label=f'Naiver Default (0.50) -> {default_cost:.2f} BRL')

    ax1.set_title("1. Cost Curve (Wann müssen wir handeln?)", fontsize=14, fontweight='bold', color='white')
    ax1.set_xlabel("Wahrscheinlichkeits-Threshold für Intervention (Gutschein)")
    ax1.set_ylabel("Durchschnittliche Fehlerkosten pro Bestellung (BRL)")
    
    legend1 = ax1.legend(loc='upper right', facecolor='#1e1e1e', edgecolor='#333333')
    plt.setp(legend1.get_texts(), color='white')

    # Info-Box Plot 1
    info_1 = (
        "Das 0.5-Dogma zerstören:\n"
        "------------------------\n"
        "Die meisten Modelle reagieren erst, wenn sie sich zu >50% sicher sind.\n"
        "Aber ein unentdeckter Fehler (50€ Verlust) tut 10x mehr weh als ein\n"
        "falscher Alarm (5€ Gutschein). Die grüne Linie zeigt: Wir müssen den\n"
        "Gutschein schon verschicken, sobald das Modell auch nur eine\n"
        f"Verspätungswahrscheinlichkeit von {optimal_threshold*100:.0f}% wittert!"
    )
    props1 = dict(boxstyle='round,pad=1', facecolor='#1e1e1e', alpha=0.9, edgecolor='#2ecc71', linewidth=1.5)
    ax1.text(0.05, 0.45, info_1, transform=ax1.transAxes, fontsize=11,
             verticalalignment='center', bbox=props1, color='white')

    # --- Plot 2: Kostenvergleich (Bar Chart Business Impact) ---
    # Hochrechnung auf 100.000 Bestellungen
    scale_factor = 100000
    total_cost_optimal = min_cost * scale_factor
    total_cost_default = default_cost * scale_factor
    savings = total_cost_default - total_cost_optimal
    
    bars = ax2.bar(['Naiver Ansatz (Threshold 0.50)', f'Kosten-optimiert (Threshold {optimal_threshold:.2f})'], 
                   [total_cost_default, total_cost_optimal], 
                   color=['#e74c3c', '#2ecc71'], edgecolor='none')
    
    ax2.set_title("2. Business Impact: Kosten bei 100.000 Bestellungen", fontsize=14, fontweight='bold', color='white')
    ax2.set_ylabel("Gesamtkosten für Logistik-Fehler (BRL)")

    # Werte und Einsparungen über die Balken schreiben
    ax2.text(0, total_cost_default + 20000, f"{total_cost_default:,.0f} BRL", ha='center', color='white', fontweight='bold', fontsize=12)
    ax2.text(1, total_cost_optimal + 20000, f"{total_cost_optimal:,.0f} BRL", ha='center', color='white', fontweight='bold', fontsize=12)

    # Info-Box Plot 2
    info_2 = (
        "Der direkte Business Value (ROI):\n"
        "---------------------------------\n"
        "Wenn wir das Machine Learning Modell nicht auf 'Accuracy',\n"
        "sondern auf harte Business-Metriken (Profit Calibration) kalibrieren,\n"
        "sparen wir bei 100.000 Bestellungen massiv Geld ein.\n\n"
        f"Netto-Ersparnis durch Anpassung EINER einzigen Zeile Code:\n"
        f"--> {savings:,.0f} BRL gespart! <--"
    )
    props2 = dict(boxstyle='round,pad=1', facecolor='#1e1e1e', alpha=0.9, edgecolor='#f1c40f', linewidth=1.5)
    ax2.text(0.5, 0.5, info_2, transform=ax2.transAxes, fontsize=11,
             verticalalignment='center', ha='center', bbox=props2, color='white')

    plt.suptitle("Profit Calibration: Machine Learning an harte Business-KPIs koppeln", 
                 fontsize=18, fontweight='bold', color='white', y=1.05)
    plt.tight_layout()
    
    # Speichern der Grafik
    plt.savefig(GRAPHIC_PATH, dpi=300, bbox_inches='tight', facecolor=fig.get_facecolor())
    plt.close()
    
    print(f"✅ Grafik erfolgreich gespeichert unter:\n{GRAPHIC_PATH}")

if __name__ == "__main__":
    main()
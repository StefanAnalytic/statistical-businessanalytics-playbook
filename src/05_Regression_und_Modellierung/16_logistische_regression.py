"""
Griff 16: Faktoren beeinflussen ein Ja/Nein-Ereignis? (Logistische Regression) - VARIATION: RATENZAHLUNG

Was ist das und was macht das Skript?
Wir nutzen die Logistische Regression für eine neue, hochrelevante binäre Business-Entscheidung (1/0):
Entscheidet sich ein Kreditkarten-Kunde für eine Ratenzahlung (>1 Rate) oder zahlt er alles auf einmal?

Business Case in diesem Skript:
Ratenzahlungen (Installments) sind in Brasilien extrem wichtig für die Conversion Rate, 
kosten den Händler aber Gebühren und Liquidität. Wir wollen verstehen, welche Faktoren 
Kunden in die Ratenzahlung treiben. 
Als Einflussfaktoren (Features) testen wir diesmal: 
- Gesamtbestellwert ('payment_value')
- Versandkosten ('freight_value')
- Gewicht der Lieferung in KG ('product_weight_kg')

Der Super-Power der Logistischen Regression bleiben die "Odds Ratios" (Quotenverhältnisse). 
Wir lesen ab: "Wenn der Preis um 100 BRL steigt, wie sehr erhöht sich die Chance auf Ratenzahlung?"
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm

# -------------------------------------------------------------------
# 1. Setup & Pfade (Neue Datensatz-Kombination!)
# -------------------------------------------------------------------
PAYMENTS_PATH = "/Users/stefanbechthold/statistik_business_analytics/data/olist_order_payments_dataset.csv"
ITEMS_PATH = "/Users/stefanbechthold/statistik_business_analytics/data/olist_order_items_dataset.csv"
PRODUCTS_PATH = "/Users/stefanbechthold/statistik_business_analytics/data/olist_products_dataset.csv"

GRAPHIC_DIR = "/Users/stefanbechthold/statistik_business_analytics/Grafiken/06_Klassifikation_und_Validierung"
GRAPHIC_PATH = os.path.join(GRAPHIC_DIR, "16_logistische_regression_raten.png")

os.makedirs(GRAPHIC_DIR, exist_ok=True)

def main():
    print("🚀 Starte Griff 16 (Variation): Logistische Regression (Ratenzahlung)...")
    
    # -------------------------------------------------------------------
    # 2. Daten laden, verknüpfen & Feature Engineering
    # -------------------------------------------------------------------
    print("Lade Zahlungs-, Artikel- und Produktdaten...")
    df_payments = pd.read_csv(PAYMENTS_PATH)
    df_items = pd.read_csv(ITEMS_PATH)
    df_products = pd.read_csv(PRODUCTS_PATH)
    
    # 1. Zahlungen vorbereiten (Nur Kreditkarte, da Boleto keine echten Raten hat)
    cc_payments = df_payments[df_payments['payment_type'] == 'credit_card'].copy()
    # Wir nehmen den maximalen Raten-Wert und die Summe der Zahlung pro Bestellung
    order_pay = cc_payments.groupby('order_id').agg({
        'payment_value': 'sum',
        'payment_installments': 'max'
    }).reset_index()
    
    # 2. Artikel und Produkte mergen, um das Gewicht zu bekommen
    items_prod = pd.merge(df_items, df_products[['product_id', 'product_weight_g']], on='product_id', how='inner')
    
    # 3. Versandkosten und Gewicht pro Bestellung aggregieren
    order_features = items_prod.groupby('order_id').agg({
        'freight_value': 'sum',
        'product_weight_g': 'sum'
    }).reset_index()
    
    # Umrechnung in KG für besser interpretierbare Koeffizienten im Modell
    order_features['product_weight_kg'] = order_features['product_weight_g'] / 1000.0
    order_features = order_features.drop(columns=['product_weight_g'])
    
    # 4. Alles zusammenführen
    df_merged = pd.merge(order_pay, order_features, on='order_id', how='inner')
    
    # Bereinigen (Extreme Ausreißer kappen)
    df_clean = df_merged.dropna().copy()
    df_clean = df_clean[(df_clean['payment_value'] > 10) & (df_clean['payment_value'] < 1500)]
    df_clean = df_clean[(df_clean['freight_value'] < 200) & (df_clean['product_weight_kg'] < 30)]
    
    # 5. Zielvariable (Target) erstellen: 1 = Ratenzahlung (>1 Rate), 0 = Einmalzahlung
    df_clean['is_installment'] = (df_clean['payment_installments'] > 1).astype(int)
    
    # Zufallssample für saubere Visualisierung
    df_sample = df_clean.sample(n=10000, random_state=42)
    
    print(f"Trainiere Logit-Modell mit {len(df_sample)} Kreditkarten-Bestellungen...")

    # -------------------------------------------------------------------
    # 3. Logistisches Regressionsmodell trainieren
    # -------------------------------------------------------------------
    # X (Features) und y (Target)
    X = df_sample[['payment_value', 'freight_value', 'product_weight_kg']]
    y = df_sample['is_installment']
    
    # Konstante hinzufügen
    X_with_const = sm.add_constant(X)
    
    # Logistisches Modell (Logit) trainieren
    logit_model = sm.Logit(y, X_with_const).fit(disp=False)
    
    print("\n" + "="*80)
    print("LOGISTISCHE REGRESSION - ZUSAMMENFASSUNG")
    print("="*80)
    print(logit_model.summary())
    print("="*80 + "\n")

    # -------------------------------------------------------------------
    # 4. Odds Ratios (Quotenverhältnisse) berechnen
    # -------------------------------------------------------------------
    odds_ratios = np.exp(logit_model.params)
    conf_int = np.exp(logit_model.conf_int())
    conf_int.columns = ['Lower CI', 'Upper CI']
    
    or_df = pd.DataFrame({'Odds Ratio': odds_ratios})
    or_df = or_df.join(conf_int).drop('const') # Konstante verwerfen
    
    print("Odds Ratios (Wie stark beeinflusst 1 Einheit des Features die Chance auf Ratenzahlung?):")
    print(or_df)

    # -------------------------------------------------------------------
    # 5. Wunderschöne Visualisierung (2-teiliges Dashboard)
    # -------------------------------------------------------------------
    print("\nGeneriere Visualisierungs-Grid...")
    sns.set_theme(style="whitegrid")
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))

    # --- Plot 1: Die Sigmoid-Kurve (Wahrscheinlichkeit vs. Bestellwert) ---
    # Wir sortieren den Bestellwert für eine saubere Linienführung
    sorted_idx = np.argsort(df_sample['payment_value'])
    x_plot = df_sample['payment_value'].iloc[sorted_idx]
    
    # Wahrscheinlichkeiten vorhersagen (Versand & Gewicht bleiben auf Durchschnitt fixiert)
    mean_freight = df_sample['freight_value'].mean()
    mean_weight = df_sample['product_weight_kg'].mean()
    
    X_pred = pd.DataFrame({
        'const': 1,
        'payment_value': x_plot,
        'freight_value': mean_freight,
        'product_weight_kg': mean_weight
    })
    
    y_prob = logit_model.predict(X_pred)
    
    # Echte 1/0 Datenpunkte (halbtransparent & mit leichtem vertikalen Rauschen 'Jitter')
    axes[0].scatter(df_sample['payment_value'], y + np.random.uniform(-0.02, 0.02, size=len(y)), 
                    color='#8e44ad', alpha=0.05, s=10, label='Echte Käufe (1=Raten, 0=Einmal)')
    
    # Logistische S-Kurve
    axes[0].plot(x_plot, y_prob, color='#f39c12', linewidth=4, label='P(Ratenzahlung) - Vorhersage')
    
    axes[0].set_title("1. Modell-Logik: Treiber für Ratenzahlung", fontsize=14, fontweight="bold")
    axes[0].set_xlabel("Gesamtbestellwert (BRL)")
    axes[0].set_ylabel("Wahrscheinlichkeit für Ratenzahlung")
    axes[0].set_ylim(-0.05, 1.05)
    axes[0].legend(loc='lower right')

    # --- Plot 2: Forest Plot der Odds Ratios ---
    features = ['Bestellwert\n(pro BRL)', 'Versandkosten\n(pro BRL)', 'Gewicht\n(pro KG)']
    y_pos = np.arange(len(features))
    
    xerr_lower = or_df['Odds Ratio'] - or_df['Lower CI']
    xerr_upper = or_df['Upper CI'] - or_df['Odds Ratio']
    
    axes[1].axvline(1.0, color='black', linestyle='--', linewidth=2, zorder=1)
    
    axes[1].errorbar(or_df['Odds Ratio'], y_pos, xerr=[xerr_lower, xerr_upper], 
                     fmt='D', color='#2c3e50', ecolor='#2c3e50', elinewidth=3, capsize=6, markersize=10, zorder=2)
    
    axes[1].set_yticks(y_pos)
    axes[1].set_yticklabels(features, fontsize=12)
    axes[1].set_title("2. Business Impact: Quotenverhältnis (Odds Ratios)", fontsize=14, fontweight="bold")
    axes[1].set_xlabel("Odds Ratio (< 1 = Eher Einmalzahlung | > 1 = Eher Ratenzahlung)")
    
    # Für kleine Effekte (wie pro 1 BRL) multiplizieren wir gedanklich für das Text-Label
    for i, (or_val, name) in enumerate(zip(or_df['Odds Ratio'], features)):
        impact = (or_val - 1) * 100
        direction = "Steigerung" if impact > 0 else "Senkung"
        axes[1].text(or_val, y_pos[i] + 0.15, f"{or_val:.4f} ({abs(impact):.2f}% {direction})", 
                     ha='center', va='bottom', fontsize=11, fontweight='bold', color='#16a085')

    # Zusatzinfo in Plot 2
    info_text = (
        "Lesebeispiel:\nEin Odds Ratio von 1.006 beim Bestellwert bedeutet,\ndass "
        "jeder zusätzliche BRL die Chance auf eine\nRatenzahlung um 0.6% erhöht."
    )
    props = dict(boxstyle='round,pad=0.5', facecolor='white', alpha=0.9, edgecolor='gray')
    axes[1].text(0.95, 0.05, info_text, transform=axes[1].transAxes, fontsize=10,
                 verticalalignment='bottom', horizontalalignment='right', bbox=props)

    plt.suptitle("Logistische Regression: Warum wählen Kunden Ratenzahlung?", 
                 fontsize=18, fontweight="bold", y=1.05)
    plt.tight_layout()
    sns.despine()
    
    plt.savefig(GRAPHIC_PATH, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✅ Grafik erfolgreich gespeichert unter:\n{GRAPHIC_PATH}")

if __name__ == "__main__":
    main()
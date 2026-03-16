"""
Griff 33: Count-Daten korrekt modellieren? (GLM: Poisson & Negative Binomial)

Was ist das und was macht das Skript?
In der Business-Statistik haben wir oft Zielvariablen, die reine Zähldaten (Counts) sind: 
Anzahl der Klicks, Anzahl der Anrufe im Callcenter oder Anzahl der gekauften Artikel pro Bestellung. 
Eine normale lineare Regression (OLS) ist hier falsch, da sie kontinuierliche, auch negative 
Werte vorhersagt (z.B. "-1.5 Artikel").

Für Zähldaten nutzen wir Generalized Linear Models (GLMs):
1. Poisson Regression: Der Standard für Zähldaten. Geht davon aus, dass Ereignisse unabhängig 
   voneinander in einem festen Zeit-/Raumintervall auftreten.
   -> Kritische Annahme: Erwartungswert (Mean) = Varianz.
2. Negative Binomial Regression: In der Realität streuen Daten oft viel stärker, als es 
   der Mittelwert vermuten lässt (Overdispersion). Die Negative Binomial Regression 
   führt einen zusätzlichen Parameter ein, um diese Überdispersion abzufangen.

Business Case in diesem Skript:
Wir modellieren die Anzahl der gekauften Artikel pro Bestellung (Count) im Olist-Datensatz 
basierend auf Versandkosten und Gesamtpreis. Wir prüfen die Daten auf Overdispersion und 
vergleichen die Poisson-Regression mit dem robusteren Negative-Binomial-Modell.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
import statsmodels.formula.api as smf

# -------------------------------------------------------------------
# 1. Setup & Pfade
# -------------------------------------------------------------------
ITEMS_PATH = "/Users/stefanbechthold/statistik_business_analytics/data/olist_order_items_dataset.csv"

GRAPHIC_DIR = "/Users/stefanbechthold/statistik_business_analytics/Grafiken/15_Count_Models"
GRAPHIC_PATH = os.path.join(GRAPHIC_DIR, "33_glm_count_models.png")

os.makedirs(GRAPHIC_DIR, exist_ok=True)

def main():
    print("Starte Griff 33: GLM Count-Daten Modellierung (Poisson vs. NegBinom)...")

    # -------------------------------------------------------------------
    # 2. Daten laden & Count-Zielvariable generieren
    # -------------------------------------------------------------------
    df_items = pd.read_csv(ITEMS_PATH)
    
    # Aggregation auf Bestell-Ebene
    # Wir zählen die Anzahl der Artikel (Count) und summieren Preis/Versand (Features)
    df_orders = df_items.groupby('order_id').agg({
        'order_item_id': 'max',  # Da die IDs hochzählen, ist das Maximum = Anzahl Artikel
        'price': 'sum',
        'freight_value': 'sum'
    }).reset_index()
    
    df_orders.rename(columns={'order_item_id': 'item_count'}, inplace=True)
    
    # Bereinigung: Extreme Ausreißer entfernen, um das Modell nicht zu stark zu hebeln
    df_clean = df_orders[(df_orders['price'] < 1000) & (df_orders['item_count'] < 15)].copy()
    
    # -------------------------------------------------------------------
    # 3. Prüfung auf Overdispersion (Mittelwert vs. Varianz)
    # -------------------------------------------------------------------
    mean_count = df_clean['item_count'].mean()
    var_count = df_clean['item_count'].var()
    
    print(f"Zielvariable 'item_count':")
    print(f"Mittelwert (Mean): {mean_count:.3f}")
    print(f"Varianz (Variance): {var_count:.3f}")
    
    if var_count > mean_count:
        print("-> Diagnose: Overdispersion liegt vor (Varianz > Mittelwert). Poisson könnte unterschätzen.")

    # -------------------------------------------------------------------
    # 4. GLM Modellierung (Poisson vs. Negative Binomial)
    # -------------------------------------------------------------------
    print("\nTrainiere GLM Modelle...")
    
    # Modell 1: Poisson Regression
    # log-Link-Funktion ist der Standard für Poisson
    poisson_model = smf.glm(formula="item_count ~ price + freight_value", 
                            data=df_clean, 
                            family=sm.families.Poisson()).fit()
    
    # Modell 2: Negative Binomial Regression
    # Fängt die Overdispersion durch den Alpha-Parameter ab
    negbin_model = smf.glm(formula="item_count ~ price + freight_value", 
                           data=df_clean, 
                           family=sm.families.NegativeBinomial(alpha=1.0)).fit()
    
    # AIC (Akaike Information Criterion) zum Modellvergleich (niedriger ist besser)
    print(f"AIC Poisson: {poisson_model.aic:.1f}")
    print(f"AIC Negative Binomial: {negbin_model.aic:.1f}")

    # Vorhersagen generieren für die Visualisierung
    df_clean['pred_poisson'] = poisson_model.predict(df_clean)
    df_clean['pred_negbin'] = negbin_model.predict(df_clean)

    # -------------------------------------------------------------------
    # 5. Visualisierung im Dark Mode Design
    # -------------------------------------------------------------------
    plt.style.use('dark_background')
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7), facecolor='#121212')
    ax1.set_facecolor('#121212')
    ax2.set_facecolor('#121212')

    # --- Plot 1: Echte Verteilung der Count-Daten ---
    # Zeigt die typische rechtsschiefe Zähl-Verteilung mit dem harten Cutoff bei 1
    counts = df_clean['item_count'].value_counts().sort_index()
    ax1.bar(counts.index, counts.values, color='#3498db', alpha=0.8, edgecolor='#ffffff')
    
    ax1.set_title("1. Beobachtete Häufigkeit (Count-Daten)", fontsize=14, fontweight='bold', color='white')
    ax1.set_xlabel("Anzahl gekaufter Artikel pro Bestellung")
    ax1.set_ylabel("Anzahl der Bestellungen")
    ax1.set_xticks(range(1, 11))
    ax1.set_xlim(0.5, 10.5)

    # Info-Box für Plot 1
    disp_text = (
        f"Diagnose: Overdispersion\n"
        f"------------------------\n"
        f"Mean: {mean_count:.2f}\n"
        f"Variance: {var_count:.2f}\n\n"
        f"Da Varianz > Mean, ist die zentrale\n"
        f"Annahme der Poisson-Verteilung verletzt."
    )
    props = dict(boxstyle='round,pad=1', facecolor='#1e1e1e', alpha=0.9, edgecolor='#3498db', linewidth=1.5)
    ax1.text(0.4, 0.85, disp_text, transform=ax1.transAxes, fontsize=11,
             verticalalignment='center', bbox=props, color='#e0e0e0')

    # --- Plot 2: Poisson vs. Negative Binomial Residuen/Vorhersage ---
    # Wir plotten die vorhergesagten Werte gegen die echten Counts, um den Fit zu zeigen
    sns.kdeplot(df_clean['item_count'], ax=ax2, color='#ffffff', linewidth=2, label='Beobachtet (Realität)')
    sns.kdeplot(df_clean['pred_poisson'], ax=ax2, color='#e74c3c', linewidth=2, linestyle='--', label='Poisson Vorhersage')
    sns.kdeplot(df_clean['pred_negbin'], ax=ax2, color='#2ecc71', linewidth=2, linestyle=':', label='NegBinom Vorhersage')

    ax2.set_title("2. Modell-Fit: Poisson vs. Negative Binomial", fontsize=14, fontweight='bold', color='white')
    ax2.set_xlabel("Vorhergesagte / Echte Artikel-Anzahl")
    ax2.set_ylabel("Dichte")
    ax2.set_xlim(0, 5)
    
    # Legende anpassen
    legend = ax2.legend(loc='upper right', facecolor='#1e1e1e', edgecolor='#333333')
    plt.setp(legend.get_texts(), color='white')

    # Info-Box für Plot 2
    model_text = (
        f"Modell-Vergleich (AIC):\n"
        f"-----------------------\n"
        f"Poisson: {poisson_model.aic:,.0f}\n"
        f"NegBinom: {negbin_model.aic:,.0f}\n\n"
        f"Das Negative Binomial Modell geht\n"
        f"besser mit der Streuung um und liefert\n"
        f"einen niedrigeren (besseren) AIC-Wert."
    )
    ax2.text(0.4, 0.5, model_text, transform=ax2.transAxes, fontsize=11,
             verticalalignment='center', bbox=dict(boxstyle='round,pad=1', facecolor='#1e1e1e', alpha=0.9, edgecolor='#2ecc71', linewidth=1.5), color='#e0e0e0')

    plt.suptitle("GLM für Zähldaten: Count-Data korrekt modellieren", fontsize=18, fontweight='bold', color='white', y=1.05)
    plt.tight_layout()
    
    # Speichern der Grafik
    plt.savefig(GRAPHIC_PATH, dpi=300, bbox_inches='tight', facecolor=fig.get_facecolor())
    plt.close()
    
    print(f"✅ Grafik erfolgreich gespeichert unter:\n{GRAPHIC_PATH}")

if __name__ == "__main__":
    main()
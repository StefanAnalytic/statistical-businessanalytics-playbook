"""
Griff 9: Ist der Umsatzunterschied zufällig? (t-Tests)

Was ist das und was macht das Skript?
Dieses Skript führt einen Welch's t-Test durch. Wir vergleichen den durchschnittlichen 
Bestellwert von zwei Kundengruppen: Einmalzahler (1 Rate) vs. Ratenzahler (>1 Rate).
Der t-Test berechnet einen p-Wert (p-value). Dieser sagt uns, wie wahrscheinlich es ist, 
dass der beobachtete Unterschied zwischen den Gruppen reiner Zufall ist.

Warum der Welch's t-Test?
Der klassische Student's t-Test geht davon aus, dass beide Gruppen exakt die gleiche 
Varianz (Streuung) haben. Im Business ist das fast nie der Fall (Ratenzahler haben oft 
eine viel extremere Streuung nach oben). Der Welch's t-Test korrigiert das und ist der 
heutige Goldstandard.

Wofür ist das im Business gut?
A/B-Testing! Du änderst den Checkout-Button und die Conversion Rate steigt von 2.1% auf 2.3%. 
Ist das ein echter Erfolg oder nur statistisches Rauschen? Wenn der p-Wert deines t-Tests 
unter 0.05 (5%) liegt, sprichst du von "statistischer Signifikanz" – der Effekt ist höchstwahrscheinlich echt.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats

# -------------------------------------------------------------------
# 1. Setup & Pfade
# -------------------------------------------------------------------
DATA_PATH = "/Users/stefanbechthold/statistik_business_analytics/data/olist_order_payments_dataset.csv"
GRAPHIC_DIR = "/Users/stefanbechthold/statistik_business_analytics/Grafiken/03_Hypothesentests_und_Inferenz"
GRAPHIC_PATH = os.path.join(GRAPHIC_DIR, "09_t_test_ergebnis.png")

os.makedirs(GRAPHIC_DIR, exist_ok=True)

def main():
    print("🚀 Starte Griff 9: Signifikanzprüfung mit t-Tests...")
    
    # -------------------------------------------------------------------
    # 2. Daten laden & vorbereiten (Einmalzahler vs. Ratenzahler)
    # -------------------------------------------------------------------
    print(f"Lade Daten von: {DATA_PATH}")
    df = pd.read_csv(DATA_PATH)
    
    # Wir filtern uns die Zahlungswerte der beiden Gruppen heraus
    # dropna() entfernt leere Zeilen, um Berechnungsfehler zu vermeiden
    group_single = df[df['payment_installments'] == 1]['payment_value'].dropna().values
    group_installments = df[df['payment_installments'] > 1]['payment_value'].dropna().values
    
    print(f"Stichprobengröße Einmalzahler: {len(group_single)}")
    print(f"Stichprobengröße Ratenzahler:  {len(group_installments)}")
    
    mean_single = np.mean(group_single)
    mean_inst = np.mean(group_installments)
    print(f"\nDurchschnitt Einmalzahler: {mean_single:.2f} BRL")
    print(f"Durchschnitt Ratenzahler:  {mean_inst:.2f} BRL")
    print(f"Beobachtete Differenz:     {mean_inst - mean_single:.2f} BRL")

    # -------------------------------------------------------------------
    # 3. Welch's t-Test durchführen
    # -------------------------------------------------------------------
    print("\nFühre Welch's t-Test durch (nimmt ungleiche Varianzen an)...")
    
    # ttest_ind ist der t-Test für unabhängige Stichproben.
    # equal_var=False macht daraus automatisch den überlegenen Welch's t-Test!
    t_stat, p_value = stats.ttest_ind(group_installments, group_single, equal_var=False)
    
    print(f"T-Statistik: {t_stat:.4f}")
    print(f"P-Wert:      {p_value:.4e} (Wissenschaftliche Notation)")
    
    # Interpretation festlegen
    alpha = 0.05
    if p_value < alpha:
        interpretation = "Signifikant! (Der Unterschied ist höchstwahrscheinlich echt)"
        color_sig = "#27ae60" # Grün
    else:
        interpretation = "Nicht signifikant. (Der Unterschied könnte Zufall sein)"
        color_sig = "#e74c3c" # Rot

    # -------------------------------------------------------------------
    # 4. Professionelle Visualisierung (Violin-Plot & Stats-Text)
    # -------------------------------------------------------------------
    print("Generiere Visualisierungs-Grid...")
    sns.set_theme(style="whitegrid")
    
    fig, ax = plt.subplots(figsize=(12, 7))
    
    # Wir erstellen ein DataFrame für Seaborn, begrenzen Ausreißer für den Plot auf 1000 BRL
    plot_df = pd.DataFrame({
        'Zahlungsart': ['Einmalzahlung'] * len(group_single) + ['Ratenzahlung'] * len(group_installments),
        'Umsatz (BRL)': np.concatenate([group_single, group_installments])
    })
    # Filter für die Grafik (damit die Boxplots nicht zu Strichen zusammengestaucht werden)
    plot_df = plot_df[plot_df['Umsatz (BRL)'] <= 800]

    # Ein Violinplot kombiniert Boxplot mit Dichteverteilung (KDE)
    sns.violinplot(
        data=plot_df, 
        x='Zahlungsart', 
        y='Umsatz (BRL)', 
        palette=['#3498db', '#9b59b6'], 
        inner="quartile", # Zeigt die Quartile als gestrichelte Linien im Violin an
        ax=ax
    )
    
    # Mittelwerte als Rote Rauten einzeichnen
    ax.scatter(0, mean_single, color='red', marker='D', s=100, zorder=3, label="Mittelwert")
    ax.scatter(1, mean_inst, color='red', marker='D', s=100, zorder=3)
    
    ax.set_title("Umsatzvergleich: Einmalzahler vs. Ratenzahler (Welch's t-Test)", fontsize=16, fontweight="bold")
    ax.set_ylabel("Bestellwert in BRL (begrenzt auf 800 für Sichtbarkeit)", fontsize=12)
    ax.set_xlabel("")
    ax.legend(loc="upper left")

    # Textbox mit den statistischen Ergebnissen ins Diagramm einfügen
    stats_text = (
        f"Statistische Auswertung (Welch's t-Test):\n"
        f"-----------------------------------------\n"
        f"Differenz: +{mean_inst - mean_single:.2f} BRL\n"
        f"t-Statistik: {t_stat:.2f}\n"
        f"p-Wert: {p_value:.5f}\n"
        f"Fazit: {interpretation}"
    )
    
    # Die Bounding Box (bbox) macht einen schönen Kasten um den Text
    props = dict(boxstyle='round,pad=0.8', facecolor='white', alpha=0.9, edgecolor=color_sig, linewidth=2)
    ax.text(0.5, 0.85, stats_text, transform=ax.transAxes, fontsize=12,
            verticalalignment='top', horizontalalignment='center', bbox=props, color='#2c3e50')

    # Layout optimieren und speichern
    plt.tight_layout()
    sns.despine()
    
    plt.savefig(GRAPHIC_PATH, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✅ Grafik erfolgreich gespeichert unter:\n{GRAPHIC_PATH}")

if __name__ == "__main__":
    main()
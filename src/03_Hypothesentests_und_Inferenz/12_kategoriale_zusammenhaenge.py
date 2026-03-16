"""
Griff 12: Zusammenhang zwischen kategorialen Variablen? (Chi-Square & Fisher Exact)

Was ist das und was macht das Skript?
Bisher haben wir immer Zahlen (Umsatz, Lieferzeit) verglichen. Aber was ist, wenn wir 
Text-Kategorien (Kategoriale Variablen) vergleichen wollen? 
Frage: Hängt die Zahlungsart (Kreditkarte vs. Boleto) mit der Kundenzufriedenheit (Gut vs. Schlecht) zusammen?

Dafür bauen wir eine Kreuztabelle (Contingency Table) und nutzen zwei Tests:
1. Chi-Square Test (Chi-Quadrat): Der Standard-Test für große Datenmengen. Er vergleicht 
   die tatsächlich beobachteten Häufigkeiten mit den Häufigkeiten, die wir erwarten würden, 
   wenn es KEINEN Zusammenhang gäbe.
2. Fisher's Exact Test: Funktioniert genau wie Chi-Square für 2x2 Tabellen, berechnet 
   die Wahrscheinlichkeit aber *exakt*. Er ist extrem wichtig, wenn wir sehr kleine 
   Datenmengen haben (z.B. < 5 Beobachtungen in einer Zelle der Tabelle), wo Chi-Square scheitert.

Wofür ist das im Business gut?
Essentiell für Conversion-Rate-Analysen, Marketing-Zielgruppen oder Feature-Adoption!
Beispiel: Konvertieren iOS-Nutzer häufiger als Android-Nutzer? (Device = Kategorie 1, 
Conversion Ja/Nein = Kategorie 2). Der Chi-Square Test sagt dir, ob der Unterschied 
echt oder nur Zufall ist.
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
PAYMENTS_PATH = "/Users/stefanbechthold/statistik_business_analytics/data/olist_order_payments_dataset.csv"
REVIEWS_PATH = "/Users/stefanbechthold/statistik_business_analytics/data/olist_order_reviews_dataset.csv"
GRAPHIC_DIR = "/Users/stefanbechthold/statistik_business_analytics/Grafiken/03_Hypothesentests_und_Inferenz"
GRAPHIC_PATH = os.path.join(GRAPHIC_DIR, "12_kategoriale_zusammenhaenge.png")

os.makedirs(GRAPHIC_DIR, exist_ok=True)

def main():
    print("🚀 Starte Griff 12: Chi-Square & Fisher Exact Test...")
    
    # -------------------------------------------------------------------
    # 2. Daten laden & vorbereiten
    # -------------------------------------------------------------------
    print("Lade Zahlungs- und Bewertungsdaten...")
    df_payments = pd.read_csv(PAYMENTS_PATH)
    df_reviews = pd.read_csv(REVIEWS_PATH)
    
    # Wir nehmen nur die Haupt-Zahlungsart pro Bestellung (um Duplikate zu vermeiden)
    df_pay_unique = df_payments.drop_duplicates(subset=['order_id'])
    df_rev_unique = df_reviews.drop_duplicates(subset=['order_id'])
    
    # Daten mergen
    df_merged = pd.merge(df_pay_unique, df_rev_unique, on='order_id', how='inner')
    
    # Wir filtern auf Kreditkarte und Boleto (Brasilianische Bar-Rechnung)
    df_filtered = df_merged[df_merged['payment_type'].isin(['credit_card', 'boleto'])].copy()
    
    # Aus den 1-5 Sternen machen wir zwei klare Kategorien: "Gut" (4-5) und "Schlecht" (1-3)
    # Das macht die Interpretation im Business oft einfacher und greifbarer.
    df_filtered['satisfaction'] = np.where(df_filtered['review_score'] >= 4, 'Gut (4-5 Sterne)', 'Schlecht (1-3 Sterne)')
    df_filtered['payment_type'] = df_filtered['payment_type'].replace({'credit_card': 'Kreditkarte', 'boleto': 'Boleto'})
    
    print(f"Analysiere {len(df_filtered)} Bestellungen...")

    # -------------------------------------------------------------------
    # 3. Kreuztabelle (Contingency Table) erstellen
    # -------------------------------------------------------------------
    # Eine Kreuztabelle zählt einfach, wie oft welche Kombination vorkommt.
    contingency_table = pd.crosstab(df_filtered['payment_type'], df_filtered['satisfaction'])
    print("\nBeobachtete Kreuztabelle:")
    print(contingency_table)

    # -------------------------------------------------------------------
    # 4. Statistische Tests durchführen
    # -------------------------------------------------------------------
    print("\nFühre Chi-Square und Fisher's Exact Test durch...")
    
    # A) Chi-Square Test
    # Gibt 4 Werte zurück: Chi2-Statistik, p-Wert, Freiheitsgrade (dof), und die erwartete Tabelle
    chi2_stat, p_value_chi2, dof, expected_table = stats.chi2_contingency(contingency_table)
    
    # B) Fisher's Exact Test
    # Funktioniert nur für 2x2 Tabellen. Berechnet Odds Ratio und exakten p-Wert.
    odds_ratio, p_value_fisher = stats.fisher_exact(contingency_table)
    
    print(f"Chi-Square p-Wert: {p_value_chi2:.4e}")
    print(f"Fisher Exact p-Wert: {p_value_fisher:.4e}")
    
    alpha = 0.05
    if p_value_chi2 < alpha:
        interpretation = "Signifikanter Zusammenhang!\nZahlungsart & Zufriedenheit sind abhängig."
        color_sig = "#27ae60" # Grün
    else:
        interpretation = "Kein signifikanter Zusammenhang.\nBeide Variablen sind unabhängig."
        color_sig = "#e74c3c" # Rot

    # -------------------------------------------------------------------
    # 5. Wunderschöne Visualisierung (Heatmap + Stacked Bar Chart)
    # -------------------------------------------------------------------
    print("Generiere Visualisierungs-Grid...")
    sns.set_theme(style="whitegrid")
    
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    
    # --- Plot 1: Heatmap der Kreuztabelle ---
    # Zeigt visuell, wo die meisten Datenpunkte liegen
    sns.heatmap(contingency_table, annot=True, fmt="d", cmap="Blues", cbar=False, 
                annot_kws={"size": 14, "weight": "bold"}, ax=axes[0])
    
    axes[0].set_title("Beobachtete Häufigkeiten (Kreuztabelle)", fontsize=14, fontweight="bold")
    axes[0].set_xlabel("Kundenzufriedenheit", fontsize=12)
    axes[0].set_ylabel("Zahlungsart", fontsize=12)
    
    # --- Plot 2: 100% Stacked Bar Chart ---
    # Zeigt die PROZENTUALE Verteilung innerhalb der Zahlungsarten.
    # So sieht man sofort, ob z.B. Boleto-Zahler prozentual öfter schlechte Bewertungen geben.
    
    # Wir berechnen die Prozente pro Zeile (axis=0)
    contingency_pct = contingency_table.div(contingency_table.sum(axis=1), axis=0) * 100
    
    # Plotten des Stacked Bar Charts
    contingency_pct.plot(kind='bar', stacked=True, color=['#2ecc71', '#e74c3c'], ax=axes[1], edgecolor='black')
    
    axes[1].set_title("Prozentuale Verteilung der Zufriedenheit pro Zahlungsart", fontsize=14, fontweight="bold")
    axes[1].set_xlabel("Zahlungsart", fontsize=12)
    axes[1].set_ylabel("Anteil in %", fontsize=12)
    axes[1].set_xticklabels(axes[1].get_xticklabels(), rotation=0)
    
    # Legende anpassen
    axes[1].legend(title="Zufriedenheit", loc='upper left', bbox_to_anchor=(1, 1))
    
    # Eine horizontale Linie bei z.B. 50% zur besseren Lesbarkeit
    axes[1].axhline(50, color='gray', linestyle='--', alpha=0.5)

    # Textbox mit den statistischen Ergebnissen
    stats_text = (
        f"Statistische Tests:\n"
        f"-------------------\n"
        f"Chi-Square p-Wert: {p_value_chi2:.5f}\n"
        f"Fisher Exact p-Wert: {p_value_fisher:.5f}\n"
        f"Fazit: {interpretation}"
    )
    
    props = dict(boxstyle='round,pad=0.8', facecolor='white', alpha=0.9, edgecolor=color_sig, linewidth=2)
    axes[1].text(1.05, 0.5, stats_text, transform=axes[1].transAxes, fontsize=12,
                 verticalalignment='center', bbox=props, color='#2c3e50')

    # Layout optimieren und speichern
    plt.suptitle("Kategoriale Analyse: Hängt die Zahlungsart mit der Zufriedenheit zusammen?", 
                 fontsize=18, fontweight="bold", y=1.05)
    
    # Adjust layout to make room for the legend and text box on the right
    plt.tight_layout(rect=[0, 0, 0.85, 1]) 
    sns.despine()
    
    plt.savefig(GRAPHIC_PATH, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✅ Grafik erfolgreich gespeichert unter:\n{GRAPHIC_PATH}")

if __name__ == "__main__":
    main()
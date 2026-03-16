"""
Griff 11: Unterschied über >2 Gruppen? (ANOVA + Post-hoc Tukey)

Was ist das und was macht das Skript?
Wenn wir nur zwei Gruppen vergleichen (z.B. Einmalzahler vs. Ratenzahler), reicht ein t-Test. 
Aber was, wenn wir 4 Gruppen haben? (z.B. Umsatz nach Zahlungsart: Kreditkarte vs. Boleto vs. 
Voucher vs. Debitkarte). 

Würden wir hier einfach für jedes Paar einen eigenen t-Test machen, würden wir uns das 
Problem des "Multiple Testings" einhandeln: Die Fehlerwahrscheinlichkeit addiert sich auf 
und wir finden plötzlich signifikante Unterschiede, wo gar keine sind!

Die Lösung in 2 Schritten:
1. ANOVA (Analysis of Variance): Ein globaler Test. Er fragt: "Gibt es ZUMINDEST EINEN 
   signifikanten Unterschied irgendwo zwischen diesen 4 Gruppen?"
2. Post-Hoc Tukey HSD Test: Wenn die ANOVA "Ja" sagt, sucht Tukey als Detektiv genau heraus, 
   WELCHE spezifischen Gruppen sich voneinander unterscheiden, ohne unsere Fehlerrate zu ruinieren.

Wofür ist das im Business gut?
Perfekt für Preis-Tiers (Basic vs. Pro vs. Enterprise), Marketing-Kanäle (SEO vs. SEA vs. Social) 
oder wie hier: Zahlungsarten. Du erkennst sauber und statistisch wasserdicht, ob sich ein Kanal/Tier 
wirklich vom anderen abhebt.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats

# Statsmodels liefert uns den Tukey HSD Test
from statsmodels.stats.multicomp import pairwise_tukeyhsd

# -------------------------------------------------------------------
# 1. Setup & Pfade
# -------------------------------------------------------------------
DATA_PATH = "/Users/stefanbechthold/statistik_business_analytics/data/olist_order_payments_dataset.csv"
GRAPHIC_DIR = "/Users/stefanbechthold/statistik_business_analytics/Grafiken/03_Hypothesentests_und_Inferenz"
GRAPHIC_PATH = os.path.join(GRAPHIC_DIR, "11_anova_tukey.png")

os.makedirs(GRAPHIC_DIR, exist_ok=True)

def main():
    print("🚀 Starte Griff 11: ANOVA & Tukey HSD Test...")
    
    # -------------------------------------------------------------------
    # 2. Daten laden & vorbereiten (Umsatz nach 4 Zahlungsarten)
    # -------------------------------------------------------------------
    print(f"Lade Daten von: {DATA_PATH}")
    df = pd.read_csv(DATA_PATH)
    
    # Wir filtern auf die 4 Haupt-Zahlungsarten in Brasilien
    valid_types = ['credit_card', 'boleto', 'voucher', 'debit_card']
    df_filtered = df[df['payment_type'].isin(valid_types)].copy()
    
    # Um die extreme Schiefe des Umsatzes zu mildern und die ANOVA-Annahmen 
    # (Normalverteilung) besser zu erfüllen, begrenzen wir extreme Ausreißer (>1000 BRL).
    # In der Praxis könnte man hier auch die Log-Transformation aus Griff 4 nutzen!
    df_filtered = df_filtered[df_filtered['payment_value'] <= 1000]
    
    # Gruppen-Arrays für die ANOVA vorbereiten
    group_cc = df_filtered[df_filtered['payment_type'] == 'credit_card']['payment_value'].values
    group_bol = df_filtered[df_filtered['payment_type'] == 'boleto']['payment_value'].values
    group_vou = df_filtered[df_filtered['payment_type'] == 'voucher']['payment_value'].values
    group_deb = df_filtered[df_filtered['payment_type'] == 'debit_card']['payment_value'].values
    
    print("\nGruppengrößen:")
    print(f"Kreditkarte: {len(group_cc)} | Boleto: {len(group_bol)} | Voucher: {len(group_vou)} | Debit: {len(group_deb)}")

    # -------------------------------------------------------------------
    # 3. Schritt 1: One-way ANOVA durchführen
    # -------------------------------------------------------------------
    print("\nFühre globale ANOVA durch...")
    # Nullhypothese der ANOVA: Alle Gruppenmittelwerte sind exakt gleich.
    f_stat, p_value_anova = stats.f_oneway(group_cc, group_bol, group_vou, group_deb)
    
    print(f"ANOVA F-Statistik: {f_stat:.2f}")
    print(f"ANOVA p-Wert:      {p_value_anova:.4e}")
    
    if p_value_anova < 0.05:
        print("-> ANOVA ist signifikant! Mindestens eine Gruppe unterscheidet sich. Starte Tukey Test...")
        
        # -------------------------------------------------------------------
        # 4. Schritt 2: Tukey HSD (Honestly Significant Difference)
        # -------------------------------------------------------------------
        # Wir übergeben alle Werte und die dazugehörigen Gruppen-Labels
        tukey_result = pairwise_tukeyhsd(endog=df_filtered['payment_value'], 
                                         groups=df_filtered['payment_type'], 
                                         alpha=0.05)
        
        # Zeige die Tukey-Ergebnistabelle im Terminal
        print("\nTukey HSD Ergebnisse:")
        print(tukey_result)
        
        # Wir wandeln die Ergebnisse in einen DataFrame um, um sie später im Plot zu nutzen
        tukey_df = pd.DataFrame(data=tukey_result._results_table.data[1:], 
                                columns=tukey_result._results_table.data[0])
    else:
        print("-> ANOVA ist NICHT signifikant. Kein Unterschied vorhanden. Breche ab.")
        return

    # -------------------------------------------------------------------
    # 5. Wunderschöne Visualisierung
    # -------------------------------------------------------------------
    print("\nGeneriere Visualisierung (Boxplot & Tukey-Intervalle)...")
    sns.set_theme(style="whitegrid")
    
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    
    # --- Plot 1: Boxplot der Gruppen ---
    # Zeigt die Verteilung und die Mediane/Quartile der 4 Zahlungsarten
    order = ['credit_card', 'boleto', 'voucher', 'debit_card']
    sns.boxplot(x='payment_type', y='payment_value', data=df_filtered, 
                order=order, palette='viridis', ax=axes[0], showfliers=False)
    
    # Mittelwerte als rote Punkte hinzufügen, da ANOVA Mittelwerte vergleicht!
    means = df_filtered.groupby('payment_type')['payment_value'].mean().reindex(order)
    axes[0].scatter(range(len(means)), means, color='red', marker='D', s=80, label='Mittelwert', zorder=5)
    
    axes[0].set_title("Umsatzverteilung nach Zahlungsart (bis 1000 BRL)", fontsize=14, fontweight="bold")
    axes[0].set_xlabel("Zahlungsart", fontsize=12)
    axes[0].set_ylabel("Bestellwert (BRL)", fontsize=12)
    axes[0].legend()

    # --- Plot 2: Tukey Konfidenzintervalle der Differenzen ---
    # Tukey berechnet das Konfidenzintervall für den *Unterschied* zwischen zwei Gruppen.
    # Schneidet das Intervall die Nulllinie (0), gibt es KEINEN signifikanten Unterschied.
    
    # Wir bauen ein eigenes kleines Plot für die Tukey-Ergebnisse
    y_labels = tukey_df['group1'] + " vs. " + tukey_df['group2']
    lower_bounds = tukey_df['lower']
    upper_bounds = tukey_df['upper']
    means_diff = tukey_df['meandiff']
    reject = tukey_df['reject'] # True = Signifikanter Unterschied
    
    y_pos = np.arange(len(y_labels))
    
    # Null-Linie einzeichnen (Referenz für "Kein Unterschied")
    axes[1].axvline(0, color='black', linestyle='--', linewidth=2, alpha=0.7)
    
    for i in range(len(y_pos)):
        # Farbe: Grün, wenn signifikant unterschiedlich (reject=True), sonst Grau
        color = '#27ae60' if reject[i] else '#7f8c8d'
        
        # Linie für das Konfidenzintervall zeichnen
        axes[1].plot([lower_bounds[i], upper_bounds[i]], [y_pos[i], y_pos[i]], color=color, linewidth=3)
        # Punkt für den Mittelwert der Differenz
        axes[1].plot(means_diff[i], y_pos[i], 'o', color=color, markersize=8)
        
    axes[1].set_yticks(y_pos)
    axes[1].set_yticklabels(y_labels)
    axes[1].set_title("Tukey HSD: Unterscheiden sich die Paare?", fontsize=14, fontweight="bold")
    axes[1].set_xlabel("Unterschied im Durchschnittsumsatz (BRL)")
    
    # Erklärung in den Plot schreiben
    props = dict(boxstyle='round,pad=0.5', facecolor='white', alpha=0.9, edgecolor='gray')
    axes[1].text(0.95, 0.05, "Schneidet die Linie die 0,\ngibt es KEINEN signifikanten\nUnterschied zwischen den Gruppen.", 
                 transform=axes[1].transAxes, fontsize=10, verticalalignment='bottom', horizontalalignment='right', bbox=props)

    # Layout optimieren und speichern
    plt.suptitle("Multi-Gruppen-Vergleich: ANOVA & Post-hoc Tukey Test", 
                 fontsize=18, fontweight="bold", y=1.05)
    plt.tight_layout()
    sns.despine()
    
    plt.savefig(GRAPHIC_PATH, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✅ Grafik erfolgreich gespeichert unter:\n{GRAPHIC_PATH}")

if __name__ == "__main__":
    main()
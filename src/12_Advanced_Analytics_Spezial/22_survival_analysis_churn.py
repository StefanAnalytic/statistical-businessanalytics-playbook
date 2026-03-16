"""
Griff 22: Wie lange bleiben Kunden? (Survival Analysis: Kaplan-Meier & Cox PH)

Was ist das und was macht das Skript?
"Survival Analysis" (Überlebenszeitanalyse) kommt aus der Medizin (Wie lange überlebt ein 
Patient nach einer Therapie?). Im Business nutzen wir sie für Churn (Kundenabwanderung) 
oder Retention (Wann kauft ein Kunde zum zweiten Mal?).

Das Geniale an dieser Methode: Sie kann mit "zensierten" Daten (Right-Censoring) umgehen!
Wenn ein Kunde heute bestellt hat, wissen wir nicht, ob er nie wieder bestellt (Churn) 
oder einfach nur *noch nicht* (weil erst 1 Tag vergangen ist). Normale Modelle scheitern hier.
Survival Analysis nutzt diese unvollständigen Informationen mathematisch korrekt.

Wir nutzen zwei Modelle:
1. Kaplan-Meier Schätzer: Berechnet die Überlebenskurve. "Wie viel Prozent der Kunden 
   haben nach 100 Tagen noch KEINEN zweiten Kauf getätigt?"
2. Cox Proportional Hazards (Cox PH): Die logistische Regression der Überlebenszeit! 
   Zeigt, welche Faktoren (z.B. Höhe des Erstkaufs, Versandkosten) das "Risiko" 
   (hier: die Chance auf einen Zweitkauf) erhöhen oder senken.

Business Case in diesem Skript:
Time-to-Second-Purchase! Wie viele Tage vergehen bis zum zweiten Kauf? 
Welchen Einfluss hat der Bestellwert des ersten Kaufs auf die Rückkehr-Wahrscheinlichkeit?


"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Lifelines ist DIE Python-Bibliothek für Survival Analysis
try:
    from lifelines import KaplanMeierFitter, CoxPHFitter
except ImportError:
    print("FEHLER: Die Bibliothek 'lifelines' fehlt.")
    print("Bitte öffne dein Terminal und tippe: pip install lifelines")
    exit()

# -------------------------------------------------------------------
# 1. Setup & Pfade
# -------------------------------------------------------------------
# Wir wechseln thematisch in Metriken & Wachstum (Churn/Retention)
ORDERS_PATH = "/Users/stefanbechthold/statistik_business_analytics/data/olist_orders_dataset.csv"
CUSTOMERS_PATH = "/Users/stefanbechthold/statistik_business_analytics/data/olist_customers_dataset.csv"
PAYMENTS_PATH = "/Users/stefanbechthold/statistik_business_analytics/data/olist_order_payments_dataset.csv"

GRAPHIC_DIR = "/Users/stefanbechthold/statistik_business_analytics/Grafiken/04_Metriken_und_Wachstum"
GRAPHIC_PATH = os.path.join(GRAPHIC_DIR, "22_survival_analysis_retention.png")

os.makedirs(GRAPHIC_DIR, exist_ok=True)

def main():
    print("🚀 Starte Griff 22: Survival Analysis (Kaplan-Meier & Cox PH)...")
    
    # -------------------------------------------------------------------
    # 2. Daten laden & Lifelines-Datensatz aufbauen (T und E)
    # -------------------------------------------------------------------
    print("Lade Bestellungen, Kunden und Zahlungen...")
    df_orders = pd.read_csv(ORDERS_PATH)
    df_customers = pd.read_csv(CUSTOMERS_PATH)
    df_payments = pd.read_csv(PAYMENTS_PATH)
    
    # 1. Zahlungen pro Bestellung aufsummieren
    order_spend = df_payments.groupby('order_id')['payment_value'].sum().reset_index()
    
    # 2. Wir brauchen die 'customer_unique_id', um echte Stammkunden zu erkennen
    # (In Olist kriegt man pro Bestellung eine neue 'customer_id', aber die 'unique_id' bleibt gleich)
    df_merged = pd.merge(df_orders, df_customers[['customer_id', 'customer_unique_id']], on='customer_id', how='inner')
    df_merged = pd.merge(df_merged, order_spend, on='order_id', how='left')
    
    # Datum konvertieren und sortieren
    df_merged['purchase_date'] = pd.to_datetime(df_merged['order_purchase_timestamp']).dt.date
    df_merged = df_merged.sort_values(by=['customer_unique_id', 'purchase_date'])
    
    print("\nBaue Survival-Tabelle (Dauer T und Event E)...")
    # Wann endet unser Datensatz? (Um zensierte Daten zu berechnen)
    MAX_DATE = df_merged['purchase_date'].max()
    
    # Wir gruppieren nach Kunde und holen uns das 1. und 2. Kaufdatum
    # .nth(0) ist der Erstkauf, .nth(1) ist der Zweitkauf
    first_orders = df_merged.groupby('customer_unique_id').nth(0).reset_index()
    second_orders = df_merged.groupby('customer_unique_id').nth(1).reset_index()
    
    # Wir benennen die Spalten um, für einen sauberen Merge
    first_orders = first_orders[['customer_unique_id', 'purchase_date', 'payment_value']]
    first_orders.columns = ['customer_unique_id', 'first_purchase_date', 'first_spend']
    
    second_orders = second_orders[['customer_unique_id', 'purchase_date']]
    second_orders.columns = ['customer_unique_id', 'second_purchase_date']
    
    # Zusammenführen: Jeder Kunde hat nun ein Erstkaufdatum und (vielleicht) ein Zweitkaufdatum
    survival_df = pd.merge(first_orders, second_orders, on='customer_unique_id', how='left')
    
    # Feature Engineering für Survival Analysis:
    # T (Time) = Dauer bis zum Event (oder bis zum Ende des Datensatzes)
    # E (Event) = 1 (Hat nochmal gekauft) oder 0 (Hat nicht nochmal gekauft = zensiert)
    
    # Wenn ein Zweitkauf existiert:
    survival_df['E'] = survival_df['second_purchase_date'].notna().astype(int)
    
    # Berechnung von T (in Tagen)
    def calculate_t(row):
        if row['E'] == 1:
            # Dauer zwischen erstem und zweitem Kauf
            return (row['second_purchase_date'] - row['first_purchase_date']).days
        else:
            # Dauer zwischen erstem Kauf und dem letzten Tag im gesamten Datensatz (Zensierung!)
            return (MAX_DATE - row['first_purchase_date']).days

    survival_df['T'] = survival_df.apply(calculate_t, axis=1)
    
    # Wir filtern unsinnige Datenpunkte heraus (z.B. zwei Bestellungen am exakt gleichen Tag T=0)
    survival_df = survival_df[survival_df['T'] > 0]
    
    # Für die Performance (und da Olist über 90k Kunden hat), ziehen wir ein Sample für das Modell
    # In der Realität würden wir natürlich alles nutzen, aber das reicht für die Erkenntnis.
    survival_sample = survival_df.sample(n=15000, random_state=42).copy()
    
    churn_rate = 1 - survival_sample['E'].mean()
    print(f"Datensatz: {len(survival_sample)} Kunden.")
    print(f"Davon haben {survival_sample['E'].sum()} einen Zweitkauf getätigt.")
    print(f"Right-Censored (Noch kein Zweitkauf): {churn_rate*100:.1f}%")

    # -------------------------------------------------------------------
    # 3. Kaplan-Meier Schätzer (Die Basis-Überlebenskurve)
    # -------------------------------------------------------------------
    print("\nBerechne Kaplan-Meier Kurve...")
    kmf = KaplanMeierFitter()
    # Wir geben dem Modell die Dauer (T) und ob das Event eingetreten ist (E)
    kmf.fit(durations=survival_sample['T'], event_observed=survival_sample['E'], label='Alle Kunden')

    # -------------------------------------------------------------------
    # 4. Cox Proportional Hazards (Einflussfaktoren modellieren)
    # -------------------------------------------------------------------
    print("Trainiere Cox PH Modell (Einfluss des Erstkauf-Werts)...")
    
    # Wir bereiten ein kleines DataFrame für das Cox-Modell vor.
    # Da Olist-Umsätze stark rechtsschief sind, logarithmieren wir den Wert, 
    # damit das Modell lineare Zusammenhänge besser greifen kann (Griff 4 lässt grüßen!).
    cox_df = survival_sample[['T', 'E', 'first_spend']].copy()
    cox_df['log_first_spend'] = np.log1p(cox_df['first_spend'])
    cox_df = cox_df.drop(columns=['first_spend']).dropna()
    
    cph = CoxPHFitter()
    # Wir fitten das Modell. duration_col ist T, event_col ist E
    cph.fit(cox_df, duration_col='T', event_col='E')
    
    print("\n" + "="*60)
    print("COX PROPORTIONAL HAZARDS - ZUSAMMENFASSUNG")
    print("="*60)
    cph.print_summary()
    print("="*60 + "\n")

    # -------------------------------------------------------------------
    # 5. Wunderschöne Visualisierung (2-teiliges Dashboard)
    # -------------------------------------------------------------------
    print("Generiere Visualisierungs-Grid...")
    sns.set_theme(style="whitegrid")
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))

    # --- Plot 1: Kaplan-Meier Überlebenskurve ---
    # Die "Survival Function" zeigt die Wahrscheinlichkeit, dass das Event (Zweitkauf) NOCH NICHT eingetreten ist.
    # Bei Churn/Retention drehen wir die Logik oft gedanklich um: Wir wollen sehen, wie die Kurve der 
    # "Noch-Nicht-Wiederkäufer" abflacht.
    
    kmf.plot_survival_function(ax=axes[0], color='#2980b9', linewidth=3, ci_alpha=0.2)
    
    axes[0].set_title("1. Kaplan-Meier: Zeit bis zum Zweitkauf", fontsize=14, fontweight="bold")
    axes[0].set_xlabel("Tage seit dem Erstkauf")
    axes[0].set_ylabel("Wahrscheinlichkeit (Noch KEIN Zweitkauf)")
    
    # Da extrem wenige Kunden in Olist ein zweites Mal kaufen (E-Commerce typisch für diesen Markt),
    # wird die Kurve nicht bis auf 0 fallen, sondern bei ca. 97% stehen bleiben.
    axes[0].set_ylim(0.95, 1.01) # Reinzoomen, damit man den Drop überhaupt sieht!
    
    # Text-Einschub in Plot 1
    axes[0].text(300, 0.96, "Die Kurve fällt extrem langsam.\nDas bedeutet: Sehr wenige Kunden\nkaufen ein zweites Mal.\nDie meisten bleiben 'zensiert'.", 
                 fontsize=11, bbox=dict(facecolor='white', alpha=0.9, edgecolor='#2980b9'))

    # --- Plot 2: Cox PH - Wie verändert der Umsatz die Überlebenskurve? ---
    # Das absolut coolste an Cox PH: Wir können simulieren, wie die Kurve aussieht,
    # wenn wir das Feature (log_first_spend) verändern!
    
    # Wir nehmen den Median, das 10% Quantil (Billig-Käufer) und 90% Quantil (Teuer-Käufer)
    spend_levels = [
        np.percentile(cox_df['log_first_spend'], 10), # Billig
        np.median(cox_df['log_first_spend']),         # Normal
        np.percentile(cox_df['log_first_spend'], 90)  # Premium
    ]
    
    # Wir nutzen plot_partial_effects_on_outcome aus lifelines
    # Es zeigt, wie sich die Kurve verhält, wenn wir 'log_first_spend' variieren
    cph.plot_partial_effects_on_outcome(covariates='log_first_spend', values=spend_levels, 
                                        cmap='coolwarm', ax=axes[1], linewidth=2)
    
    axes[1].set_title("2. Cox PH: Einfluss des Erstkauf-Werts auf Retention", fontsize=14, fontweight="bold")
    axes[1].set_xlabel("Tage seit dem Erstkauf")
    axes[1].set_ylabel("Wahrscheinlichkeit (Noch KEIN Zweitkauf)")
    
    # Eigene Legende für bessere Lesbarkeit
    axes[1].legend(['Billiger Erstkauf (10% Quantil)', 'Normaler Erstkauf (Median)', 'Teurer Erstkauf (90% Quantil)'], loc='lower left')
    axes[1].set_ylim(0.95, 1.01) # Gleicher Zoom-Faktor

    # Info-Box für Plot 2
    # Achtung: In Cox-Modellen bedeutet ein Hazard Ratio < 1, dass das Event WENIGER WAHRSCHEINLICH ist.
    # Event = Zweitkauf. Wenn HR < 1 bei teuren Käufen, heißt das: Teure Käufer kommen SELTENER zurück!
    hazard_ratio = np.exp(cph.params_['log_first_spend'])
    
    if hazard_ratio < 1:
        impact_text = "Teure Erstkäufer haben eine\nGERINGERE Chance zurückzukehren."
    else:
        impact_text = "Teure Erstkäufer haben eine\nHÖHERE Chance zurückzukehren."

    info_text = (
        f"Hazard Ratio (Umsatz): {hazard_ratio:.3f}\n"
        f"----------------------\n"
        f"{impact_text}\n"
        f"Die Kurven spalten sich auf:\n"
        f"Die rote Kurve (Teuer) bleibt höher,\n"
        f"das Event (Zweitkauf) tritt seltener ein."
    )
    props = dict(boxstyle='round,pad=0.8', facecolor='white', alpha=0.9, edgecolor='gray', linewidth=1.5)
    axes[1].text(300, 0.985, info_text, fontsize=11, verticalalignment='top', bbox=props, color='#2c3e50')

    # Layout optimieren und speichern
    plt.suptitle("Customer Retention: Time-to-Second-Purchase analysieren", 
                 fontsize=18, fontweight="bold", y=1.05)
    plt.tight_layout()
    sns.despine()
    
    plt.savefig(GRAPHIC_PATH, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✅ Grafik erfolgreich gespeichert unter:\n{GRAPHIC_PATH}")

if __name__ == "__main__":
    main()
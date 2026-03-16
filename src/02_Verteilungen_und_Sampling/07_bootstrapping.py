"""
Griff 7: Wie ziehe ich verlässliche Schlussfolgerungen ohne starke Annahmen? (Bootstrapping)

Was ist das und was macht das Skript?
Dieses Skript wendet "Bootstrapping" auf reale E-Commerce-Umsatzdaten (Payment per Order) an.
Anstatt komplizierte mathematische Formeln zu nutzen, die voraussetzen, dass unsere Daten 
perfekt normalverteilt sind (was sie fast nie sind!), simulieren wir einfach die Realität.

Wie funktioniert Bootstrapping? 
1. Wir ziehen aus unseren echten Daten zufällig eine neue Stichprobe in der gleichen Größe.
2. WICHTIG: Wir ziehen "mit Zurücklegen" (replace=True). Das heißt, ein einzelner Kauf 
   kann in der neuen Stichprobe mehrfach auftauchen, ein anderer gar nicht.
3. Wir berechnen unsere KPI (z.B. Mittelwert oder Median) für diese neue Stichprobe.
4. Das wiederholen wir 5.000 Mal!
5. So erhalten wir eine Verteilung unserer KPI und können exakt ablesen, in welchem 
   Bereich unser wahrer Durchschnitt mit 95%iger Wahrscheinlichkeit liegt (Konfidenzintervall).

Wofür ist das im Business gut?
Umsatzdaten sind oft extrem schief (einige wenige riesige Käufe). Klassische Konfidenzintervalle 
(wie der t-Test) scheitern hier oft oder liefern ungenaue Grenzen. Bootstrapping ist 
"Non-Parametrisch" (macht keine Annahmen über die Form der Daten) und liefert dir absolut 
robuste und verlässliche Konfidenzintervalle für JEDE Metrik (Mean, Median, Conversion Rate, etc.).
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# -------------------------------------------------------------------
# 1. Setup & Pfade
# -------------------------------------------------------------------
DATA_PATH = "/Users/stefanbechthold/statistik_business_analytics/data/olist_order_payments_dataset.csv"
GRAPHIC_DIR = "/Users/stefanbechthold/statistik_business_analytics/Grafiken/02_Verteilungen_und_Sampling"
GRAPHIC_PATH = os.path.join(GRAPHIC_DIR, "07_bootstrapping.png")

# Sicherstellen, dass der Zielordner existiert
os.makedirs(GRAPHIC_DIR, exist_ok=True)

def main():
    print("🚀 Starte Griff 7: Bootstrapping (Konfidenzintervalle simulieren)...")
    
    # -------------------------------------------------------------------
    # 2. Daten laden & vorbereiten (Umsatz pro Bestellung)
    # -------------------------------------------------------------------
    print(f"Lade Daten von: {DATA_PATH}")
    df_payments = pd.read_csv(DATA_PATH)
    
    # Da eine Bestellung aus mehreren Raten/Zahlungsarten bestehen kann, 
    # summieren wir den payment_value pro order_id auf.
    order_values = df_payments.groupby('order_id')['payment_value'].sum().values
    
    # Um die Rechenzeit für das Beispiel moderat zu halten, ziehen wir eine 
    # repräsentative Stichprobe von 3.000 Bestellungen aus unseren echten Daten.
    # In der Realität würdest du oft alle Daten nehmen, wenn die Rechenleistung reicht.
    np.random.seed(42)
    sample_data = np.random.choice(order_values, size=3000, replace=False)
    
    original_mean = np.mean(sample_data)
    original_median = np.median(sample_data)
    print(f"Originaler Mittelwert der Stichprobe: {original_mean:.2f} BRL")
    print(f"Originaler Median der Stichprobe:     {original_median:.2f} BRL")

    # -------------------------------------------------------------------
    # 3. Das Bootstrapping durchführen
    # -------------------------------------------------------------------
    n_iterations = 5000  # Wir ziehen 5.000 neue Stichproben!
    n_size = len(sample_data)
    
    bootstrapped_means = np.zeros(n_iterations)
    bootstrapped_medians = np.zeros(n_iterations)
    
    print(f"\nStarte Bootstrapping mit {n_iterations} Iterationen. Das dauert einen Moment...")
    
    for i in range(n_iterations):
        # Der Kern des Bootstrappings: Ziehen MIT Zurücklegen (replace=True)
        # Wir kreieren quasi 5.000 parallele Universen unserer Daten
        bootstrap_sample = np.random.choice(sample_data, size=n_size, replace=True)
        
        # Berechne die Metrik für dieses "Paralleluniversum"
        bootstrapped_means[i] = np.mean(bootstrap_sample)
        bootstrapped_medians[i] = np.median(bootstrap_sample)

    # -------------------------------------------------------------------
    # 4. Konfidenzintervalle (CI) berechnen
    # -------------------------------------------------------------------
    # Wir sortieren unsere 5.000 Ergebnisse und schneiden die extremsten 2.5% 
    # oben und unten ab. Was übrig bleibt, ist das 95% Konfidenzintervall.
    alpha = 0.95
    lower_p = ((1.0 - alpha) / 2.0) * 100
    upper_p = (alpha + ((1.0 - alpha) / 2.0)) * 100
    
    ci_mean = np.percentile(bootstrapped_means, [lower_p, upper_p])
    ci_median = np.percentile(bootstrapped_medians, [lower_p, upper_p])
    
    print("\nErgebnisse der 95% Konfidenzintervalle:")
    print(f"Mean:   Mittelwert liegt mit 95% Wahrscheinlichkeit zwischen {ci_mean[0]:.2f} und {ci_mean[1]:.2f} BRL")
    print(f"Median: Median liegt mit 95% Wahrscheinlichkeit zwischen {ci_median[0]:.2f} und {ci_median[1]:.2f} BRL")

    # -------------------------------------------------------------------
    # 5. Wunderschöne Visualisierung
    # -------------------------------------------------------------------
    print("\nGeneriere Visualisierungs-Grid...")
    sns.set_theme(style="whitegrid")
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # --- Plot 1: Bootstrapped Mean ---
    sns.histplot(bootstrapped_means, bins=50, color='#3498db', kde=True, ax=axes[0])
    axes[0].axvline(original_mean, color='#2c3e50', linestyle='-', linewidth=2, 
                    label=f'Original Mean ({original_mean:.2f})')
    axes[0].axvline(ci_mean[0], color='#e74c3c', linestyle='--', linewidth=2, 
                    label=f'95% CI Lower ({ci_mean[0]:.2f})')
    axes[0].axvline(ci_mean[1], color='#e74c3c', linestyle='--', linewidth=2, 
                    label=f'95% CI Upper ({ci_mean[1]:.2f})')
    
    # Markiere den Bereich des Konfidenzintervalls farblich
    axes[0].axvspan(ci_mean[0], ci_mean[1], color='#e74c3c', alpha=0.1)
    
    axes[0].set_title("Bootstrapping: Verteilung des Mittelwerts", fontsize=14, fontweight="bold")
    axes[0].set_xlabel("Durchschnittlicher Bestellwert (BRL)")
    axes[0].set_ylabel("Häufigkeit in 5.000 Simulationen")
    axes[0].legend()

    # --- Plot 2: Bootstrapped Median ---
    sns.histplot(bootstrapped_medians, bins=50, color='#2ecc71', kde=False, ax=axes[1])
    axes[1].axvline(original_median, color='#2c3e50', linestyle='-', linewidth=2, 
                    label=f'Original Median ({original_median:.2f})')
    axes[1].axvline(ci_median[0], color='#e74c3c', linestyle='--', linewidth=2, 
                    label=f'95% CI Lower ({ci_median[0]:.2f})')
    axes[1].axvline(ci_median[1], color='#e74c3c', linestyle='--', linewidth=2, 
                    label=f'95% CI Upper ({ci_median[1]:.2f})')
    
    axes[1].axvspan(ci_median[0], ci_median[1], color='#e74c3c', alpha=0.1)
    
    axes[1].set_title("Bootstrapping: Verteilung des Medians", fontsize=14, fontweight="bold")
    axes[1].set_xlabel("Median Bestellwert (BRL)")
    axes[1].legend()

    # Layout optimieren und speichern
    plt.suptitle("Non-Parametrische Inferenz: Bootstrapped 95% Konfidenzintervalle", 
                 fontsize=18, fontweight="bold", y=1.05)
    plt.tight_layout()
    sns.despine()
    
    plt.savefig(GRAPHIC_PATH, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✅ Grafik erfolgreich gespeichert unter:\n{GRAPHIC_PATH}")

if __name__ == "__main__":
    main()
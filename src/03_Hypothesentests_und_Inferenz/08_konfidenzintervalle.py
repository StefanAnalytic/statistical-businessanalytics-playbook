"""
Griff 8: Ist der beobachtete Unterschied signifikant? (Konfidenzintervalle für Mittelwerte & Anteile)

Was ist das und was macht das Skript?
In diesem Skript analysieren wir E-Commerce-Zahlungen (Kreditkarte vs. Boleto/Rechnung).
Wir berechnen nicht nur den reinen Durchschnitt (Mean) und den Anteil (Proportion) an Ratenzahlungen, 
sondern auch das 95% Konfidenzintervall (CI). 

Das CI sagt uns: "Basierend auf unserer Stichprobe liegt der wahre Durchschnitt aller Kunden 
mit 95%iger Wahrscheinlichkeit genau in diesem Bereich."

Wofür ist das im Business gut?
Stell dir vor, eine A/B-Test-Variante macht im Schnitt 5€ mehr Umsatz. Ist das Zufall oder 
ein echter Business-Gewinn? Wenn das Konfidenzintervall von Variante A (10€ bis 15€) sich 
nicht mit dem von Variante B (3€ bis 7€) überschneidet, hast du einen statistisch 
signifikanten Unterschied. Du weißt also genau, wie viel Umsatz du mindestens erwarten kannst.
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
# Wir wechseln nun in den Ordner "03_Hypothesentests_und_Inferenz"!
DATA_PATH = "/Users/stefanbechthold/statistik_business_analytics/data/olist_order_payments_dataset.csv"
GRAPHIC_DIR = "/Users/stefanbechthold/statistik_business_analytics/Grafiken/03_Hypothesentests_und_Inferenz"
GRAPHIC_PATH = os.path.join(GRAPHIC_DIR, "08_konfidenzintervalle.png")

# Sicherstellen, dass der Zielordner existiert
os.makedirs(GRAPHIC_DIR, exist_ok=True)

def main():
    print("🚀 Starte Griff 8: Konfidenzintervalle (Mean & Proportion)...")
    
    # -------------------------------------------------------------------
    # 2. Daten laden & vorbereiten
    # -------------------------------------------------------------------
    print(f"Lade Daten von: {DATA_PATH}")
    df = pd.read_csv(DATA_PATH)
    
    # Wir vergleichen zwei beliebte Zahlungsarten in Brasilien:
    # 'credit_card' (Kreditkarte) und 'boleto' (eine Art Barzahlungs-Rechnung)
    df_cc = df[df['payment_type'] == 'credit_card']['payment_value'].dropna()
    df_boleto = df[df['payment_type'] == 'boleto']['payment_value'].dropna()
    
    print(f"Datenpunkte: {len(df_cc)} Kreditkarte, {len(df_boleto)} Boleto.")

    # -------------------------------------------------------------------
    # 3. Konfidenzintervall für den Mittelwert (Mean CI) berechnen
    # -------------------------------------------------------------------
    print("\nBerechne 95% Konfidenzintervalle für den Umsatz (Mean)...")
    
    def calculate_mean_ci(data, confidence=0.95):
        n = len(data)
        mean = np.mean(data)
        # Standardfehler des Mittelwerts (Standard Error of Mean, SEM)
        # Sagt uns, wie stark der Mittelwert bei neuen Stichproben schwanken würde.
        sem = stats.sem(data) 
        
        # t-Verteilung nutzen, um das Intervall zu berechnen (robust bei allen Stichprobengrößen)
        ci_lower, ci_upper = stats.t.interval(confidence, df=n-1, loc=mean, scale=sem)
        return mean, ci_lower, ci_upper

    mean_cc, ci_low_cc, ci_up_cc = calculate_mean_ci(df_cc)
    mean_bol, ci_low_bol, ci_up_bol = calculate_mean_ci(df_boleto)

    print(f"Kreditkarte: Mean = {mean_cc:.2f} BRL | 95% CI: [{ci_low_cc:.2f}, {ci_up_cc:.2f}]")
    print(f"Boleto:      Mean = {mean_bol:.2f} BRL | 95% CI: [{ci_low_bol:.2f}, {ci_up_bol:.2f}]")

    # -------------------------------------------------------------------
    # 4. Konfidenzintervall für einen Anteil (Proportion CI) berechnen
    # -------------------------------------------------------------------
    print("\nBerechne 95% Konfidenzintervalle für die Ratenzahlungs-Quote (Proportion)...")
    
    # Business-Frage: Wie viel Prozent der Kreditkarten-Zahler nutzen Ratenzahlung (>1 Rate)?
    cc_installments = df[df['payment_type'] == 'credit_card']['payment_installments']
    
    # Anzahl der Erfolge (Ratenzahlung > 1) und totale Versuche
    successes = sum(cc_installments > 1)
    n_trials = len(cc_installments)
    proportion = successes / n_trials
    
    # Standardfehler für Proportionen berechnen
    # Formel: Wurzel aus (p * (1-p) / n)
    se_prop = np.sqrt(proportion * (1 - proportion) / n_trials)
    
    # Für Proportionen bei großem n nutzen wir die Normalverteilung (z-score für 95% ist ~1.96)
    z_score = stats.norm.ppf(0.975) 
    ci_low_prop = proportion - z_score * se_prop
    ci_up_prop = proportion + z_score * se_prop
    
    print(f"Ratenzahler-Quote bei KK: {proportion*100:.2f}% | 95% CI: [{ci_low_prop*100:.2f}%, {ci_up_prop*100:.2f}%]")

    # -------------------------------------------------------------------
    # 5. Wunderschöne Visualisierung
    # -------------------------------------------------------------------
    print("\nGeneriere Visualisierungs-Grid...")
    sns.set_theme(style="whitegrid")
    
    # Wir erstellen ein Bild mit 2 Diagrammen nebeneinander (1 Reihe, 2 Spalten)
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # --- Plot 1: Mittelwerte mit Fehlerbalken (Error Bars) ---
    # Wir berechnen die Differenz vom Mean zum oberen/unteren Rand für den Plot
    error_cc = [[mean_cc - ci_low_cc], [ci_up_cc - mean_cc]]
    error_bol = [[mean_bol - ci_low_bol], [ci_up_bol - mean_bol]]
    
    x_labels = ['Kreditkarte', 'Boleto']
    means = [mean_cc, mean_bol]
    
    axes[0].bar(x_labels[0], means[0], yerr=error_cc, capsize=10, color='#3498db', alpha=0.8, edgecolor='black', linewidth=1.5)
    axes[0].bar(x_labels[1], means[1], yerr=error_bol, capsize=10, color='#e67e22', alpha=0.8, edgecolor='black', linewidth=1.5)
    
    axes[0].set_title("95% Konfidenzintervall: Durchschnittlicher Bestellwert", fontsize=14, fontweight="bold")
    axes[0].set_ylabel("Zahlungsbetrag (BRL)")
    
    # Füge den genauen Text in die Balken ein
    axes[0].text(0, 50, f"Mean: {mean_cc:.1f}\nCI: [{ci_low_cc:.1f}, {ci_up_cc:.1f}]", ha='center', color='white', fontweight='bold')
    axes[0].text(1, 50, f"Mean: {mean_bol:.1f}\nCI: [{ci_low_bol:.1f}, {ci_up_bol:.1f}]", ha='center', color='white', fontweight='bold')

    # --- Plot 2: Anteil (Proportion) mit Fehlerbalken ---
    axes[1].bar(['Ratenzahlung (>1 Rate)'], [proportion], yerr=[[proportion - ci_low_prop], [ci_up_prop - proportion]], 
                capsize=10, color='#9b59b6', alpha=0.8, edgecolor='black', linewidth=1.5, width=0.4)
    
    axes[1].set_title("95% Konfidenzintervall: Ratenzahler-Quote (Kreditkarte)", fontsize=14, fontweight="bold")
    axes[1].set_ylabel("Anteil in %")
    
    # Y-Achse als Prozent formatieren
    axes[1].set_ylim(0, 1.0)
    vals = axes[1].get_yticks()
    axes[1].set_yticklabels(['{:,.0%}'.format(x) for x in vals])
    
    axes[1].text(0, 0.2, f"Proportion: {proportion*100:.1f}%\nCI: [{ci_low_prop*100:.1f}%, {ci_up_prop*100:.1f}%]", 
                 ha='center', color='white', fontweight='bold')

    # Layout optimieren und speichern
    plt.suptitle("Statistische Sicherheit: Wahre Werte vs. Stichprobenergebnisse", 
                 fontsize=18, fontweight="bold", y=1.05)
    plt.tight_layout()
    sns.despine()
    
    plt.savefig(GRAPHIC_PATH, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✅ Grafik erfolgreich gespeichert unter:\n{GRAPHIC_PATH}")

if __name__ == "__main__":
    main()
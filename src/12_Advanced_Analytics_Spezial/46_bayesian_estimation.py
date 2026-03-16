"""
Griff 46: Unsicherheit bayesianisch darstellen? (Bayesian Estimation: Priors & Posteriors)

Was ist das und was macht das Skript?
In der klassischen (frequentistischen) Statistik verlassen wir uns oft rein auf die Daten, 
die uns vorliegen. Das führt bei kleinen Stichproben (Small-n) zu extremen Fehlentscheidungen.
Beispiel: Ein neuer Verkäufer auf dem Marktplatz hat 2 Bewertungen, beide sind 5 Sterne. 
Ist er ein perfekter Verkäufer (100% Top-Rating)? Der Frequentist sagt: Ja. 

Der Bayesianer sagt: Halt, stopp! Wir haben Vorwissen ("Prior"). Wir wissen, dass der 
durchschnittliche Olist-Verkäufer in etwa 57% 5-Sterne-Bewertungen bekommt. Wir nehmen 
dieses Vorwissen und "updaten" es mit den neuen Daten (Likelihood). Das Ergebnis ist 
der "Posterior".

Das führt zum sogenannten "Shrinkage-Effekt": Extreme Werte von kleinen Stichproben 
werden in Richtung des globalen Durchschnitts "geschrumpft". Erst wenn der Verkäufer 
hunderte Bewertungen hat, überschreiben die Daten den Prior vollständig.

Business Case in diesem Skript:
Wir bewerten die "wahren" 5-Sterne-Raten von Olist-Verkäufern. Wir nutzen Beta-Binomial 
Updating (eine analytische Form von Bayes), um zu zeigen, wie ein neuer Verkäufer mit 
nur 3 perfekten Reviews bayesianisch völlig anders bewertet wird als ein etablierter 
Verkäufer mit 300 perfekten Reviews.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import beta

# -------------------------------------------------------------------
# 1. Setup & Pfade
# -------------------------------------------------------------------
REVIEWS_PATH = "/Users/stefanbechthold/statistik_business_analytics/data/olist_order_reviews_dataset.csv"
ITEMS_PATH = "/Users/stefanbechthold/statistik_business_analytics/data/olist_order_items_dataset.csv"

GRAPHIC_DIR = "/Users/stefanbechthold/statistik_business_analytics/Grafiken/03_Hypothesentests_und_Inferenz"
GRAPHIC_PATH = os.path.join(GRAPHIC_DIR, "46_bayesian_estimation_shrinkage.png")

os.makedirs(GRAPHIC_DIR, exist_ok=True)

def main():
    print("🚀 Starte Griff 46: Bayesian Estimation (Priors, Posteriors & Shrinkage)...")
    
    # -------------------------------------------------------------------
    # 2. Daten laden & 5-Sterne Metrik (Target) bauen
    # -------------------------------------------------------------------
    df_reviews = pd.read_csv(REVIEWS_PATH)
    df_items = pd.read_csv(ITEMS_PATH)
    
    # Erste Bewertung pro Bestellung
    reviews_unique = df_reviews.drop_duplicates(subset=['order_id'])[['order_id', 'review_score']]
    
    # Verkäufer zur Bestellung mappen
    items_unique = df_items.drop_duplicates(subset=['order_id'])[['order_id', 'seller_id']]
    
    df_merged = pd.merge(reviews_unique, items_unique, on='order_id', how='inner')
    df_clean = df_merged.dropna().copy()
    
    # Target: Ist es eine 5-Sterne Bewertung? (1 = Ja, 0 = Nein)
    df_clean['is_5_star'] = (df_clean['review_score'] == 5).astype(int)

    # -------------------------------------------------------------------
    # 3. Bayesianisches Setup (Prior definieren)
    # -------------------------------------------------------------------
    # Wir berechnen den globalen Durchschnitt als unser "Vorwissen" (Prior)
    global_mean = df_clean['is_5_star'].mean()
    
    # Wir definieren einen schwachen Prior (als hätten wir 10 fiktive "durchschnittliche" Reviews gesehen)
    # Die Beta-Verteilung ist konjugiert zur Binomialverteilung (Perfekt für Ja/Nein Daten)
    prior_weight = 10
    prior_alpha = global_mean * prior_weight
    prior_beta = (1 - global_mean) * prior_weight

    # -------------------------------------------------------------------
    # 4. Posterior Berechnung (Das Update)
    # -------------------------------------------------------------------
    # Aggregation auf Verkäufer-Ebene
    seller_stats = df_clean.groupby('seller_id')['is_5_star'].agg(
        k='sum',       # Anzahl 5-Sterne Reviews (Erfolge)
        n='count'      # Gesamtzahl der Reviews (Versuche)
    ).reset_index()
    
    # 1. Frequentistischer Schätzer (Maximum Likelihood Estimate - MLE)
    # Einfach Erfolge / Versuche. (Gefährlich bei kleinen n!)
    seller_stats['MLE_Rate'] = seller_stats['k'] / seller_stats['n']
    
    # 2. Bayesianischer Schätzer (Posterior Mean)
    # Formel: (Erfolge + Prior_Alpha) / (Versuche + Prior_Alpha + Prior_Beta)
    seller_stats['Bayes_Rate'] = (seller_stats['k'] + prior_alpha) / (seller_stats['n'] + prior_alpha + prior_beta)

    # Zwei Beispiel-Verkäufer für den Verteilungs-Plot heraussuchen:
    # A) "Newcomer" mit 3 von 3 perfekten Reviews
    newcomer = seller_stats[(seller_stats['n'] == 3) & (seller_stats['k'] == 3)].iloc[0]
    
    # B) "Veteran" mit 300 Reviews und ca. 90% 5-Sterne Rate
    veteran_candidates = seller_stats[(seller_stats['n'] >= 300) & (seller_stats['MLE_Rate'] >= 0.85)]
    veteran = veteran_candidates.iloc[0] if not veteran_candidates.empty else seller_stats.iloc[0]

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
    
    x = np.linspace(0, 1, 500)

    # --- Plot 1: Die Wahrscheinlichkeitsverteilungen (Prior vs Posterior) ---
    # Prior Plotten
    y_prior = beta.pdf(x, prior_alpha, prior_beta)
    ax1.plot(x, y_prior, 'w--', lw=2, alpha=0.6, label=f'Prior (Plattform-Avg: {global_mean*100:.1f}%)')
    
    # Posterior Newcomer
    y_newcomer = beta.pdf(x, newcomer['k'] + prior_alpha, newcomer['n'] - newcomer['k'] + prior_beta)
    ax1.plot(x, y_newcomer, color='#e74c3c', lw=3, label=f'Posterior Newcomer (3 von 3 Top)')
    ax1.fill_between(x, 0, y_newcomer, color='#e74c3c', alpha=0.2)
    
    # Posterior Veteran
    y_veteran = beta.pdf(x, veteran['k'] + prior_alpha, veteran['n'] - veteran['k'] + prior_beta)
    ax1.plot(x, y_veteran, color='#2ecc71', lw=3, label=f'Posterior Veteran ({veteran["k"]}/{veteran["n"]} Top)')
    ax1.fill_between(x, 0, y_veteran, color='#2ecc71', alpha=0.2)

    ax1.set_title("1. Die Bayesianische Update-Logik", fontsize=14, fontweight='bold', color='white')
    ax1.set_xlabel("Wahre 5-Sterne Wahrscheinlichkeit")
    ax1.set_ylabel("Dichte (Sicherheit)")
    ax1.set_xlim(0, 1)
    
    legend1 = ax1.legend(loc='upper left', facecolor='#1e1e1e', edgecolor='#333333')
    plt.setp(legend1.get_texts(), color='white')

    # Info-Box Plot 1
    info_1 = (
        "Warum der Newcomer nicht 100% bekommt:\n"
        "Obwohl der Newcomer (rot) 3 von 3 5-Sterne-Reviews hat,\n"
        "sagt der Posterior: 'Wir sind uns super unsicher (breite Kurve),\n"
        "die Wahrheit liegt eher bei 70% als bei 100%.'\n"
        "Der Veteran (grün) hat extrem viele Daten. Die Kurve\n"
        "ist messerscharf: Die Daten haben den Prior völlig überschrieben."
    )
    props1 = dict(boxstyle='round,pad=1', facecolor='#1e1e1e', alpha=0.9, edgecolor='#e74c3c', linewidth=1.5)
    ax1.text(0.02, 0.45, info_1, transform=ax1.transAxes, fontsize=11,
             verticalalignment='center', bbox=props1, color='white')

    # --- Plot 2: Der Shrinkage Effekt (Scatterplot) ---
    # Nur Verkäufer mit n < 50 plotten, da hier der Effekt am stärksten ist
    plot_data = seller_stats[seller_stats['n'] <= 50]
    
    sns.scatterplot(x='n', y='MLE_Rate', data=plot_data, color='#3498db', alpha=0.4, 
                    label='Naiv (Frequentist)', s=30, edgecolor='none', ax=ax2)
    sns.scatterplot(x='n', y='Bayes_Rate', data=plot_data, color='#f1c40f', alpha=0.8, 
                    label='Bayesian (Posterior)', s=30, edgecolor='none', ax=ax2)
    
    # Linie für den Prior (Global Mean)
    ax2.axhline(global_mean, color='white', linestyle='--', lw=2, alpha=0.7, label='Prior (Global Mean)')
    
    ax2.set_title("2. Der Shrinkage-Effekt (Small-n Experimente)", fontsize=14, fontweight='bold', color='white')
    ax2.set_xlabel("Anzahl der Reviews pro Verkäufer")
    ax2.set_ylabel("Geschätzte 5-Sterne Rate")
    
    legend2 = ax2.legend(loc='lower right', facecolor='#1e1e1e', edgecolor='#333333')
    plt.setp(legend2.get_texts(), color='white')

    # Info-Box Plot 2
    info_2 = (
        "Business Impact (Ranking & Sortierung):\n"
        "---------------------------------------\n"
        "Links im Bild (n=1 bis 5) sehen wir massive Unterschiede.\n"
        "Die naiven blauen Punkte liegen oft bei 1.0 (100%) oder 0.0 (0%).\n"
        "Die bayesianischen gelben Punkte werden wie von einem\n"
        "Magneten zur Mitte (Prior) gezogen ('Shrinkage').\n\n"
        "So verhindert Bayes, dass ein Verkäufer mit 1 Review\n"
        "plötzlich im Marktplatz-Ranking über einem Verkäufer\n"
        "mit 500 fast perfekten Reviews steht!"
    )
    props2 = dict(boxstyle='round,pad=1', facecolor='#1e1e1e', alpha=0.9, edgecolor='#f1c40f', linewidth=1.5)
    ax2.text(0.4, 0.25, info_2, transform=ax2.transAxes, fontsize=11,
             verticalalignment='center', bbox=props2, color='white')

    plt.suptitle("Bayesian Inference: Wahrscheinlichkeitsaussagen unter Unsicherheit", 
                 fontsize=18, fontweight='bold', color='white', y=1.05)
    plt.tight_layout()
    
    # Speichern der Grafik
    plt.savefig(GRAPHIC_PATH, dpi=300, bbox_inches='tight', facecolor=fig.get_facecolor())
    plt.close()
    
    print(f"✅ Grafik erfolgreich gespeichert unter:\n{GRAPHIC_PATH}")

if __name__ == "__main__":
    main()
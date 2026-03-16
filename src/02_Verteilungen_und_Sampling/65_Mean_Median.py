import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

# -------------------------------------------------------------------
# 1. Setup & Pfade
# -------------------------------------------------------------------
GRAPHIC_DIR = "/Users/stefanbechthold/statistik_business_analytics/Grafiken/01_Deskriptive_Statistik_und_EDA"
GRAPHIC_PATH = os.path.join(GRAPHIC_DIR, "65_mean_vs_median_4_plots.png")

os.makedirs(GRAPHIC_DIR, exist_ok=True)

def plot_distribution(ax, data, title, description, is_skewed):
    # Metriken berechnen
    median_val = np.median(data)
    mean_val = np.mean(data)
    skew_val = stats.skew(data)
    
    # Flächen-Berechnung (Prozentzahl der Datenpunkte links vom Mean)
    area_left_mean = np.sum(data <= mean_val) / len(data)
    
    # Fehler berechnen (L1 Norm / Mean Absolute Error)
    mae_median = np.mean(np.abs(data - median_val))
    mae_mean = np.mean(np.abs(data - mean_val))
    
    # KDE Plot erzeugen
    kde = stats.gaussian_kde(data)
    x = np.linspace(np.min(data), np.max(data), 1000)
    y = kde(x)
    
    ax.plot(x, y, color='white', lw=2)
    
    # Flächen füllen für den Median (Exakt 50/50)
    ax.fill_between(x, y, where=(x <= median_val), color='#2ecc71', alpha=0.5)
    ax.fill_between(x, y, where=(x > median_val), color='#27ae60', alpha=0.2)
    
    # Linien für Mean und Median
    ax.axvline(median_val, color='#2ecc71', linestyle='-', lw=3, label=f'Median: {median_val:.1f}')
    ax.axvline(mean_val, color='#e74c3c', linestyle='--', lw=3, label=f'Mean: {mean_val:.1f}')
    
    ax.set_title(f"{title} | Skewness: {skew_val:.2f}", fontsize=13, fontweight='bold', color='white')
    ax.set_ylabel("Dichte (Fläche = 100%)")
    
    legend = ax.legend(loc='upper right', facecolor='#1e1e1e', edgecolor='#333333')
    plt.setp(legend.get_texts(), color='white')
    
    # Mathe-Info-Box mit den Beweisen aus deinem Text
    info = (
        f"{description}\n\n"
        f"Flächen-Beweis:\n"
        f"Fläche LINKS vom Mean: {area_left_mean*100:.1f}%\n\n"
        f"Mathe-Beweis (L1 Fehler):\n"
        f"Fehler Median: {mae_median:.1f}\n"
        f"Fehler Mean: {mae_mean:.1f}"
    )
    props = dict(boxstyle='round,pad=1', facecolor='#1e1e1e', alpha=0.9, edgecolor='#e74c3c' if is_skewed else '#3498db', linewidth=1.5)
    ax.text(0.35, 0.45, info, transform=ax.transAxes, fontsize=10, verticalalignment='center', bbox=props, color='white')

def main():
    print("🚀 Starte Griff 65: Mean vs. Median (4 Stufen der Schiefe)...")
    np.random.seed(42)
    
    # -------------------------------------------------------------------
    # 2. Daten simulieren (4 Eskalations-Stufen)
    # -------------------------------------------------------------------
    # A) Skew ~ 0 (Symmetrisch - z.B. Produktions-Toleranzen)
    data_0 = np.random.normal(loc=15.0, scale=3.0, size=10000)
    
    # B) Skew ~ 0.5 (Leicht schief - Grauzone)
    data_05 = np.random.lognormal(mean=2.7, sigma=0.4, size=10000)
    
    # C) Skew ~ 1.5 (Stark schief - z.B. Umsatz, Lognormal aus deinem Skript)
    data_15 = np.random.lognormal(mean=4.0, sigma=1.0, size=10000)
    data_15 = data_15[data_15 < 800] # Extreme kappen für Plot
    
    # D) Skew > 3.0 (Extrem schief - Whale-Kunden)
    data_30 = np.random.lognormal(mean=2.0, sigma=1.5, size=10000)
    data_30 = data_30[data_30 < np.percentile(data_30, 98)]

    # -------------------------------------------------------------------
    # 3. Visualisierung im Dark Mode Design
    # -------------------------------------------------------------------
    plt.style.use('dark_background')
    sns.set_theme(style="darkgrid", rc={
        "axes.facecolor": "#121212", "figure.facecolor": "#121212", 
        "text.color": "white", "axes.labelcolor": "white", 
        "xtick.color": "white", "ytick.color": "white", "grid.color": "#2c3e50"
    })
    
    fig, axes = plt.subplots(2, 2, figsize=(20, 16), facecolor='#121212')
    
    # Plots generieren
    desc_0 = "Symmetrisch: Mean und Median fallen zusammen.\nNutze den MEAN (minimiert L2 Varianz)."
    plot_distribution(axes[0, 0], data_0, "1. Symmetrisch (Paketgewicht)", desc_0, False)
    
    desc_05 = "Grenzwertig: Mean zieht leicht nach rechts.\nMean oft noch okay, aber Vorsicht."
    plot_distribution(axes[0, 1], data_05, "2. Leicht schief (Grauzone)", desc_05, False)
    
    desc_15 = "Schief: Mean liegt in der oberen Hälfte.\nÜber 65% sind 'unterdurchschnittlich'.\nNutze MEDIAN!"
    plot_distribution(axes[1, 0], data_15, "3. Stark schief (Umsatz)", desc_15, True)
    
    desc_30 = "Extrem schief: Ausreißer zerstören den Mean.\nFast 80% sind 'unterdurchschnittlich'.\nMEDIAN ist Pflicht!"
    plot_distribution(axes[1, 1], data_30, "4. Extrem schief (Whales)", desc_30, True)
    
    plt.suptitle("Deskriptive Mathematik: Ab welcher Skewness kippt die Wahl zu Median?", 
                 fontsize=22, fontweight='bold', color='white', y=1.02)
    plt.tight_layout()
    
    # Speichern der Grafik
    plt.savefig(GRAPHIC_PATH, dpi=300, bbox_inches='tight', facecolor=fig.get_facecolor())
    plt.close()
    
    print(f"✅ Grafik erfolgreich gespeichert unter:\n{GRAPHIC_PATH}")

if __name__ == "__main__":
    main()
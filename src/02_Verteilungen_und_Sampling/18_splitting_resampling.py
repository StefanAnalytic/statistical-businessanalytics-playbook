"""
Griff 18: Wie splitte & resample ich korrekt? (Sampling: Train/Test, Stratified, TimeSeriesSplit)

Was ist das und was macht das Skript?
Bevor ein Machine-Learning-Modell in Produktion geht, muss es validiert werden. 
Wir dürfen es niemals auf den Daten testen, auf denen es trainiert wurde (sonst lügt es uns 
mit perfekten Ergebnissen an - das sogenannte "Data Leakage"). Wir müssen die Daten aufteilen (Splitting).

Aber ein einfacher Zufalls-Split (Random Split) reicht oft nicht!
1. Stratified Split: Wenn unsere Zielgröße extrem ungleich verteilt ist (z. B. 95% pünktlich, 
   nur 5% verspätet), kann ein reiner Zufallssplit dazu führen, dass im Testset plötzlich 
   0% verspätete Lieferungen landen. "Stratified" garantiert, dass das exakte 95/5-Verhältnis 
   in Train- und Testset erhalten bleibt.
2. TimeSeries Split: E-Commerce-Daten sind zeitabhängig (Trends, Saisonalität, Black Friday).
   Wenn wir Bestellungen aus dem Dezember nutzen, um den November vorherzusagen, schummeln wir! 
   Zeitreihen-Splits trainieren immer nur auf der Vergangenheit, um die Zukunft vorherzusagen.

Business Case in diesem Skript:
Wir nutzen die Bestelldaten ('olist_orders_dataset.csv') und bauen einen Vorhersage-Usecase 
für "Verspätete Lieferungen" (is_late). Da Verspätungen (zum Glück) selten sind, haben wir ein 
"Imbalanced Class" Problem. Wir zeigen visuell, warum Random-Splits hier gefährlich sind und 
wie ein Zeitreihen-Split unsere Daten respektiert.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.patches import Patch

from sklearn.model_selection import train_test_split, TimeSeriesSplit

# -------------------------------------------------------------------
# 1. Setup & Pfade
# -------------------------------------------------------------------
ORDERS_PATH = "/Users/stefanbechthold/statistik_business_analytics/data/olist_orders_dataset.csv"

# Wir bleiben im Validierungs-Ordner
GRAPHIC_DIR = "/Users/stefanbechthold/statistik_business_analytics/Grafiken/06_Klassifikation_und_Validierung"
GRAPHIC_PATH = os.path.join(GRAPHIC_DIR, "18_daten_splits_validierung.png")

os.makedirs(GRAPHIC_DIR, exist_ok=True)

def main():
    print("🚀 Starte Griff 18: Daten richtig aufteilen (Splitting & Resampling)...")
    
    # -------------------------------------------------------------------
    # 2. Daten laden & zeitlich sortieren
    # -------------------------------------------------------------------
    print(f"Lade Bestelldaten von: {ORDERS_PATH}")
    df_orders = pd.read_csv(ORDERS_PATH)
    
    # Zeitstempel konvertieren
    df_orders['purchase_time'] = pd.to_datetime(df_orders['order_purchase_timestamp'], errors='coerce')
    df_orders['delivered_time'] = pd.to_datetime(df_orders['order_delivered_customer_date'], errors='coerce')
    df_orders['estimated_time'] = pd.to_datetime(df_orders['order_estimated_delivery_date'], errors='coerce')
    
    # Für saubere Zeitreihen-Splits MÜSSEN die Daten zwingend chronologisch sortiert sein!
    df_clean = df_orders.dropna(subset=['purchase_time', 'delivered_time', 'estimated_time']).copy()
    df_clean = df_clean.sort_values('purchase_time').reset_index(drop=True)
    
    # Zielvariable: War die Lieferung verspätet? (1 = Ja, 0 = Nein)
    df_clean['is_late'] = (df_clean['delivered_time'] > df_clean['estimated_time']).astype(int)
    
    # Wir nehmen ein Sample von 15.000 chronologischen Bestellungen für die Visualisierung
    df_sample = df_clean.iloc[:15000].copy()
    
    total_late_rate = df_sample['is_late'].mean()
    print(f"\nAnalysiere {len(df_sample)} Bestellungen.")
    print(f"Echte Verspätungsquote (Grundwahrheit): {total_late_rate*100:.2f}%")

    # -------------------------------------------------------------------
    # 3. Splitting-Strategien anwenden
    # -------------------------------------------------------------------
    X = df_sample.drop(columns=['is_late'])
    y = df_sample['is_late']

    print("\nFühre verschiedene Splits durch...")
    
    # A) Random Split (Gefährlich bei seltenen Ereignissen!)
    # Wir machen einen extremen 90/10 Split, um das Problem künstlich zu provozieren
    _, _, y_train_rand, y_test_rand = train_test_split(X, y, test_size=0.10, random_state=42)
    
    # B) Stratified Split (Der Goldstandard für ungleiche Klassen)
    _, _, y_train_strat, y_test_strat = train_test_split(X, y, test_size=0.10, random_state=42, stratify=y)
    
    # Quoten berechnen
    rate_rand_train = y_train_rand.mean()
    rate_rand_test = y_test_rand.mean()
    rate_strat_train = y_train_strat.mean()
    rate_strat_test = y_test_strat.mean()

    # C) TimeSeries Split (Der Goldstandard für historische Daten)
    # Erstellt wachsende Trainings-Fenster und testet immer auf der direkten Zukunft
    tscv = TimeSeriesSplit(n_splits=4)

    # -------------------------------------------------------------------
    # 4. Wunderschöne Visualisierung (3-teiliges Dashboard)
    # -------------------------------------------------------------------
    print("Generiere Visualisierungs-Grid...")
    sns.set_theme(style="whitegrid")
    fig = plt.figure(figsize=(18, 10))
    
    # Layout-Grid
    ax_bar = plt.subplot2grid((2, 2), (0, 0)) # Oben links
    ax_strat = plt.subplot2grid((2, 2), (1, 0)) # Unten links
    ax_time = plt.subplot2grid((2, 2), (0, 1), rowspan=2) # Rechts (Groß)

    # --- Plot 1: Grundverteilung (Warum Stratifizieren?) ---
    sns.countplot(x='is_late', data=df_sample, ax=ax_bar, palette=['#2ecc71', '#e74c3c'])
    ax_bar.set_title("1. Problem: Ungleiche Klassen (Imbalanced Data)", fontsize=14, fontweight="bold")
    ax_bar.set_xticklabels(['Pünktlich (0)', 'Verspätet (1)'])
    ax_bar.set_xlabel("")
    ax_bar.set_ylabel("Anzahl der Bestellungen")
    
    # Text-Einschub
    ax_bar.text(0.5, 0.5, f"{total_late_rate*100:.1f}% Verspätet", transform=ax_bar.transAxes, 
                ha='center', va='center', fontsize=16, fontweight='bold', color='#c0392b',
                bbox=dict(facecolor='white', alpha=0.8, edgecolor='#e74c3c', boxstyle='round,pad=0.5'))

    # --- Plot 2: Random vs. Stratified Split ---
    labels = ['Train (Random)', 'Test (Random)', 'Train (Stratified)', 'Test (Stratified)']
    rates = [rate_rand_train*100, rate_rand_test*100, rate_strat_train*100, rate_strat_test*100]
    colors = ['#34495e', '#7f8c8d', '#2980b9', '#3498db']
    
    bars = ax_strat.bar(labels, rates, color=colors, edgecolor='black', alpha=0.9)
    ax_strat.axhline(total_late_rate*100, color='#e74c3c', linestyle='--', linewidth=2, label=f'Ideale Quote ({total_late_rate*100:.1f}%)')
    
    ax_strat.set_title("2. Lösung für Imbalance: Stratified Split", fontsize=14, fontweight="bold")
    ax_strat.set_ylabel("Verspätungsquote im Datensatz (%)")
    ax_strat.tick_params(axis='x', rotation=15)
    ax_strat.legend()
    
    # Prozentzahlen auf die Balken
    for bar in bars:
        ax_strat.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.2,
                      f"{bar.get_height():.2f}%", ha='center', va='bottom', fontweight='bold', fontsize=11)

    # --- Plot 3: TimeSeries Split (Cross-Validation) ---
    # Wir visualisieren, wie der Algorithmus durch die Zeit wandert.
    ax_time.set_title("3. Lösung für Zeitreihen: TimeSeries Split (Cross-Validation)", fontsize=14, fontweight="bold")
    
    # Ein kleines Dummy-Dataset für die visuelle Darstellung der Splits
    n_samples = len(df_sample)
    
    for i, (train_index, test_index) in enumerate(tscv.split(X)):
        # Y-Achse ist der jeweilige "Split-Durchlauf"
        # Train-Daten in Blau
        ax_time.scatter(train_index, [i+1] * len(train_index), c='#2980b9', marker='s', s=15, alpha=0.6)
        # Test-Daten in Rot (IMMER nach den Train-Daten!)
        ax_time.scatter(test_index, [i+1] * len(test_index), c='#e74c3c', marker='s', s=15, alpha=0.9)
        
    ax_time.set_yticks(np.arange(1, tscv.n_splits + 1))
    ax_time.set_yticklabels([f"CV Durchlauf {i}" for i in range(1, tscv.n_splits + 1)])
    ax_time.set_xlabel("Chronologischer Zeitverlauf (Bestell-Index)")
    ax_time.invert_yaxis() # Durchlauf 1 oben
    
    # Eigene Legende für den TimeSeries Split bauen
    legend_elements = [Patch(facecolor='#2980b9', edgecolor='black', label='Trainings-Daten (Vergangenheit)'),
                       Patch(facecolor='#e74c3c', edgecolor='black', label='Test-Daten (Zukunft)')]
    ax_time.legend(handles=legend_elements, loc='lower right')
    
    # Info-Box für TimeSeries
    info_ts = (
        "Regel bei Zeitreihen:\n"
        "Niemals die Zukunft nutzen,\n"
        "um die Vergangenheit zu erklären!\n"
        "Das Testset liegt chronologisch\n"
        "immer NACH dem Trainingsset."
    )
    props = dict(boxstyle='round,pad=0.8', facecolor='white', alpha=0.9, edgecolor='gray', linewidth=1.5)
    ax_time.text(0.05, 0.05, info_ts, transform=ax_time.transAxes, fontsize=11,
                 verticalalignment='bottom', horizontalalignment='left', bbox=props, color='#2c3e50')

    # Layout optimieren und speichern
    plt.suptitle("Machine Learning Validierung: Daten korrekt aufteilen (Splitting)", 
                 fontsize=18, fontweight="bold", y=1.05)
    plt.tight_layout()
    sns.despine()
    
    plt.savefig(GRAPHIC_PATH, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✅ Grafik erfolgreich gespeichert unter:\n{GRAPHIC_PATH}")

if __name__ == "__main__":
    main()
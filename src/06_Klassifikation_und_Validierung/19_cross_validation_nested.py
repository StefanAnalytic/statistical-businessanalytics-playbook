"""
Griff 19: Saubere Schätzung der Modell-Leistung (Cross-Validation & Nested CV)

Was ist das und was macht das Skript?
Ein einfacher Train/Test-Split (80/20) reicht oft nicht aus. Was, wenn zufällig alle 
"schwierigen" Kunden im Test-Set landen? Dann sieht unser Modell künstlich schlecht aus. 
Oder umgekehrt: Es sieht künstlich gut aus ("Optimistic Bias").

Lösung 1: K-Fold Cross-Validation (Kreuzvalidierung)
Wir teilen die Daten z.B. in 5 gleich große Blöcke (Folds). Das Modell wird 5-mal trainiert, 
wobei jedes Mal ein anderer Block als Test-Set dient. So wird jeder Datenpunkt exakt 
einmal fair getestet!

Lösung 2: Nested CV (Verschachtelte Kreuzvalidierung) - Der absolute Goldstandard!
Wenn wir ein Modell "tunen" (z.B. die optimale Baumtiefe in einem Random Forest suchen), 
nutzen wir oft Cross-Validation, um die besten Einstellungen zu finden. Wenn wir DIESELBEN 
Ergebnisse dann als finale Leistungsbewertung nehmen, lügen wir uns in die Tasche, 
weil das Modell genau auf diese Daten optimiert wurde. 
Nested CV nutzt eine "innere" Schleife zum Tunen und eine strikt getrennte "äußere" 
Schleife zur ehrlichen Bewertung.

Business Case in diesem Skript:
Wir trainieren einen Random Forest, der vorhersagt, ob ein Kunde eine 5-Sterne-Bewertung 
gibt. Wir vergleichen die naive Schätzung mit der realistischen, harten Schätzung durch Nested CV.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score, KFold, GridSearchCV
from sklearn.metrics import make_scorer, accuracy_score

# -------------------------------------------------------------------
# 1. Setup & Pfade
# -------------------------------------------------------------------
# Wir sind im Ordner "06_Klassifikation_und_Validierung"
ORDERS_PATH = "/Users/stefanbechthold/statistik_business_analytics/data/olist_orders_dataset.csv"
ITEMS_PATH = "/Users/stefanbechthold/statistik_business_analytics/data/olist_order_items_dataset.csv"
REVIEWS_PATH = "/Users/stefanbechthold/statistik_business_analytics/data/olist_order_reviews_dataset.csv"

GRAPHIC_DIR = "/Users/stefanbechthold/statistik_business_analytics/Grafiken/06_Klassifikation_und_Validierung"
GRAPHIC_PATH = os.path.join(GRAPHIC_DIR, "19_cross_validation_nested.png")

os.makedirs(GRAPHIC_DIR, exist_ok=True)

def main():
    print("🚀 Starte Griff 19: Cross-Validation & Nested CV...")
    
    # -------------------------------------------------------------------
    # 2. Daten laden & Feature Engineering (Kompakt)
    # -------------------------------------------------------------------
    print("Lade Bestelldaten und baue Features...")
    df_orders = pd.read_csv(ORDERS_PATH)
    df_items = pd.read_csv(ITEMS_PATH)
    df_reviews = pd.read_csv(REVIEWS_PATH)
    
    # Lieferzeit
    df_orders['purchase_time'] = pd.to_datetime(df_orders['order_purchase_timestamp'], errors='coerce')
    df_orders['delivery_time'] = pd.to_datetime(df_orders['order_delivered_customer_date'], errors='coerce')
    df_orders['delivery_days'] = (df_orders['delivery_time'] - df_orders['purchase_time']).dt.total_seconds() / (24 * 3600)
    
    # Preise und Bewertungen mergen
    order_costs = df_items.groupby('order_id')[['price', 'freight_value']].sum().reset_index()
    order_reviews = df_reviews.drop_duplicates(subset=['order_id'])[['order_id', 'review_score']]
    
    df_merged = pd.merge(df_orders[['order_id', 'delivery_days']], order_costs, on='order_id', how='inner')
    df_merged = pd.merge(df_merged, order_reviews, on='order_id', how='inner')
    
    # Bereinigung
    df_clean = df_merged.dropna().copy()
    df_clean = df_clean[(df_clean['delivery_days'] > 0) & (df_clean['delivery_days'] < 60)]
    df_clean['is_5_star'] = (df_clean['review_score'] == 5).astype(int)
    
    # Nested CV ist SEHR rechenintensiv (Modell wird dutzende Male trainiert).
    # Wir ziehen ein kleines, repräsentatives Sample von 2.000 Zeilen, damit das Skript 
    # in Sekunden statt in Stunden durchläuft.
    df_sample = df_clean.sample(n=2000, random_state=42)
    
    X = df_sample[['price', 'freight_value', 'delivery_days']]
    y = df_sample['is_5_star']
    
    print(f"Datensatz bereit: {len(X)} Zeilen.")

    # -------------------------------------------------------------------
    # 3. Cross-Validation durchführen (Standard vs. Nested)
    # -------------------------------------------------------------------
    print("\nStarte Trainings-Schleifen (Das kann ein paar Sekunden dauern)...")
    
    # Wir nutzen einen Random Forest Klassifikator
    rf = RandomForestClassifier(random_state=42)
    
    # Wir definieren die äußere und innere Schleife (je 5 Folds)
    outer_cv = KFold(n_splits=5, shuffle=True, random_state=42)
    inner_cv = KFold(n_splits=3, shuffle=True, random_state=42)
    
    # Wir wollen diese "Hyperparameter" (Einstellungen des Modells) optimieren:
    param_grid = {
        'n_estimators': [50, 100],      # Anzahl der Bäume
        'max_depth': [None, 5, 10],     # Wie tief darf der Baum wachsen?
        'min_samples_split': [2, 10]    # Ab wann darf sich ein Ast weiter teilen?
    }

    # --- Methode A: Standard Cross-Validation (Ohne Tuning) ---
    # Nimmt einfach die Standard-Einstellungen (Default) des Random Forests
    print("1/3 Berechne Standard 5-Fold CV...")
    standard_cv_scores = cross_val_score(rf, X, y, cv=outer_cv, scoring='accuracy')

    # --- Methode B: Non-Nested CV (Klassisches Tuning mit Data Leakage Risiko) ---
    # Wir nutzen GridSearchCV, um die besten Einstellungen auf dem GANZEN Datensatz zu finden.
    # Die gemeldete beste Leistung ist oft zu optimistisch (Optimistic Bias).
    print("2/3 Berechne Non-Nested CV (GridSearch)...")
    clf_tuned = GridSearchCV(estimator=rf, param_grid=param_grid, cv=outer_cv)
    clf_tuned.fit(X, y)
    # Wir nehmen die besten Scores aus jedem Split der GridSearch
    non_nested_scores = []
    for i in range(outer_cv.n_splits):
        # Extrahiere die Scores für den besten Parameter-Satz in jedem Split
        best_index = clf_tuned.best_index_
        score = clf_tuned.cv_results_[f'split{i}_test_score'][best_index]
        non_nested_scores.append(score)

    # --- Methode C: Nested CV (Ehrliche Schätzung für getunte Modelle) ---
    # Eine GridSearch (mit inner_cv) steckt INSIDE einer cross_val_score (outer_cv).
    # Das bedeutet: Für jeden der 5 äußeren Splits wird eine komplette GridSearch auf 
    # den restlichen 4 Splits gemacht. Dann wird das Gewinner-Modell auf dem 1 absolut 
    # ungesehenen äußeren Split getestet.
    print("3/3 Berechne Nested CV (Ehrliche Validierung)...")
    clf_nested = GridSearchCV(estimator=rf, param_grid=param_grid, cv=inner_cv)
    nested_cv_scores = cross_val_score(clf_nested, X, y, cv=outer_cv, scoring='accuracy')

    # -------------------------------------------------------------------
    # 4. Ergebnisse auswerten und visualisieren
    # -------------------------------------------------------------------
    print("\nErgebnisse (Durchschnittliche Accuracy):")
    print(f"Standard CV (Default RF):  {np.mean(standard_cv_scores):.4f}")
    print(f"Non-Nested CV (Optimiert): {np.mean(non_nested_scores):.4f} <- Oft zu optimistisch!")
    print(f"Nested CV (Ehrlicher Wert):{np.mean(nested_cv_scores):.4f} <- Die reale Erwartung!")

    print("\nGeneriere Visualisierung (Boxplot der Scores)...")
    sns.set_theme(style="whitegrid")
    
    fig, ax = plt.subplots(figsize=(12, 7))

    # Daten für Seaborn vorbereiten
    plot_data = pd.DataFrame({
        'Validierungs-Methode': ['1. Standard CV (Defaults)'] * 5 + 
                                ['2. Non-Nested (Optimistisch)'] * 5 + 
                                ['3. Nested CV (Ehrlich)'] * 5,
        'Accuracy': np.concatenate([standard_cv_scores, non_nested_scores, nested_cv_scores])
    })

    # Boxplot + Swarmplot (zeigt die einzelnen 5 Folds als Punkte)
    sns.boxplot(x='Validierungs-Methode', y='Accuracy', data=plot_data, 
                palette=['#95a5a6', '#e74c3c', '#2ecc71'], ax=ax, width=0.5, boxprops=dict(alpha=0.7))
    
    sns.swarmplot(x='Validierungs-Methode', y='Accuracy', data=plot_data, 
                  color='black', alpha=0.8, size=8, ax=ax)

    # Durchschnitte als rote Rauten
    means = [np.mean(standard_cv_scores), np.mean(non_nested_scores), np.mean(nested_cv_scores)]
    ax.scatter([0, 1, 2], means, color='red', marker='D', s=100, label='Mittelwert (Expected Performance)', zorder=5)

    ax.set_title("Modell-Validierung: Kampf gegen den Optimismus-Bias (Overfitting)", fontsize=16, fontweight="bold")
    ax.set_ylabel("Modell-Genauigkeit (Accuracy)", fontsize=12)
    ax.set_xlabel("")
    
    # Y-Achse als Prozent formatieren
    vals = ax.get_yticks()
    ax.set_yticklabels(['{:,.1%}'.format(x) for x in vals])
    ax.legend()

    # Info-Box erklären
    info_text = (
        "Warum Nested CV?\n"
        "Wenn wir Parameter (wie die Baumtiefe) optimieren,\n"
        "merkt sich das Modell indirekt Eigenschaften der Testdaten.\n"
        "Die rote Box (Non-Nested) zeigt oft eine höhere Leistung,\n"
        "die in der echten Welt (Produktion) aber sofort einbricht.\n"
        "Die grüne Box (Nested) trennt Tuning und Testing strikt\n"
        "und liefert die einzig ehrliche Performance-Schätzung."
    )
    props = dict(boxstyle='round,pad=0.8', facecolor='white', alpha=0.9, edgecolor='gray', linewidth=1.5)
    ax.text(0.03, 0.05, info_text, transform=ax.transAxes, fontsize=11,
            verticalalignment='bottom', horizontalalignment='left', bbox=props, color='#2c3e50')

    # Layout optimieren und speichern
    plt.tight_layout()
    sns.despine()
    
    plt.savefig(GRAPHIC_PATH, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✅ Grafik erfolgreich gespeichert unter:\n{GRAPHIC_PATH}")

if __name__ == "__main__":
    main()
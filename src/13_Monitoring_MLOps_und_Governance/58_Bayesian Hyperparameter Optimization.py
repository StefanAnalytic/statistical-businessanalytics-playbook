"""
Griff 58: Modelle intelligent tunen? (Bayesian Hyperparameter Optimization)

Was ist das und was macht das Skript?
Jedes Machine-Learning-Modell hat "Knöpfe und Regler" (Hyperparameter), die wir vor 
dem Training einstellen müssen (z.B. Wie tief darf ein Baum wachsen? Wie viele Bäume?). 
Die klassische Methode ist "Grid Search": Wir probieren einfach stumpf alle Kombinationen aus. 
Das Problem: Bei 5 Parametern und 10 Werten sind das 100.000 Kombinationen. Das dauert Wochen!

Die moderne Lösung: Bayesian Optimization (hier mit dem Framework 'Optuna').
Anstatt blind zu suchen, lernt der Algorithmus aus vergangenen Versuchen (Trials). 
Wenn er merkt, dass "Tiefe Bäume" immer zu schlechten Ergebnissen führen, hört er auf, 
diesen Bereich zu testen, und fokussiert seine Rechenpower auf vielversprechende Zonen.
Das ist "Directed Search" statt "Brute Force".

Business Case in diesem Skript:
Wir wollen eine Klassifikation bauen (Kommt das Paket zu spät?). Wir nutzen einen 
Random Forest. Anstatt die Parameter zu raten, lassen wir Optuna in 30 gezielten 
Versuchen das perfekte Setup für unseren spezifischen Datensatz finden. Danach 
analysieren wir, welcher "Regler" eigentlich den größten Hebel auf die Performance hatte.

WICHTIG: Benötigt das Paket 'optuna' -> pip install optuna
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import roc_auc_score

try:
    import optuna
    # Wir schalten den exzessiven Optuna-Konsolen-Output für das Skript ab
    optuna.logging.set_verbosity(optuna.logging.WARNING)
except ImportError:
    print("FEHLER: Die Bibliothek 'optuna' fehlt. Bitte ausführen: pip install optuna")
    exit()

# -------------------------------------------------------------------
# 1. Setup & Pfade
# -------------------------------------------------------------------
ORDERS_PATH = "/Users/stefanbechthold/statistik_business_analytics/data/olist_orders_dataset.csv"
ITEMS_PATH = "/Users/stefanbechthold/statistik_business_analytics/data/olist_order_items_dataset.csv"
PRODUCTS_PATH = "/Users/stefanbechthold/statistik_business_analytics/data/olist_products_dataset.csv"

GRAPHIC_DIR = "/Users/stefanbechthold/statistik_business_analytics/Grafiken/06_Klassifikation_und_Validierung"
GRAPHIC_PATH = os.path.join(GRAPHIC_DIR, "58_optuna_bayesian_optimization.png")

os.makedirs(GRAPHIC_DIR, exist_ok=True)

def main():
    print("🚀 Starte Griff 58: Bayesian Hyperparameter Optimization mit Optuna...")
    
    # -------------------------------------------------------------------
    # 2. Daten laden & Feature Engineering
    # -------------------------------------------------------------------
    df_orders = pd.read_csv(ORDERS_PATH)
    df_items = pd.read_csv(ITEMS_PATH)
    df_products = pd.read_csv(PRODUCTS_PATH)
    
    # Target: Verspätete Lieferung? (1 = Ja, 0 = Nein)
    df_orders['delivered'] = pd.to_datetime(df_orders['order_delivered_customer_date'], errors='coerce')
    df_orders['estimated'] = pd.to_datetime(df_orders['order_estimated_delivery_date'], errors='coerce')
    df_orders['is_late'] = (df_orders['delivered'] > df_orders['estimated']).astype(int)
    
    # Produktdaten mergen
    df_items = pd.merge(df_items, df_products[['product_id', 'product_weight_g', 'product_length_cm']], on='product_id', how='left')
    
    # Aggregation auf Order-Ebene
    order_features = df_items.groupby('order_id').agg({
        'price': 'sum',
        'freight_value': 'sum',
        'product_weight_g': 'sum',
        'order_item_id': 'max' # Anzahl Items
    }).reset_index()
    
    df_merged = pd.merge(df_orders[['order_id', 'is_late']], order_features, on='order_id', how='inner')
    df_clean = df_merged.dropna().copy()
    
    # Sample ziehen für annehmbare Rechenzeit im Skript
    df_sample = df_clean.sample(n=10000, random_state=42).reset_index(drop=True)
    
    X = df_sample[['price', 'freight_value', 'product_weight_g', 'order_item_id']]
    y = df_sample['is_late']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

    # -------------------------------------------------------------------
    # 3. Optuna Objective Function definieren
    # -------------------------------------------------------------------
    # Das ist das Herzstück: Optuna testet Werte, führt diese Funktion aus und versucht,
    # den Rückgabewert (hier ROC-AUC) zu maximieren.
    def objective(trial):
        # 1. Hyperparameter-Suchraum definieren
        n_estimators = trial.suggest_int('n_estimators', 50, 300, step=50)
        max_depth = trial.suggest_int('max_depth', 3, 15)
        min_samples_split = trial.suggest_int('min_samples_split', 2, 20)
        min_samples_leaf = trial.suggest_int('min_samples_leaf', 1, 10)
        
        # 2. Modell mit diesen Parametern initialisieren
        model = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            random_state=42,
            n_jobs=-1
        )
        
        # 3. Modell via Cross-Validation evaluieren (verhindert Overfitting auf einem fixen Test-Set)
        # Wir nutzen nur 3 Folds für den Speed im Skript
        scores = cross_val_score(model, X_train, y_train, cv=3, scoring='roc_auc')
        
        return scores.mean()

    # -------------------------------------------------------------------
    # 4. Die Optimierung starten (Die "Study")
    # -------------------------------------------------------------------
    print("Starte Optuna Study (30 Trials)... Das kann ein paar Sekunden dauern.")
    # direction='maximize', weil wir einen möglichst hohen ROC-AUC wollen
    study = optuna.create_study(direction='maximize', sampler=optuna.samplers.TPESampler(seed=42))
    
    # Wir machen 30 Versuche. (In der Praxis eher 100-500)
    study.optimize(objective, n_trials=30)
    
    print(f"Bester ROC-AUC: {study.best_value:.4f}")
    print(f"Beste Parameter: {study.best_params}")
    
    # Daten für die Visualisierung extrahieren
    df_trials = study.trials_dataframe()
    
    # Wir berechnen die Importance der einzelnen Parameter
    importances = optuna.importance.get_param_importances(study)
    df_importances = pd.DataFrame(list(importances.items()), columns=['Parameter', 'Importance'])

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

    # --- Plot 1: Optimization History (Wie lernt Optuna?) ---
    # Alle Versuche als Punkte
    ax1.scatter(df_trials['number'], df_trials['value'], color='#3498db', alpha=0.6, s=50, label='Einzelner Trial (Versuch)')
    
    # Die "Best Value So Far" Linie berechnen
    best_values = df_trials['value'].expanding().max()
    ax1.plot(df_trials['number'], best_values, color='#e74c3c', linewidth=3, label='Bestes Modell bisher')
    
    ax1.set_title("1. Optimization History (Bayesian Learning)", fontsize=14, fontweight='bold', color='white')
    ax1.set_xlabel("Trial Nummer (Fortschritt)")
    ax1.set_ylabel("Modell-Performance (Cross-Val ROC-AUC)")
    
    legend1 = ax1.legend(loc='lower right', facecolor='#1e1e1e', edgecolor='#333333')
    plt.setp(legend1.get_texts(), color='white')

    # Info-Box Plot 1
    info_1 = (
        "Directed Search vs. Brute Force:\n"
        "--------------------------------\n"
        "Optuna rät nicht einfach blind. Nach den ersten Versuchen\n"
        "baut es intern ein probabilistisches Modell auf, das vorhersagt,\n"
        "welche Regler-Kombinationen am vielversprechendsten sind.\n"
        "Die rote Linie zeigt, wie das Modell mit der Zeit (von links nach rechts)\n"
        "immer stärkere Parameter-Setups findet."
    )
    props1 = dict(boxstyle='round,pad=1', facecolor='#1e1e1e', alpha=0.9, edgecolor='#e74c3c', linewidth=1.5)
    ax1.text(0.4, 0.25, info_1, transform=ax1.transAxes, fontsize=11,
             verticalalignment='center', bbox=props1, color='white')

    # --- Plot 2: Hyperparameter Importance ---
    # Welcher Parameter hatte den größten Einfluss auf Erfolg oder Misserfolg?
    sns.barplot(x='Importance', y='Parameter', data=df_importances, palette='viridis', ax=ax2, edgecolor='none')
    
    ax2.set_title("2. Hyperparameter Importance", fontsize=14, fontweight='bold', color='white')
    ax2.set_xlabel("Einfluss auf die Modell-Performance (0 bis 1)")
    ax2.set_ylabel("")

    # Info-Box Plot 2
    info_2 = (
        "Wo lohnt sich das Tuning?\n"
        "-------------------------\n"
        "Dieser Plot rettet Tage an Rechenzeit! Er zeigt uns,\n"
        "welcher Regler wirklich wichtig für den Datensatz ist.\n"
        "Oft ist z.B. die Baumtiefe (max_depth) extrem dominant,\n"
        "während die Anzahl der Bäume (n_estimators) ab einem\n"
        "gewissen Punkt fast egal ist.\n\n"
        "Erkenntnis: Fokus für das nächste Tuning nur auf die Top 2 Parameter legen!"
    )
    props2 = dict(boxstyle='round,pad=1', facecolor='#1e1e1e', alpha=0.9, edgecolor='#2ecc71', linewidth=1.5)
    ax2.text(0.35, 0.4, info_2, transform=ax2.transAxes, fontsize=11,
             verticalalignment='center', bbox=props2, color='white')

    plt.suptitle("MLOps & Tuning: Bayesian Hyperparameter Optimization (Optuna)", 
                 fontsize=18, fontweight='bold', color='white', y=1.05)
    plt.tight_layout()
    
    # Speichern der Grafik
    plt.savefig(GRAPHIC_PATH, dpi=300, bbox_inches='tight', facecolor=fig.get_facecolor())
    plt.close()
    
    print(f"✅ Grafik erfolgreich gespeichert unter:\n{GRAPHIC_PATH}")

if __name__ == "__main__":
    main()
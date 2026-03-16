"""
Griff 56: Ensemble & Stacking (Wisdom of the Crowds)

Was ist das und was macht das Skript?
Ein einzelnes Machine-Learning-Modell (z.B. ein Entscheidungsbaum) macht Fehler. 
"Ensemble Learning" ist die Idee, dass eine Gruppe von Modellen ("Die Weisheit der Vielen") 
besser ist als ein einzelnes Experten-Modell – solange die Modelle unterschiedliche 
Stärken und Schwächen haben und unterschiedliche Fehler machen.

1. Bagging (z.B. Random Forest): Baut viele unabhängige Bäume parallel und lässt sie abstimmen. Reduziert Varianz.
2. Boosting (z.B. Gradient Boosting): Baut Bäume nacheinander. Jeder neue Baum versucht, die Fehler des vorherigen Baums zu korrigieren. Reduziert Bias.
3. Stacking (Stacked Generalization): Die Königsklasse (oft Sieger in Kaggle-Wettbewerben). 
   Wir nehmen völlig verschiedene Modelle (z.B. Random Forest + Gradient Boosting) als "Base Learner". 
   Ihre Vorhersagen werden als NEUE Features in ein "Meta-Modell" (z.B. Logistische Regression) gefüttert. 
   Das Meta-Modell lernt, welchem Base Learner es in welcher Situation vertrauen kann!

Business Case in diesem Skript:
Wir wollen vorhersagen, ob ein Kunde eine perfekte 5-Sterne-Bewertung abgibt (Top-Tier Customer Experience). 
Wir vergleichen ein einzelnes starkes Modell (Random Forest) mit einem Stacking-Ensemble, 
das Random Forest und Gradient Boosting kombiniert. Wir zeigen, wie das Ensemble 
die ROC-AUC (Trennpräzision) nochmal um die letzten entscheidenden Prozentpunkte nach oben drückt.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_curve, roc_auc_score

# -------------------------------------------------------------------
# 1. Setup & Pfade
# -------------------------------------------------------------------
ORDERS_PATH = "/Users/stefanbechthold/statistik_business_analytics/data/olist_orders_dataset.csv"
REVIEWS_PATH = "/Users/stefanbechthold/statistik_business_analytics/data/olist_order_reviews_dataset.csv"
ITEMS_PATH = "/Users/stefanbechthold/statistik_business_analytics/data/olist_order_items_dataset.csv"

GRAPHIC_DIR = "/Users/stefanbechthold/statistik_business_analytics/Grafiken/06_Klassifikation_und_Validierung"
GRAPHIC_PATH = os.path.join(GRAPHIC_DIR, "56_ensemble_stacking_performance.png")

os.makedirs(GRAPHIC_DIR, exist_ok=True)

def main():
    print("🚀 Starte Griff 56: Model Ensembling & Stacking...")
    
    # -------------------------------------------------------------------
    # 2. Daten laden & Feature Engineering
    # -------------------------------------------------------------------
    df_orders = pd.read_csv(ORDERS_PATH)
    df_reviews = pd.read_csv(REVIEWS_PATH)
    df_items = pd.read_csv(ITEMS_PATH)
    
    # Target (Y): 5-Sterne-Bewertung (1 = Ja, 0 = Nein)
    reviews_unique = df_reviews.drop_duplicates(subset=['order_id'])[['order_id', 'review_score']]
    reviews_unique['is_5_star'] = (reviews_unique['review_score'] == 5).astype(int)
    
    # Lieferzeit (Das stärkste Feature)
    df_orders['purchase_time'] = pd.to_datetime(df_orders['order_purchase_timestamp'], errors='coerce')
    df_orders['delivered_time'] = pd.to_datetime(df_orders['order_delivered_customer_date'], errors='coerce')
    df_orders['delivery_days'] = (df_orders['delivered_time'] - df_orders['purchase_time']).dt.total_seconds() / (24 * 3600)
    
    # Kosten aggregieren
    order_costs = df_items.groupby('order_id').agg({'price': 'sum', 'freight_value': 'sum', 'order_item_id': 'max'}).reset_index()
    order_costs.rename(columns={'order_item_id': 'item_count'}, inplace=True)
    
    # Mergen
    df_merged = pd.merge(df_orders[['order_id', 'delivery_days']], reviews_unique[['order_id', 'is_5_star']], on='order_id')
    df_merged = pd.merge(df_merged, order_costs, on='order_id')
    
    df_clean = df_merged.dropna().copy()
    
    # Ausreißer entfernen für stabile Modelle
    df_clean = df_clean[(df_clean['delivery_days'] > 0) & (df_clean['delivery_days'] < 60) & (df_clean['price'] < 1000)]
    
    # Sample ziehen für schnelle Ausführung im Skript (Stacking ist rechenintensiv!)
    df_sample = df_clean.sample(n=15000, random_state=42).reset_index(drop=True)
    
    features = ['delivery_days', 'price', 'freight_value', 'item_count']
    X = df_sample[features]
    y = df_sample['is_5_star']

    # -------------------------------------------------------------------
    # 3. Modell-Training (Base vs. Stacking)
    # -------------------------------------------------------------------
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
    
    # Modell 1: Random Forest (Base Learner A)
    rf = RandomForestClassifier(n_estimators=100, max_depth=6, random_state=42)
    rf.fit(X_train, y_train)
    rf_probs = rf.predict_proba(X_test)[:, 1]
    
    # Modell 2: Gradient Boosting (Base Learner B)
    gb = GradientBoostingClassifier(n_estimators=100, max_depth=4, random_state=42)
    gb.fit(X_train, y_train)
    gb_probs = gb.predict_proba(X_test)[:, 1]
    
    # Modell 3: Stacking Ensemble
    # Wir kombinieren RF und GB. Die Logistische Regression ist das "Meta-Modell" (Final Estimator).
    # Sie nimmt die Wahrscheinlichkeits-Vorhersagen von RF und GB als Input und lernt die finale Entscheidung.
    estimators = [
        ('rf', RandomForestClassifier(n_estimators=100, max_depth=6, random_state=42)),
        ('gb', GradientBoostingClassifier(n_estimators=100, max_depth=4, random_state=42))
    ]
    
    stacking_clf = StackingClassifier(
        estimators=estimators, 
        final_estimator=LogisticRegression(),
        cv=3 # Cross-Validation intern, um Overfitting der Base Learner zu vermeiden!
    )
    
    stacking_clf.fit(X_train, y_train)
    stack_probs = stacking_clf.predict_proba(X_test)[:, 1]

    # ROC AUC Scores berechnen
    auc_rf = roc_auc_score(y_test, rf_probs)
    auc_gb = roc_auc_score(y_test, gb_probs)
    auc_stack = roc_auc_score(y_test, stack_probs)

    # -------------------------------------------------------------------
    # 4. Visualisierung im Dark Mode Design
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

    # --- Plot 1: ROC Kurven Vergleich ---
    # Berechnen der False Positive Rates und True Positive Rates für die Kurven
    fpr_rf, tpr_rf, _ = roc_curve(y_test, rf_probs)
    fpr_gb, tpr_gb, _ = roc_curve(y_test, gb_probs)
    fpr_stack, tpr_stack, _ = roc_curve(y_test, stack_probs)
    
    # Zufalls-Raten (Diagonal)
    ax1.plot([0, 1], [0, 1], color='white', linestyle='--', alpha=0.5)
    
    # Kurven plotten
    ax1.plot(fpr_rf, tpr_rf, color='#3498db', linewidth=2, label=f'Random Forest (AUC = {auc_rf:.3f})')
    ax1.plot(fpr_gb, tpr_gb, color='#f1c40f', linewidth=2, label=f'Gradient Boosting (AUC = {auc_gb:.3f})')
    ax1.plot(fpr_stack, tpr_stack, color='#2ecc71', linewidth=3, label=f'Stacking Ensemble (AUC = {auc_stack:.3f})')
    
    ax1.set_title("1. ROC Kurven (Trennpräzision)", fontsize=14, fontweight='bold', color='white')
    ax1.set_xlabel("False Positive Rate (Fehlalarme)")
    ax1.set_ylabel("True Positive Rate (Find Rate)")
    
    legend1 = ax1.legend(loc='lower right', facecolor='#1e1e1e', edgecolor='#333333')
    plt.setp(legend1.get_texts(), color='white')

    # --- Plot 2: Performance Boost (Bar Chart) ---
    models = ['Random Forest\n(Bagging)', 'Gradient Boosting\n(Boosting)', 'Stacking Ensemble\n(Meta-Learning)']
    scores = [auc_rf, auc_gb, auc_stack]
    colors = ['#3498db', '#f1c40f', '#2ecc71']
    
    bars = ax2.bar(models, scores, color=colors, edgecolor='none')
    
    # Y-Achse zoomen, um den feinen Unterschied im Top-Bereich sichtbar zu machen
    min_score = min(scores) - 0.02
    ax2.set_ylim(min_score, max(scores) + 0.01)
    
    ax2.set_title("2. ROC-AUC Performance Boost durch Stacking", fontsize=14, fontweight='bold', color='white')
    ax2.set_ylabel("ROC-AUC Score")

    # Werte über die Balken schreiben
    for bar in bars:
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height + 0.001, 
                 f"{height:.3f}", ha='center', va='bottom', color='white', fontweight='bold', fontsize=12)

    # Info-Box Plot 2
    info_text = (
        "Die Stacking-Architektur:\n"
        "-------------------------\n"
        "Schritt 1: RF und GB machen ihre eigenen Vorhersagen.\n"
        "Schritt 2: Das Meta-Modell analysiert, wer von beiden oft recht hat.\n"
        "Ergebnis: Das Ensemble schlägt isolierte Einzelmodelle fast immer.\n\n"
        "Der Trade-off: Stacking bringt 1-3% mehr Performance,\n"
        "verdoppelt aber die Trainingszeit und macht das System\n"
        "zu einer massiven Blackbox (Interpretability sinkt!).\n"
        "Sinnvoll für Kaggle-Wettbewerbe oder Hoch-Risiko-Scoring."
    )
    props = dict(boxstyle='round,pad=1', facecolor='#1e1e1e', alpha=0.9, edgecolor='#2ecc71', linewidth=1.5)
    ax2.text(0.05, 0.95, info_text, transform=ax2.transAxes, fontsize=11,
             verticalalignment='top', bbox=props, color='white')

    plt.suptitle("Ensemble & Stacking: Maximale Predictive Performance", 
                 fontsize=18, fontweight='bold', color='white', y=1.05)
    plt.tight_layout()
    
    # Speichern der Grafik
    plt.savefig(GRAPHIC_PATH, dpi=300, bbox_inches='tight', facecolor=fig.get_facecolor())
    plt.close()
    
    print(f"✅ Grafik erfolgreich gespeichert unter:\n{GRAPHIC_PATH}")

if __name__ == "__main__":
    main()
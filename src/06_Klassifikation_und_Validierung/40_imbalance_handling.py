"""
Griff 40: Imbalance im Klassifikationsproblem? (SMOTE, Undersample, Class Weights)

Was ist das und was macht das Skript?
In der Realität sind Zielvariablen oft extrem ungleich verteilt (Imbalanced Data). 
Beispiele: Betrugsfälle (Fraud) machen nur 0.1% aus, Kündigungen (Churn) nur 5%, 
oder – wie in diesem Skript – extrem schlechte 1-Stern-Bewertungen.

Wenn wir ein Standard-Modell (z.B. Random Forest) auf solche Daten trainieren, 
wird es "faul". Es sagt einfach immer die Mehrheitsklasse vorher und erreicht 
damit 90% Accuracy – aber es findet keinen einzigen Betrüger/Kündiger!

Strategien gegen Imbalance:
1. Class Weights: Wir bestrafen das Modell härter, wenn es die Minderheitsklasse falsch vorhersagt.
2. Undersampling: Wir werfen zufällig Daten der Mehrheitsklasse weg, bis ein 50/50 Verhältnis entsteht.
   Vorteil: Extrem schnell. Nachteil: Wir verlieren wertvolle Informationen.
3. SMOTE (Synthetic Minority Over-sampling Technique): Wir erzeugen künstliche (synthetische) 
   Datenpunkte der Minderheitsklasse, indem wir zwischen existierenden Punkten interpolieren.
   Achtung: SMOTE darf IMMER NUR auf den Trainingsdaten angewendet werden, niemals auf den Testdaten!

Business Case in diesem Skript:
Wir wollen eskalierende Kunden (1-Stern-Bewertung) frühzeitig erkennen, um den 
Kundenservice proaktiv einzuschalten. Wir vergleichen die Performance der verschiedenen 
Imbalance-Strategien mit Fokus auf den "Recall" (Wie viele der echten 1-Stern-Fälle finden wir?).

WICHTIG: Benötigt das Paket 'imbalanced-learn' -> pip install imbalanced-learn
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import recall_score, f1_score, precision_score

try:
    from imblearn.over_sampling import SMOTE
    from imblearn.under_sampling import RandomUnderSampler
except ImportError:
    print("FEHLER: Die Bibliothek 'imbalanced-learn' fehlt. Bitte ausführen: pip install imbalanced-learn")
    exit()

# -------------------------------------------------------------------
# 1. Setup & Pfade
# -------------------------------------------------------------------
ORDERS_PATH = "/Users/stefanbechthold/statistik_business_analytics/data/olist_orders_dataset.csv"
REVIEWS_PATH = "/Users/stefanbechthold/statistik_business_analytics/data/olist_order_reviews_dataset.csv"
ITEMS_PATH = "/Users/stefanbechthold/statistik_business_analytics/data/olist_order_items_dataset.csv"

GRAPHIC_DIR = "/Users/stefanbechthold/statistik_business_analytics/Grafiken/06_Klassifikation_und_Validierung"
GRAPHIC_PATH = os.path.join(GRAPHIC_DIR, "40_imbalanced_classification.png")

os.makedirs(GRAPHIC_DIR, exist_ok=True)

def main():
    # -------------------------------------------------------------------
    # 2. Daten laden & Eskalations-Target bauen
    # -------------------------------------------------------------------
    df_orders = pd.read_csv(ORDERS_PATH)
    df_reviews = pd.read_csv(REVIEWS_PATH)
    df_items = pd.read_csv(ITEMS_PATH)
    
    # Target (Y): 1-Stern-Bewertung (Eskalation = 1, Rest = 0)
    reviews_unique = df_reviews.drop_duplicates(subset=['order_id'])[['order_id', 'review_score']]
    reviews_unique['is_escalation'] = (reviews_unique['review_score'] == 1).astype(int)
    
    # Features (X): Lieferverzögerung, Preis, Fracht
    df_orders['purchase_time'] = pd.to_datetime(df_orders['order_purchase_timestamp'], errors='coerce')
    df_orders['delivered_time'] = pd.to_datetime(df_orders['order_delivered_customer_date'], errors='coerce')
    df_orders['estimated_time'] = pd.to_datetime(df_orders['order_estimated_delivery_date'], errors='coerce')
    
    # Wie viele Tage ZU SPÄT kam das Paket? (Negative Werte = zu früh)
    df_orders['delay_days'] = (df_orders['delivered_time'] - df_orders['estimated_time']).dt.total_seconds() / (24 * 3600)
    
    # Kosten aggregieren
    order_costs = df_items.groupby('order_id').agg({'price': 'sum', 'freight_value': 'sum'}).reset_index()
    
    # Mergen
    df_merged = pd.merge(df_orders[['order_id', 'delay_days']], reviews_unique[['order_id', 'is_escalation']], on='order_id', how='inner')
    df_merged = pd.merge(df_merged, order_costs, on='order_id', how='inner')
    
    df_clean = df_merged.dropna().copy()
    
    # Sample ziehen
    df_sample = df_clean.sample(n=25000, random_state=42)
    
    X = df_sample[['delay_days', 'price', 'freight_value']]
    y = df_sample['is_escalation']
    
    print(f"Klassenverteilung im Datensatz:\n{y.value_counts(normalize=True) * 100}")

    # -------------------------------------------------------------------
    # 3. Resampling & Modellierung
    # -------------------------------------------------------------------
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
    
    models = {}
    
    # 1. Baseline (Ignoriert Imbalance)
    rf_base = RandomForestClassifier(n_estimators=100, max_depth=6, random_state=42)
    rf_base.fit(X_train, y_train)
    models['Baseline (Ignoriert)'] = rf_base
    
    # 2. Class Weights (Gewichtet Minderheit stärker)
    rf_weight = RandomForestClassifier(n_estimators=100, max_depth=6, class_weight='balanced', random_state=42)
    rf_weight.fit(X_train, y_train)
    models['Class Weights'] = rf_weight
    
    # 3. Random Undersampling (Trainingsdaten schrumpfen)
    rus = RandomUnderSampler(random_state=42)
    X_train_rus, y_train_rus = rus.fit_resample(X_train, y_train)
    rf_under = RandomForestClassifier(n_estimators=100, max_depth=6, random_state=42)
    rf_under.fit(X_train_rus, y_train_rus)
    models['Undersampling'] = rf_under
    
    # 4. SMOTE (Synthetische Datenpunkte generieren)
    smote = SMOTE(random_state=42)
    X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)
    rf_smote = RandomForestClassifier(n_estimators=100, max_depth=6, random_state=42)
    rf_smote.fit(X_train_smote, y_train_smote)
    models['SMOTE'] = rf_smote

    # Metriken berechnen (Fokus auf Recall für die Klasse 1)
    results = []
    for name, model in models.items():
        y_pred = model.predict(X_test)
        rec = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred)
        results.append({'Strategie': name, 'Recall (Find Rate)': rec, 'F1-Score': f1, 'Precision': prec})
        
    df_results = pd.DataFrame(results)

    # -------------------------------------------------------------------
    # 4. Visualisierung im Dark Mode Design
    # -------------------------------------------------------------------
    plt.style.use('dark_background')
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 7), facecolor='#121212')
    ax1.set_facecolor('#121212')
    ax2.set_facecolor('#121212')

    # --- Plot 1: Visualisierung von SMOTE (Original vs Synthetisch) ---
    # Wir zeigen ein 2D-Streudiagramm der Trainingsdaten
    # Um SMOTE sichtbar zu machen, plotten wir die Original-Minderheitsklasse und die SMOTE-Punkte
    X_train_orig_min = X_train[y_train == 1]
    
    # Die synthetischen Punkte sind diejenigen in X_train_smote, die NICHT in X_train sind (vereinfacht über Längen-Differenz)
    num_orig = len(X_train)
    X_synthetic = X_train_smote.iloc[num_orig:]
    
    sns.scatterplot(x=X_train_orig_min['delay_days'], y=X_train_orig_min['price'], 
                    color='#3498db', label='Originale 1-Stern Fälle', alpha=0.6, s=30, edgecolor='none', ax=ax1)
    sns.scatterplot(x=X_synthetic['delay_days'], y=X_synthetic['price'], 
                    color='#e74c3c', label='Synthetische SMOTE Fälle', alpha=0.3, s=15, marker='x', ax=ax1)
    
    ax1.set_title("1. Wie SMOTE funktioniert (Synthetische Daten)", fontsize=14, fontweight='bold', color='white')
    ax1.set_xlabel("Lieferverzögerung (Tage)")
    ax1.set_ylabel("Preis (BRL)")
    ax1.set_xlim(-10, 30) # Ausschnitt für bessere Sichtbarkeit
    ax1.set_ylim(0, 500)
    
    legend1 = ax1.legend(loc='upper right', facecolor='#1e1e1e', edgecolor='#333333')
    plt.setp(legend1.get_texts(), color='white')

    # Info-Box Plot 1
    info_1 = (
        "SMOTE (Synthetic Minority Over-sampling):\n"
        "Zieht Linien zwischen bestehenden Eskalations-Fällen\n"
        "(blau) und generiert neue, künstliche Datenpunkte\n"
        "darauf (rot). So bekommt der Algorithmus genug\n"
        "Beispiele, um Muster zu lernen, ohne einfach\n"
        "nur dieselben Punkte zu duplizieren."
    )
    props1 = dict(boxstyle='round,pad=1', facecolor='#1e1e1e', alpha=0.9, edgecolor='#e74c3c', linewidth=1.5)
    ax1.text(0.02, 0.95, info_1, transform=ax1.transAxes, fontsize=11,
             verticalalignment='top', bbox=props1, color='#e0e0e0')

    # --- Plot 2: Performance Vergleich (Recall vs F1) ---
    # Daten für Barplot umstrukturieren
    df_melt = df_results.melt(id_vars='Strategie', value_vars=['Recall (Find Rate)', 'F1-Score'], 
                              var_name='Metrik', value_name='Score')
    
    sns.barplot(x='Strategie', y='Score', hue='Metrik', data=df_melt, palette=['#2ecc71', '#9b59b6'], ax=ax2, edgecolor='none')
    
    ax2.set_title("2. Business Impact: Finden wir die wütenden Kunden?", fontsize=14, fontweight='bold', color='white')
    ax2.set_xlabel("Imbalance-Strategie")
    ax2.set_ylabel("Score (0 bis 1)")
    ax2.set_ylim(0, 1)
    
    legend2 = ax2.legend(loc='upper left', facecolor='#1e1e1e', edgecolor='#333333')
    plt.setp(legend2.get_texts(), color='white')

    # Werte über die Balken schreiben
    for i, bar in enumerate(ax2.patches):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02, 
                 f"{bar.get_height():.2f}", 
                 ha='center', va='bottom', color='white', fontweight='bold', fontsize=11)

    # Info-Box Plot 2
    info_2 = (
        "Business Trade-off:\n"
        "-------------------\n"
        "Das Baseline-Modell ignoriert das Problem und findet\n"
        f"nur {df_results.iloc[0]['Recall (Find Rate)']*100:.0f}% der Eskalationen (Recall).\n\n"
        "Undersampling und SMOTE zwingen das Modell,\n"
        "die Minderheitsklasse ernst zu nehmen. Der Recall\n"
        "springt massiv an! Der Preis dafür ist eine sinkende\n"
        "Precision (mehr False Alarms), was den F1-Score drückt."
    )
    props2 = dict(boxstyle='round,pad=1', facecolor='#1e1e1e', alpha=0.9, edgecolor='#2ecc71', linewidth=1.5)
    ax2.text(0.45, 0.95, info_2, transform=ax2.transAxes, fontsize=11,
             verticalalignment='top', bbox=props2, color='#e0e0e0')

    plt.suptitle("Imbalanced Classification: Minderheitsklassen (Extreme Events) vorhersagen", fontsize=18, fontweight='bold', color='white', y=1.05)
    plt.tight_layout()
    
    # Speichern
    plt.savefig(GRAPHIC_PATH, dpi=300, bbox_inches='tight', facecolor=fig.get_facecolor())
    plt.close()

if __name__ == "__main__":
    main()
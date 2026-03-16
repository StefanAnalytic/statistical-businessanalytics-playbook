"""
Griff 27: Klassifikator Leistung richtig bewerten? (Accuracy, Precision, Recall, F1, ROC & PR AUC)

Was ist das und was macht das Skript?
Wenn wir ein Ja/Nein-Ereignis vorhersagen (z.B. "Wird die Lieferung verspätet sein?"), 
reicht "Accuracy" (Genauigkeit) absolut nicht aus! 
Beispiel: Wenn nur 5% der Pakete verspätet sind, kann ein dummes Modell einfach IMMER 
"Pünktlich" raten. Es hat dann 95% Accuracy, ist fürs Business aber komplett nutzlos!

Wir brauchen bessere Metriken:
1. Confusion Matrix: Wo genau macht das Modell Fehler? (False Positives vs. False Negatives)
2. Precision (Relevanz): Wenn das Modell sagt "Verspätet!", wie oft stimmt das wirklich?
3. Recall (Sensitivität): Von ALLEN WIRKLICH verspäteten Paketen, wie viele hat das Modell gefunden?
4. F1-Score: Der harmonische Kompromiss zwischen Precision und Recall.
5. ROC AUC: Zeigt die Trennschärfe des Modells über verschiedene Wahrscheinlichkeits-Schwellenwerte.
6. PR AUC (Precision-Recall Area): Der Goldstandard bei extrem ungleichen (imbalanced) Daten!

Business Case in diesem Skript:
Wir trainieren einen Random Forest, um vorherzusagen, ob eine Bestellung verspätet ankommt ('is_late').
Wir nutzen Vorab-Informationen (Preis, Versandkosten, geschätzte Lieferdauer). Da Verspätungen 
in der Minderheit sind, entlarven wir die "Accuracy-Lüge" und bewerten das Modell anhand 
harter Business-Metriken (PR-Curve).
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (confusion_matrix, classification_report, 
                             roc_curve, auc, precision_recall_curve, 
                             average_precision_score, accuracy_score, 
                             precision_score, recall_score, f1_score)

# -------------------------------------------------------------------
# 1. Setup & Pfade
# -------------------------------------------------------------------
# Wir sind im Ordner "06_Klassifikation_und_Validierung"
ORDERS_PATH = "/Users/stefanbechthold/statistik_business_analytics/data/olist_orders_dataset.csv"
ITEMS_PATH = "/Users/stefanbechthold/statistik_business_analytics/data/olist_order_items_dataset.csv"

GRAPHIC_DIR = "/Users/stefanbechthold/statistik_business_analytics/Grafiken/06_Klassifikation_und_Validierung"
GRAPHIC_PATH = os.path.join(GRAPHIC_DIR, "27_klassifikator_bewertung.png")

os.makedirs(GRAPHIC_DIR, exist_ok=True)

def main():
    print("🚀 Starte Griff 27: Klassifikator Leistung richtig bewerten...")
    
    # -------------------------------------------------------------------
    # 2. Daten laden & Imbalanced Dataset aufbauen
    # -------------------------------------------------------------------
    print("Lade Bestelldaten und berechne 'is_late'...")
    df_orders = pd.read_csv(ORDERS_PATH)
    df_items = pd.read_csv(ITEMS_PATH)
    
    # Zeiten konvertieren
    df_orders['purchase_time'] = pd.to_datetime(df_orders['order_purchase_timestamp'], errors='coerce')
    df_orders['delivered_time'] = pd.to_datetime(df_orders['order_delivered_customer_date'], errors='coerce')
    df_orders['estimated_time'] = pd.to_datetime(df_orders['order_estimated_delivery_date'], errors='coerce')
    
    df_clean = df_orders.dropna(subset=['purchase_time', 'delivered_time', 'estimated_time']).copy()
    
    # Zielvariable: Verspätet? (1 = Ja, 0 = Nein)
    df_clean['is_late'] = (df_clean['delivered_time'] > df_clean['estimated_time']).astype(int)
    
    # Feature Engineering (OHNE Data Leakage! Wir dürfen nur nutzen, was beim Kauf bekannt ist)
    df_clean['estimated_delivery_days'] = (df_clean['estimated_time'] - df_clean['purchase_time']).dt.total_seconds() / (24 * 3600)
    
    # Bestellkosten aggregieren
    order_costs = df_items.groupby('order_id').agg({'price': 'sum', 'freight_value': 'sum'}).reset_index()
    df_merged = pd.merge(df_clean[['order_id', 'is_late', 'estimated_delivery_days']], order_costs, on='order_id', how='inner')
    
    # Wir nehmen ein Sample für schnelleres Training
    df_sample = df_merged.dropna().sample(n=20000, random_state=42)
    
    # X und y definieren
    X = df_sample[['estimated_delivery_days', 'price', 'freight_value']]
    y = df_sample['is_late']
    
    late_rate = y.mean() * 100
    print(f"Datensatz: {len(df_sample)} Bestellungen. Verspätungsquote (Imbalance): {late_rate:.1f}%")

    # -------------------------------------------------------------------
    # 3. Modell trainieren & Vorhersagen treffen
    # -------------------------------------------------------------------
    print("Trainiere Random Forest Classifier...")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
    
    # Wir nutzen class_weight='balanced', um dem Modell zu sagen, dass Verspätungen wichtig sind!
    rf = RandomForestClassifier(n_estimators=100, max_depth=8, class_weight='balanced', random_state=42)
    rf.fit(X_train, y_train)
    
    # Vorhersagen (Harte Klassen 0/1)
    y_pred = rf.predict(X_test)
    
    # Wahrscheinlichkeiten (Für ROC und PR Kurven) - Wir wollen die Wahrscheinlichkeit für Klasse 1 (Verspätet)
    y_probs = rf.predict_proba(X_test)[:, 1]

    # -------------------------------------------------------------------
    # 4. Metriken berechnen
    # -------------------------------------------------------------------
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    
    # ROC (Receiver Operating Characteristic)
    fpr, tpr, _ = roc_curve(y_test, y_probs)
    roc_auc = auc(fpr, tpr)
    
    # PR (Precision-Recall)
    precision_curve, recall_curve, _ = precision_recall_curve(y_test, y_probs)
    pr_auc = average_precision_score(y_test, y_probs)
    
    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)

    print("\n--- Business Metriken ---")
    print(f"Accuracy:  {acc:.3f} (Oft eine Lüge bei Imbalance!)")
    print(f"Precision: {prec:.3f} (Von den als 'Verspätet' markierten waren X% wirklich verspätet)")
    print(f"Recall:    {rec:.3f} (Wir haben X% ALLER echten Verspätungen gefunden)")
    print(f"F1-Score:  {f1:.3f}")
    print(f"ROC AUC:   {roc_auc:.3f}")
    print(f"PR AUC:    {pr_auc:.3f}")

    # -------------------------------------------------------------------
    # 5. Wunderschöne Visualisierung (4-teiliges Metriken-Dashboard)
    # -------------------------------------------------------------------
    print("\nGeneriere Visualisierungs-Grid...")
    sns.set_theme(style="whitegrid")
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    # --- Plot 1: Confusion Matrix (Wahrheit vs. Vorhersage) ---
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[0, 0], 
                annot_kws={'size': 14, 'weight': 'bold'}, cbar=False)
    
    axes[0, 0].set_title("1. Confusion Matrix (Die harten Fakten)", fontsize=14, fontweight="bold")
    axes[0, 0].set_xlabel("Vorhersage des Modells")
    axes[0, 0].set_ylabel("Die ECHTE Realität")
    axes[0, 0].set_xticklabels(['Pünktlich (0)', 'Verspätet (1)'])
    axes[0, 0].set_yticklabels(['Pünktlich (0)', 'Verspätet (1)'])
    
    # Anmerkungen in die Heatmap
    axes[0, 0].text(0.5, 0.2, "True Negatives", ha='center', color='black', fontsize=10)
    axes[0, 0].text(1.5, 0.2, "False Positives\n(Fehlalarm)", ha='center', color='red', fontsize=10)
    axes[0, 0].text(0.5, 1.2, "False Negatives\n(Übersehen)", ha='center', color='red', fontsize=10)
    axes[0, 0].text(1.5, 1.2, "True Positives\n(Erfolg!)", ha='center', color='white', fontsize=10)

    # --- Plot 2: Bar Chart der Metriken (Die Accuracy-Lüge) ---
    metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
    values = [acc, prec, rec, f1]
    colors = ['#95a5a6', '#3498db', '#e67e22', '#2ecc71']
    
    bars = axes[0, 1].bar(metrics, values, color=colors, edgecolor='black', alpha=0.85)
    axes[0, 1].set_title(f"2. Business-Metriken im Vergleich (Imbalance: {late_rate:.1f}% Late)", fontsize=14, fontweight="bold")
    axes[0, 1].set_ylim(0, 1.05)
    
    for bar in bars:
        axes[0, 1].text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.02,
                        f"{bar.get_height():.2f}", ha='center', fontweight='bold', fontsize=12)
                        
    axes[0, 1].text(0.05, 0.9, "Tipp: Bei seltenen Events\nist Accuracy hoch,\naber F1 meist niedrig!", 
                    transform=axes[0, 1].transAxes, bbox=dict(facecolor='white', alpha=0.9, edgecolor='gray'))

    # --- Plot 3: ROC Kurve ---
    # Zeigt Trade-off zwischen True Positive Rate und False Positive Rate
    axes[1, 0].plot(fpr, tpr, color='#9b59b6', linewidth=3, label=f'ROC Curve (AUC = {roc_auc:.3f})')
    axes[1, 0].plot([0, 1], [0, 1], color='black', linestyle='--', label='Zufalls-Raten (Münzwurf)')
    
    axes[1, 0].set_title("3. ROC Curve (Receiver Operating Characteristic)", fontsize=14, fontweight="bold")
    axes[1, 0].set_xlabel("False Positive Rate (Fehlalarm-Quote)")
    axes[1, 0].set_ylabel("True Positive Rate (Recall / Sensitivität)")
    axes[1, 0].legend(loc='lower right')
    
    # --- Plot 4: Precision-Recall Kurve (Goldstandard für Imbalance!) ---
    # Zeigt den Trade-off zwischen Precision und Recall. Die "Baseline" ist nicht 0.5, 
    # sondern der echte Anteil der 1er im Datensatz!
    baseline = sum(y_test) / len(y_test)
    
    axes[1, 1].plot(recall_curve, precision_curve, color='#e74c3c', linewidth=3, label=f'PR Curve (AUC = {pr_auc:.3f})')
    axes[1, 1].axhline(baseline, color='black', linestyle='--', label=f'Zufall (Baseline = {baseline:.3f})')
    
    axes[1, 1].set_title("4. PR Curve (Precision-Recall) - WICHTIG BEI CHURN/FRAUD!", fontsize=14, fontweight="bold")
    axes[1, 1].set_xlabel("Recall (Wie viele Verspätungen finden wir?)")
    axes[1, 1].set_ylabel("Precision (Wie oft liegen wir richtig?)")
    axes[1, 1].legend(loc='upper right')
    
    # Info-Box
    pr_info = (
        "Warum ist PR besser als ROC bei Imbalance?\n"
        "ROC wird künstlich 'schön' gerechnet durch\n"
        "die vielen True Negatives (Pünktliche Pakete).\n"
        "Die PR-Kurve schaut NUR auf die kritische\n"
        "Klasse (Verspätungen) und zeigt den knallharten\n"
        "Trade-off zwischen Relevanz und Findungsquote."
    )
    axes[1, 1].text(0.05, 0.05, pr_info, transform=axes[1, 1].transAxes, fontsize=11,
                    verticalalignment='bottom', bbox=dict(facecolor='white', alpha=0.9, edgecolor='#e74c3c'))

    # Layout optimieren und speichern
    plt.suptitle("Klassifikations-Evaluation: Jenseits der Accuracy-Lüge", 
                 fontsize=18, fontweight="bold", y=1.03)
    plt.tight_layout()
    sns.despine()
    
    plt.savefig(GRAPHIC_PATH, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✅ Grafik erfolgreich gespeichert unter:\n{GRAPHIC_PATH}")

if __name__ == "__main__":
    main()
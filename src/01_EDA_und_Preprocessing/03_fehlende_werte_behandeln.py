"""
Griff 3: Fehlende Werte behandeln (Imputation)

Was macht dieses Skript?
1. Es lädt die Olist-Produktdaten, bei denen teilweise Gewicht und Maße fehlen.
2. Es zeigt, wie viele Werte wirklich fehlen.
3. Es füllt die fehlenden Werte auf drei verschiedene Arten auf:
   - Einfacher Median (Simple Imputer)
   - K-Nearest Neighbors (KNN Imputer)
   - Iterative Imputation (MICE)
4. Es visualisiert die Original-Verteilung im Vergleich zu den aufgefüllten Daten, 
   damit wir sehen, welche Methode die Datenstruktur am besten erhält.

Warum ist das nützlich?
Fehlende Daten sind der Endgegner im Data Science Alltag. Wenn wir sie falsch auffüllen 
(z.B. überall einfach 0 eintragen), zerstören wir die Logik unserer Daten. 
Die richtige Imputation rettet Datenpunkte, die wir sonst wegwerfen müssten.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Scikit-Learn (sklearn) ist DIE Standard-Bibliothek für Machine Learning und Datenvorbereitung in Python
from sklearn.impute import SimpleImputer, KNNImputer

# MICE heißt in sklearn "IterativeImputer" und ist noch "experimentell", daher dieser Extra-Import
from sklearn.experimental import enable_iterative_imputer 
from sklearn.impute import IterativeImputer

# -------------------------------------------------------------------
# 1. Pfade definieren
# -------------------------------------------------------------------
DATA_PATH = "/Users/stefanbechthold/statistik_business_analytics/data/olist_products_dataset.csv"
GRAPHIC_DIR = "/Users/stefanbechthold/statistik_business_analytics/Grafiken/01_EDA_und_Preprocessing"
GRAPHIC_PATH = os.path.join(GRAPHIC_DIR, "03_imputation_vergleich.png")

os.makedirs(GRAPHIC_DIR, exist_ok=True)

# -------------------------------------------------------------------
# 2. Daten laden und Problem analysieren
# -------------------------------------------------------------------
print("Lade Produktdaten...")
df_products = pd.read_csv(DATA_PATH)

# Wir nehmen uns nur die numerischen Spalten vor, bei denen Werte fehlen können
features = ['product_weight_g', 'product_length_cm', 'product_height_cm', 'product_width_cm']
df_missing = df_products[features].copy()

print("\n--- Fehlende Werte vor der Imputation ---")
print(df_missing.isnull().sum())

# -------------------------------------------------------------------
# 3. Imputation (Das Auffüllen der Lücken)
# -------------------------------------------------------------------
print("\nStarte Imputations-Verfahren (Das kann bei KNN und MICE ein paar Sekunden dauern)...")

# Methode A: Simple Imputation (Median)
# Wir sagen dem Imputer: "Fülle alle leeren Zellen mit dem Median der jeweiligen Spalte."
imputer_median = SimpleImputer(strategy='median')
df_median = pd.DataFrame(imputer_median.fit_transform(df_missing), columns=features)

# Methode B: KNN Imputation
# Er sucht die 5 ähnlichsten Produkte (n_neighbors=5) und nimmt deren Durchschnittsgewicht.
imputer_knn = KNNImputer(n_neighbors=5)
df_knn = pd.DataFrame(imputer_knn.fit_transform(df_missing), columns=features)

# Methode C: MICE (Iterative Imputation)
# Er lernt Zusammenhänge (z.B. Volumen hängt mit Gewicht zusammen) und füllt iterativ auf.
imputer_mice = IterativeImputer(random_state=42)
df_mice = pd.DataFrame(imputer_mice.fit_transform(df_missing), columns=features)

# -------------------------------------------------------------------
# 4. Visualisierung der Ergebnisse
# -------------------------------------------------------------------
print("\nErstelle Grafik zum Vergleich der Methoden...")

# Wir schauen uns an, wie sich die Verteilung des Gewichts ('product_weight_g') verändert hat
target_col = 'product_weight_g'

# Um extreme Ausreißer in der Grafik auszublenden (damit man die Kurven gut sieht), 
# schneiden wir bei 10.000 Gramm ab.
limit = 10000 

plt.figure(figsize=(12, 6))

# Original-Daten (nur die, die vorhanden sind) als schwarze, gestrichelte Linie
sns.kdeplot(df_missing[df_missing[target_col] < limit][target_col].dropna(), 
            color='black', linewidth=3, linestyle='--', label='Original (Ohne fehlende Werte)')

# Einfacher Median
sns.kdeplot(df_median[df_median[target_col] < limit][target_col], 
            color='red', linewidth=2, label='Simple Imputer (Median)')

# KNN
sns.kdeplot(df_knn[df_knn[target_col] < limit][target_col], 
            color='blue', linewidth=2, label='KNN Imputer')

# MICE
sns.kdeplot(df_mice[df_mice[target_col] < limit][target_col], 
            color='green', linewidth=2, label='MICE (Iterative Imputer)')

# Layout aufhübschen
plt.title("Vergleich der Imputationsmethoden: Produktgewicht (Gramm)", fontsize=16, pad=20)
plt.xlabel("Gewicht in Gramm", fontsize=12)
plt.ylabel("Dichte (Häufigkeit)", fontsize=12)
plt.legend()
sns.despine()

# Speichern
plt.tight_layout()
plt.savefig(GRAPHIC_PATH, dpi=300)
print(f"Erfolg! Grafik gespeichert unter: {GRAPHIC_PATH}")

plt.close()
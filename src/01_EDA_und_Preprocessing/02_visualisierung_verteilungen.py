"""
Griff 2: Visualisierung von Distributionen & Beziehungen (Korrelations-Heatmap)

Was macht dieses Skript?
1. Es lädt drei verschiedene Olist-Datensätze (Items, Payments, Reviews).
2. Es verknüpft (mergt) diese Tabellen anhand der Bestellnummer (order_id), ähnlich wie ein SVERWEIS in Excel.
3. Es berechnet die statistische Korrelation (Pearson) zwischen wichtigen numerischen Business-Metriken.
4. Es erstellt eine wunderschöne Heatmap (Farbmatrix), die zeigt, welche Variablen zusammenhängen, 
   und speichert sie ab.

Warum ist das nützlich?
Die Heatmap zeigt uns sofort "Red Flags" oder starke Business-Treiber. 
Wir sehen auf einen Blick: Hängen hohe Preise mit schlechten Bewertungen zusammen? 
Führen teure Produkte zu mehr Ratenzahlungen?
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

# -------------------------------------------------------------------
# 1. Pfade definieren
# -------------------------------------------------------------------
BASE_DATA_DIR = "/Users/stefanbechthold/statistik_business_analytics/data"
GRAPHIC_DIR = "/Users/stefanbechthold/statistik_business_analytics/Grafiken/01_EDA_und_Preprocessing"
GRAPHIC_PATH = os.path.join(GRAPHIC_DIR, "02_korrelations_heatmap.png")

os.makedirs(GRAPHIC_DIR, exist_ok=True)

# -------------------------------------------------------------------
# 2. Daten laden und verknüpfen (Mergen)
# -------------------------------------------------------------------
print("Lade Datensätze...")
# Wir laden drei separate Tabellen, weil die Infos, die wir vergleichen wollen, verstreut sind.
df_items = pd.read_csv(os.path.join(BASE_DATA_DIR, "olist_order_items_dataset.csv"))
df_payments = pd.read_csv(os.path.join(BASE_DATA_DIR, "olist_order_payments_dataset.csv"))
df_reviews = pd.read_csv(os.path.join(BASE_DATA_DIR, "olist_order_reviews_dataset.csv"))

print("Verknüpfe (Merge) die Daten...")
# pd.merge verbindet Tabellen. 'on="order_id"' sagt Python, dass die Bestellnummer der Schlüssel ist.
# Wir nehmen nur die erste Bewertung pro Bestellung (drop_duplicates), um Verzerrungen zu vermeiden.
df_reviews_unique = df_reviews.drop_duplicates(subset=['order_id'])

# Erst Items mit Payments mergen...
df_merged = pd.merge(df_items, df_payments, on="order_id", how="inner")
# ... dann die Reviews hinzufügen.
df_merged = pd.merge(df_merged, df_reviews_unique, on="order_id", how="inner")

# -------------------------------------------------------------------
# 3. Relevante Variablen auswählen & Korrelation berechnen
# -------------------------------------------------------------------
# Wir bauen uns einen kleineren DataFrame nur mit den Spalten, die für unsere Analyse Sinn machen (numerische Werte).
cols_of_interest = [
    'price',                # Produktpreis
    'freight_value',        # Versandkosten
    'payment_value',        # Gesamtbezahlter Betrag
    'payment_installments', # Anzahl der Ratenzahlungen
    'review_score'          # Sterne-Bewertung (1-5)
]

df_analysis = df_merged[cols_of_interest]

# Fehlende Werte droppen, damit die Mathematik sauber funktioniert
df_analysis = df_analysis.dropna()

print("\nBerechne Korrelationsmatrix...")
# .corr() berechnet automatisch die Pearson-Korrelation zwischen allen ausgewählten Spalten.
# Die Werte liegen immer zwischen -1 (perfekt negativ) und 1 (perfekt positiv). 0 heißt kein linearer Zusammenhang.
corr_matrix = df_analysis.corr()

# -------------------------------------------------------------------
# 4. Heatmap visualisieren und speichern
# -------------------------------------------------------------------
print("Erstelle wunderschöne Heatmap...")

# Wir erstellen die "Leinwand"
plt.figure(figsize=(10, 8))

# sns.heatmap zeichnet die Farbmatrix.
# annot=True schreibt die genauen Zahlen in die Kästchen.
# cmap="coolwarm" ist ein Farbschema: Blau für negativ, Rot für positiv.
# vmin=-1, vmax=1 stellt sicher, dass die Farbskala exakt die statistischen Grenzen abbildet.
# fmt=".2f" rundet die Zahlen auf zwei Nachkommastellen.
sns.heatmap(
    corr_matrix, 
    annot=True, 
    cmap="coolwarm", 
    vmin=-1, 
    vmax=1, 
    center=0,
    square=True, 
    linewidths=.5, 
    fmt=".2f",
    cbar_kws={"shrink": .8} # Macht die Farblegende rechts etwas hübscher
)

# Titel hinzufügen
plt.title("Korrelations-Heatmap: Preis, Versand, Raten & Bewertung\n(Olist E-Commerce)", fontsize=16, pad=20)

# Layout anpassen und speichern
plt.tight_layout()
plt.savefig(GRAPHIC_PATH, dpi=300)
print(f"Erfolg! Grafik gespeichert unter: {GRAPHIC_PATH}")

plt.close()
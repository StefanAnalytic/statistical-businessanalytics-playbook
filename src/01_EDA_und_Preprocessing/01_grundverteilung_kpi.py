"""
Griff 1: Grundverteilung einer KPI (Deskriptive Statistik & EDA)

Was macht dieses Skript?
1. Es lädt echte E-Commerce-Daten (Zahlungswerte von Bestellungen).
2. Es prüft, ob Daten fehlen (Missingness Report).
3. Es berechnet die wichtigsten statistischen Kennzahlen (Mittelwert, Median, Varianz, Quantile).
4. Es erstellt eine wunderschöne Kombination aus Histogramm und Boxplot, um die Verteilung und Ausreißer visuell darzustellen, und speichert diese als Bild.

Warum ist das nützlich?
Dies ist IMMER der erste Schritt in jedem Datenprojekt. Bevor man Vorhersagen trifft, 
muss man verstehen, wie die Daten "normalerweise" aussehen und ob es extreme Ausreißer gibt.
"""

import pandas as pd          # pandas ist unser Werkzeug für Tabellen (wie Excel in Python)
import numpy as np           # numpy hilft bei mathematischen Berechnungen
import matplotlib.pyplot as plt # matplotlib ist für das Grundgerüst der Grafiken zuständig
import seaborn as sns        # seaborn macht die Grafiken wunderschön und modern
import os                    # os hilft uns, mit Dateipfaden auf dem Computer zu arbeiten

# -------------------------------------------------------------------
# 1. Pfade definieren (Wo liegen die Daten und wo soll das Bild hin?)
# -------------------------------------------------------------------
# Wir nutzen exakt deine Ordnerstruktur.
DATA_PATH = "/Users/stefanbechthold/statistik_business_analytics/data/olist_order_payments_dataset.csv"
GRAPHIC_DIR = "/Users/stefanbechthold/statistik_business_analytics/Grafiken/01_EDA_und_Preprocessing"
GRAPHIC_PATH = os.path.join(GRAPHIC_DIR, "01_kpi_verteilung.png")

# Stelle sicher, dass der Grafik-Ordner existiert. Wenn nicht, erstellt Python ihn heimlich.
os.makedirs(GRAPHIC_DIR, exist_ok=True)

# -------------------------------------------------------------------
# 2. Daten laden
# -------------------------------------------------------------------
print("Lade Daten...")
# pd.read_csv liest die CSV-Datei ein und speichert sie in einer Variablen namens 'df' (DataFrame)
df = pd.read_csv(DATA_PATH)

# Wir konzentrieren uns auf die Spalte 'payment_value' (der bezahlte Betrag).
kpi_series = df['payment_value']

# -------------------------------------------------------------------
# 3. Fehlende Werte prüfen (Missingness Report)
# -------------------------------------------------------------------
# .isnull() prüft für jede Zeile, ob der Wert fehlt (True) oder da ist (False).
# .sum() zählt alle "True"s zusammen.
missing_values = kpi_series.isnull().sum()
total_values = len(kpi_series)
print(f"\n--- Datenqualitäts-Check ---")
print(f"Fehlende Werte im 'payment_value': {missing_values} von {total_values} Zeilen.")

# Um saubere Statistiken zu berechnen, werfen wir (nur für diese Analyse) fehlende Werte raus.
kpi_clean = kpi_series.dropna()

# -------------------------------------------------------------------
# 4. Deskriptive Statistik berechnen
# -------------------------------------------------------------------
print("\n--- Deskriptive Statistik (Zusammenfassung) ---")

# Der Durchschnitt aller Werte. Sehr anfällig für extreme Ausreißer (z.B. ein 10.000€ Kauf).
mean_val = np.mean(kpi_clean)
print(f"Mittelwert (Mean): {mean_val:.2f}")

# Der Wert genau in der Mitte, wenn man alle Käufe dem Preis nach aufreiht. Robust gegen Ausreißer.
median_val = np.median(kpi_clean)
print(f"Median (50% Quantil): {median_val:.2f}")

# Die Varianz zeigt, wie stark die Werte um den Mittelwert streuen. 
# (Hohe Varianz = Kunden geben extrem unterschiedliche Beträge aus).
variance_val = np.var(kpi_clean, ddof=1) 
print(f"Varianz: {variance_val:.2f}")

# Quantile: Wo liegen die Grenzen für die unteren 25% und oberen 75% der Daten?
q25 = np.percentile(kpi_clean, 25)
q75 = np.percentile(kpi_clean, 75)
print(f"25% der Kunden zahlen weniger als: {q25:.2f}")
print(f"75% der Kunden zahlen weniger als: {q75:.2f}")

# -------------------------------------------------------------------
# 5. Visualisierung erstellen und speichern
# -------------------------------------------------------------------
print("\nErstelle wunderschöne Grafik...")

# Wir filtern extreme Ausreißer nur für die Visualisierung heraus, damit man überhaupt etwas erkennt.
# Wir zeigen nur Bestellungen unter 1000 an. (Über 1000 staucht die Grafik sonst extrem).
plot_data = kpi_clean[kpi_clean < 1000]

# Wir bereiten ein "Canvas" (Leinwand) vor mit 2 untereinander liegenden Diagrammen.
# Das obere wird schmal (Höhenverhältnis 15%), das untere groß (85%).
fig, (ax_box, ax_hist) = plt.subplots(
    nrows=2, 
    sharex=True, 
    gridspec_kw={"height_ratios": (.15, .85)}, 
    figsize=(10, 6)
)

# Oberes Diagramm: Ein Boxplot. Er zeigt Median, Quartile und Ausreißer als Punkte.
sns.boxplot(x=plot_data, ax=ax_box, color="#1f77b4")
ax_box.set(xlabel='') # Keine Beschriftung für die obere X-Achse
ax_box.set_title("Verteilung des Bestellwerts (Payment Value) in BRL - Olist E-Commerce", fontsize=14, pad=15)

# Unteres Diagramm: Ein Histogramm (Balkendiagramm der Häufigkeiten) plus eine geglättete Kurve (KDE).
sns.histplot(x=plot_data, ax=ax_hist, bins=50, kde=True, color="#1f77b4")
ax_hist.set_xlabel("Bestellwert", fontsize=12)
ax_hist.set_ylabel("Anzahl der Bestellungen", fontsize=12)

# Vertikale Linien für Mittelwert und Median im Histogramm einzeichnen
ax_hist.axvline(mean_val, color='r', linestyle='--', linewidth=2, label=f'Mean: {mean_val:.2f}')
ax_hist.axvline(median_val, color='g', linestyle='-', linewidth=2, label=f'Median: {median_val:.2f}')
ax_hist.legend()

# Macht das Layout schön kompakt
sns.despine(ax=ax_hist)
sns.despine(ax=ax_box, left=True)
ax_box.set(yticks=[]) # Entfernt die Zahlen auf der Y-Achse des Boxplots

# Grafik als PNG-Datei speichern
plt.tight_layout()
plt.savefig(GRAPHIC_PATH, dpi=300) # dpi=300 sorgt für eine scharfe, hochauflösende Qualität
print(f"Erfolg! Grafik gespeichert unter: {GRAPHIC_PATH}")

# Optional: plt.show() würde das Bild auch direkt im Fenster anzeigen, 
# aber wir wollen es ja primär im Ordner speichern.
plt.close()
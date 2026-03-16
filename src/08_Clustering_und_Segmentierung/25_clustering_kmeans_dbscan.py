"""
Griff 25: Kundengruppen segmentieren? (Clustering: K-Means & RFM-Analyse)

Was ist das und was macht das Skript?
In den bisherigen Modellen (Regression) hatten wir immer eine klare Zielvariable (z.B. Umsatz).
Beim Clustering arbeiten wir "Unsupervised" (Unüberwacht) – wir haben kein Ziel, sondern 
wollen, dass der Algorithmus selbstständig Muster und Gruppen in den Daten findet.

Der bekannteste Algorithmus dafür ist K-Means. Er sucht nach "k" Zentren in einer Datenwolke 
und ordnet jeden Kunden dem nächstgelegenen Zentrum zu. 

Business Case in diesem Skript:
Die Königsklasse im E-Commerce: Die RFM-Segmentierung (Recency, Frequency, Monetary).
Wir berechnen für jeden Kunden:
- Recency: Wie viele Tage ist der letzte Kauf her?
- Frequency: Wie oft hat er insgesamt bestellt?
- Monetary: Wie viel Geld hat er insgesamt ausgegeben?

Anschließend lassen wir K-Means 4 Kundensegmente bilden. Durch die Analyse der Cluster-Mittelwerte 
(Profiling) können wir den Gruppen Business-Namen geben: z.B. "VIPs", "Schläfer" oder "Einmalkäufer".
Das ist die perfekte Basis für zielgerichtete E-Mail-Marketing-Kampagnen!
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# -------------------------------------------------------------------
# 1. Setup & Pfade
# -------------------------------------------------------------------
# Wir wechseln in den Ordner "08_Clustering_und_Segmentierung"
ORDERS_PATH = "/Users/stefanbechthold/statistik_business_analytics/data/olist_orders_dataset.csv"
PAYMENTS_PATH = "/Users/stefanbechthold/statistik_business_analytics/data/olist_order_payments_dataset.csv"
CUSTOMERS_PATH = "/Users/stefanbechthold/statistik_business_analytics/data/olist_customers_dataset.csv"

GRAPHIC_DIR = "/Users/stefanbechthold/statistik_business_analytics/Grafiken/08_Clustering_und_Segmentierung"
GRAPHIC_PATH = os.path.join(GRAPHIC_DIR, "25_kmeans_rfm_segmentierung.png")

os.makedirs(GRAPHIC_DIR, exist_ok=True)

def main():
    print("🚀 Starte Griff 25: K-Means Clustering & RFM Kundensegmentierung...")
    
    # -------------------------------------------------------------------
    # 2. Daten laden & RFM-Tabelle (Feature Engineering) aufbauen
    # -------------------------------------------------------------------
    print("Lade Daten und berechne Recency, Frequency, Monetary...")
    df_orders = pd.read_csv(ORDERS_PATH)
    df_payments = pd.read_csv(PAYMENTS_PATH)
    df_customers = pd.read_csv(CUSTOMERS_PATH)
    
    # Datum konvertieren
    df_orders['purchase_date'] = pd.to_datetime(df_orders['order_purchase_timestamp']).dt.date
    
    # Zahlungen pro Bestellung aufsummieren
    order_spend = df_payments.groupby('order_id')['payment_value'].sum().reset_index()
    
    # Mergen mit Customers, um die 'customer_unique_id' (den echten Menschen) zu bekommen
    df_merged = pd.merge(df_orders, df_customers[['customer_id', 'customer_unique_id']], on='customer_id')
    df_merged = pd.merge(df_merged, order_spend, on='order_id')
    
    # Wir tun so, als wäre "Heute" der Tag nach der allerletzten Bestellung im Datensatz
    NOW = df_merged['purchase_date'].max() + pd.Timedelta(days=1)
    
    # RFM Berechnung pro Kunde
    rfm = df_merged.groupby('customer_unique_id').agg({
        'purchase_date': lambda x: (NOW - x.max()).days, # Recency: Tage seit letztem Kauf
        'order_id': 'count',                             # Frequency: Anzahl der Bestellungen
        'payment_value': 'sum'                           # Monetary: Gesamtumsatz
    }).reset_index()
    
    rfm.columns = ['customer_unique_id', 'Recency', 'Frequency', 'Monetary']
    
    # Um die extreme Schiefe zu mildern, begrenzen wir den Datensatz auf normale Kunden 
    # (schneiden die extremsten 1% der Ausreißer ab), damit die Cluster nicht verzerrt werden.
    rfm = rfm[rfm['Monetary'] < rfm['Monetary'].quantile(0.99)]
    
    print(f"RFM-Tabelle erstellt für {len(rfm)} Kunden.")

    # -------------------------------------------------------------------
    # 3. Datenvorbereitung (Log-Transformation & Skalierung)
    # -------------------------------------------------------------------
    print("\nSkaliere Daten für den K-Means Algorithmus...")
    # K-Means misst "Distanzen". Wenn Recency in Hunderten (Tagen) und Frequency in 
    # kleinen Zahlen (1-5) gemessen wird, dominiert Recency das Clustering komplett.
    # WICHTIG: Distanzbasierte Algorithmen MÜSSEN zwingend skaliert werden!
    
    # Wir logarithmieren Monetary und Frequency, da diese oft extrem rechtsschief sind
    rfm_log = rfm[['Recency', 'Frequency', 'Monetary']].copy()
    rfm_log['Monetary'] = np.log1p(rfm_log['Monetary'])
    rfm_log['Frequency'] = np.log1p(rfm_log['Frequency'])
    
    # StandardScaler zentriert die Daten auf Mittelwert 0 und Standardabweichung 1
    scaler = StandardScaler()
    rfm_scaled = scaler.fit_transform(rfm_log)

    # -------------------------------------------------------------------
    # 4. K-Means Clustering durchführen
    # -------------------------------------------------------------------
    print("Führe K-Means Clustering mit 4 Clustern durch...")
    n_clusters = 4
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    
    # Fit und Vorhersage der Cluster-Zugehörigkeit (0 bis 3)
    rfm['Cluster'] = kmeans.fit_predict(rfm_scaled)
    
    # -------------------------------------------------------------------
    # 5. Cluster Profiling (Business-Interpretation)
    # -------------------------------------------------------------------
    # Wir schauen uns die durchschnittlichen unskalierten RFM-Werte pro Cluster an,
    # um zu verstehen, WER diese Leute sind.
    cluster_profiles = rfm.groupby('Cluster').agg({
        'Recency': 'mean',
        'Frequency': 'mean',
        'Monetary': ['mean', 'count']
    }).round(1)
    
    # Spaltennamen vereinfachen
    cluster_profiles.columns = ['Recency_Mean', 'Frequency_Mean', 'Monetary_Mean', 'Kunden_Anzahl']
    
    # Wir sortieren die Cluster nach Wertigkeit (Monetary) für ein schöneres Plotting
    cluster_profiles = cluster_profiles.sort_values(by='Monetary_Mean', ascending=False)
    
    # Business-Namen zuweisen (basierend auf unserer manuellen Logik)
    # Da K-Means die Nummern (0-3) zufällig vergibt, mappen wir sie anhand der Sortierung.
    ordered_clusters = cluster_profiles.index.tolist()
    cluster_names = {
        ordered_clusters[0]: 'Premium/VIPs',       # Höchster Umsatz
        ordered_clusters[1]: 'Loyale Kunden',      # Zweithöchster
        ordered_clusters[2]: 'Neue/Geringwertige', # Dritthöchster (oder kürzlich gekauft)
        ordered_clusters[3]: 'Verlorene (Churn)'   # Niedrigster Umsatz / Längste Recency
    }
    
    rfm['Segment'] = rfm['Cluster'].map(cluster_names)
    print("\nCluster-Profile (Business-Sicht):")
    print(cluster_profiles)

    # -------------------------------------------------------------------
    # 6. Wunderschöne Visualisierung (Scatterplot & Profiling-Bars)
    # -------------------------------------------------------------------
    print("\nGeneriere Visualisierungs-Grid...")
    sns.set_theme(style="whitegrid")
    
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    
    # Farbpalette für die 4 Segmente
    palette = {'Premium/VIPs': '#9b59b6', 'Loyale Kunden': '#3498db', 
               'Neue/Geringwertige': '#2ecc71', 'Verlorene (Churn)': '#e74c3c'}

    # --- Plot 1: 2D Scatterplot (Recency vs. Monetary) ---
    # Da wir 3 Dimensionen haben, plotten wir die zwei wichtigsten für den Umsatz:
    # Wann war der letzte Kauf (X) und wie viel wurde ausgegeben (Y)?
    
    sns.scatterplot(data=rfm, x='Recency', y='Monetary', hue='Segment', 
                    palette=palette, alpha=0.5, s=20, ax=axes[0])
    
    axes[0].set_title("1. Die Kundensegmente in der Datenwolke", fontsize=14, fontweight="bold")
    axes[0].set_xlabel("Recency (Tage seit dem letzten Kauf)")
    axes[0].set_ylabel("Monetary (Gesamtumsatz in BRL)")
    
    # Wir drehen die X-Achse gedanklich um, da eine KLEINE Recency (vor kurzem gekauft) BESSER ist!
    axes[0].invert_xaxis()
    axes[0].legend(loc='upper left', title="Kundensegment")

    # --- Plot 2: Cluster-Profile (Bar Charts der Mediane) ---
    # Wir visualisieren die durchschnittlichen Eigenschaften jedes Segments, 
    # damit das Marketing-Team sofort weiß, wie man diese Leute anspricht.
    
    # Wir bereiten die Daten für einen Multi-Bar-Plot vor (Melt)
    profiling_data = rfm.groupby('Segment')[['Recency', 'Monetary']].median().reset_index()
    
    # Skalierung nur für den Plot, damit Recency (Tage) und Monetary (BRL) in ein Chart passen
    profiling_data['Monetary'] = profiling_data['Monetary'] / profiling_data['Monetary'].max()
    profiling_data['Recency'] = profiling_data['Recency'] / profiling_data['Recency'].max()
    
    profiling_melted = pd.melt(profiling_data, id_vars='Segment', var_name='Metrik', value_name='Relativer Wert (skaliert)')
    
    sns.barplot(data=profiling_melted, x='Segment', y='Relativer Wert (skaliert)', hue='Metrik', 
                palette=['#e67e22', '#2c3e50'], ax=axes[1], edgecolor='black')
    
    axes[1].set_title("2. Segment-Profile (Relativer Vergleich)", fontsize=14, fontweight="bold")
    axes[1].set_xlabel("")
    axes[1].set_ylabel("Relativer Wert (0 = Niedrig, 1 = Segment-Maximum)")
    axes[1].tick_params(axis='x', rotation=15)
    
    # Info-Box in den Plot einfügen
    info_text = (
        "Actionable Insights für Marketing:\n"
        "-----------------------------------\n"
        "Premium/VIPs: Treueprogramm anbieten!\n"
        "Loyale Kunden: Cross-Selling & Upselling.\n"
        "Neue Kunden: Onboarding-Rabatt geben.\n"
        "Verlorene: Reaktivierungs-Kampagne (E-Mail)."
    )
    props = dict(boxstyle='round,pad=0.8', facecolor='white', alpha=0.9, edgecolor='gray', linewidth=1.5)
    axes[1].text(0.95, 0.95, info_text, transform=axes[1].transAxes, fontsize=11,
                 verticalalignment='top', horizontalalignment='right', bbox=props, color='#2c3e50')

    # Layout optimieren und speichern
    plt.suptitle("Unsupervised Machine Learning: RFM Kundensegmentierung mit K-Means", 
                 fontsize=18, fontweight="bold", y=1.05)
    plt.tight_layout()
    sns.despine()
    
    plt.savefig(GRAPHIC_PATH, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✅ Grafik erfolgreich gespeichert unter:\n{GRAPHIC_PATH}")

if __name__ == "__main__":
    main() 
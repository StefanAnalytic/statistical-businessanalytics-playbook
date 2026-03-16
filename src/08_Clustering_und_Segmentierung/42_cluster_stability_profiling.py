"""
Griff 42: Segment-Stabilität & Profiling? (Cluster Profiling & Business Labeling)

Was ist das und was macht das Skript?
Clustering-Algorithmen (wie K-Means) finden mathematische Gruppen in Daten. Das Problem: 
Algorithmen spucken nur Zahlen aus (z.B. "Cluster 0", "Cluster 1"). Für das Business 
sind diese Zahlen wertlos, solange wir sie nicht interpretieren ("Profiling") und 
mit sprechenden Namen versehen ("Business Labeling").

Zusätzlich müssen wir prüfen, ob die gefundenen Segmente mathematisch Sinn ergeben 
und stabil sind. Dafür nutzen wir den Silhouette Score (Werte von -1 bis 1, wobei 
Werte > 0.5 auf gut getrennte Cluster hindeuten).

Business Case in diesem Skript:
Wir wollen die Olist-Kunden segmentieren, um gezielte Marketing-Aktionen zu steuern. 
Da Olist kaum Wiederholungskäufer hat, nutzen wir nicht das klassische RFM-Modell, 
sondern verhaltensbasierte Features:
1. Total Spend (Monetary - Wie viel wurde ausgegeben?)
2. Review Score (Satisfaction - Wie glücklich ist der Kunde?)
3. Delivery Days (Experience - Wie lange musste der Kunde warten?)

Wir finden 4 Cluster, bewerten deren Stabilität und leiten aus den Durchschnittswerten 
echte Business Personas ab (z.B. "Wütende, wartende Kunden" oder "Zufriedene VIPs").
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score

# -------------------------------------------------------------------
# 1. Setup & Pfade
# -------------------------------------------------------------------
ORDERS_PATH = "/Users/stefanbechthold/statistik_business_analytics/data/olist_orders_dataset.csv"
REVIEWS_PATH = "/Users/stefanbechthold/statistik_business_analytics/data/olist_order_reviews_dataset.csv"
PAYMENTS_PATH = "/Users/stefanbechthold/statistik_business_analytics/data/olist_order_payments_dataset.csv"
CUSTOMERS_PATH = "/Users/stefanbechthold/statistik_business_analytics/data/olist_customers_dataset.csv"

GRAPHIC_DIR = "/Users/stefanbechthold/statistik_business_analytics/Grafiken/08_Clustering_und_Segmentierung"
GRAPHIC_PATH = os.path.join(GRAPHIC_DIR, "42_cluster_profiling_labeling.png")

os.makedirs(GRAPHIC_DIR, exist_ok=True)

def main():
    print("Starte Griff 42: Customer Segmentation (Profiling & Business Labeling)...")
    
    # -------------------------------------------------------------------
    # 2. Daten laden & Customer Features bauen
    # -------------------------------------------------------------------
    df_orders = pd.read_csv(ORDERS_PATH)
    df_reviews = pd.read_csv(REVIEWS_PATH)
    df_payments = pd.read_csv(PAYMENTS_PATH)
    df_customers = pd.read_csv(CUSTOMERS_PATH)
    
    # Lieferzeit berechnen
    df_orders['purchase_time'] = pd.to_datetime(df_orders['order_purchase_timestamp'], errors='coerce')
    df_orders['delivered_time'] = pd.to_datetime(df_orders['order_delivered_customer_date'], errors='coerce')
    df_orders['delivery_days'] = (df_orders['delivered_time'] - df_orders['purchase_time']).dt.total_seconds() / (24 * 3600)
    
    # Erste Bewertung pro Bestellung
    reviews_unique = df_reviews.drop_duplicates(subset=['order_id'])[['order_id', 'review_score']]
    
    # Gesamtwert der Bestellung
    payments_agg = df_payments.groupby('order_id')['payment_value'].sum().reset_index()
    
    # Mergen zu einem Order-Datensatz
    df_merged = pd.merge(df_orders[['order_id', 'customer_id', 'delivery_days']], payments_agg, on='order_id', how='inner')
    df_merged = pd.merge(df_merged, reviews_unique, on='order_id', how='inner')
    
    # Mit Customer-ID verknüpfen und auf Kundenebene aggregieren
    df_cust_merged = pd.merge(df_merged, df_customers[['customer_id', 'customer_unique_id']], on='customer_id', how='inner')
    
    customer_features = df_cust_merged.groupby('customer_unique_id').agg({
        'payment_value': 'sum',           # Total Spend
        'review_score': 'mean',           # Avg Satisfaction
        'delivery_days': 'mean'           # Avg Wait Time
    }).reset_index()
    
    df_clean = customer_features.dropna().copy()
    
    # Extreme Ausreißer entfernen (z.B. > 1000 BRL oder > 60 Tage Lieferzeit), damit die Cluster nicht verzerrt werden
    df_clean = df_clean[(df_clean['payment_value'] < 1000) & (df_clean['delivery_days'] < 60) & (df_clean['delivery_days'] > 0)]
    
    # Sample ziehen für überschaubare Rechenzeit beim Silhouette Score (skaliert quadratisch)
    df_sample = df_clean.sample(n=10000, random_state=42).reset_index(drop=True)
    
    features = ['payment_value', 'review_score', 'delivery_days']
    X = df_sample[features]

    # -------------------------------------------------------------------
    # 3. Skalierung & Clustering
    # -------------------------------------------------------------------
    # K-Means nutzt Distanzen, daher MÜSSEN alle Features auf die gleiche Skala gebracht werden
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Wir legen uns für diesen Business Case auf 4 Segmente fest
    n_clusters = 4
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    df_sample['Cluster'] = kmeans.fit_predict(X_scaled)
    
    # Silhouette Score berechnen (Güte des Clusterings)
    sil_score = silhouette_score(X_scaled, df_sample['Cluster'])

    # -------------------------------------------------------------------
    # 4. Profiling & Business Labeling
    # -------------------------------------------------------------------
    # Durchschnittswerte pro Cluster berechnen (Das echte "Profil")
    cluster_profiles = df_sample.groupby('Cluster')[features].mean()
    
    # Wir ordnen basierend auf den Mittelwerten Business-Namen zu
    # Anmerkung: Diese Logik ist heuristisch und passend zum Olist-Datensatz geschrieben
    def assign_business_label(row):
        if row['review_score'] > 4.5 and row['payment_value'] > 250:
            return "Happy VIPs (High Value)"
        elif row['review_score'] < 3.0 and row['delivery_days'] > 20:
            return "Wütend & Wartend (Churn Risk)"
        elif row['review_score'] > 4.0 and row['payment_value'] <= 150:
            return "Standard Zufrieden (Low Value)"
        else:
            return "Neutrale Mitte"

    cluster_profiles['Business_Persona'] = cluster_profiles.apply(assign_business_label, axis=1)
    
    # Label an den Haupt-Datensatz mergen
    label_dict = cluster_profiles['Business_Persona'].to_dict()
    df_sample['Persona'] = df_sample['Cluster'].map(label_dict)

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

    # --- Plot 1: Scatterplot der Segmente (Spend vs Satisfaction) ---
    sns.scatterplot(x='payment_value', y='review_score', hue='Persona', 
                    data=df_sample, palette='Set2', alpha=0.6, s=25, edgecolor='none', ax=ax1)
    
    ax1.set_title(f"1. Customer Segments (Silhouette Score: {sil_score:.2f})", fontsize=14, fontweight='bold', color='white')
    ax1.set_xlabel("Total Spend (BRL)")
    ax1.set_ylabel("Average Review Score (Sterne)")
    
    # Y-Achse leicht anpassen wegen diskreter Stern-Bewertungen
    ax1.set_ylim(0.5, 5.5)
    
    legend1 = ax1.legend(title="Business Persona", loc='lower right', facecolor='#1e1e1e', edgecolor='#333333')
    plt.setp(legend1.get_title(), color='white')
    plt.setp(legend1.get_texts(), color='white')

    # --- Plot 2: Cluster Profiling (Heatmap der standardisierten Mittelwerte) ---
    # Um die Cluster gut zu vergleichen, zeigen wir die standardisierten Zentren in einer Heatmap.
    # Rot = Wert ist überdurchschnittlich hoch, Blau = Wert ist unterdurchschnittlich
    centers_df = pd.DataFrame(kmeans.cluster_centers_, columns=['Spend (BRL)', 'Zufriedenheit', 'Lieferzeit'])
    centers_df.index = cluster_profiles['Business_Persona']
    
    sns.heatmap(centers_df, cmap='coolwarm', annot=True, fmt=".2f", linewidths=.5, 
                ax=ax2, cbar_kws={'label': 'Z-Score (Standardabweichungen vom Mittelwert)'})
    
    ax2.set_title("2. Cluster Profiling (Was zeichnet die Gruppen aus?)", fontsize=14, fontweight='bold', color='white')
    ax2.set_ylabel("")
    ax2.tick_params(axis='y', rotation=0)

    # Info-Box
    info_text = (
        "Insight (Von Mathematik zu Business):\n"
        "-------------------------------------\n"
        "K-Means liefert nur rohe Z-Scores (Heatmap). Das Profiling\n"
        "übersetzt dies: Die Gruppe 'Wütend & Wartend' (Rot in der Heatmap\n"
        "bei Lieferzeit, Blau bei Zufriedenheit) ist hochgradig frustriert,\n"
        "weil Pakete zu spät kamen. Die 'Happy VIPs' geben am meisten Geld\n"
        "aus und vergeben Top-Bewertungen.\n\n"
        "Aktion: VIPs mit Treueprogramm binden, Wütende aktiv kontaktieren!"
    )
    props = dict(boxstyle='round,pad=1', facecolor='#1e1e1e', alpha=0.9, edgecolor='#3498db', linewidth=1.5)
    # Positionierung unterhalb des Scatterplots
    ax1.text(0.05, 0.05, info_text, transform=ax1.transAxes, fontsize=11,
             verticalalignment='bottom', bbox=props, color='white')

    plt.suptitle("Customer Segmentation: Profiling, Stabilität & Business Labeling", 
                 fontsize=18, fontweight='bold', color='white', y=1.05)
    plt.tight_layout()
    
    # Speichern
    plt.savefig(GRAPHIC_PATH, dpi=300, bbox_inches='tight', facecolor=fig.get_facecolor())
    plt.close()
    
    print(f"✅ Grafik erfolgreich gespeichert unter:\n{GRAPHIC_PATH}")

if __name__ == "__main__":
    main()
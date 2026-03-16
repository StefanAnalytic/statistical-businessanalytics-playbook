"""
Griff 59: Räumliche Daten nutzen? (Geospatial Analytics & Haversine Distance)

Was ist das und was macht das Skript?
In der Logistik und im E-Commerce reichen einfache Bundesstaaten (State) oft nicht aus, 
um Kosten und Risiken zu bewerten. Ein Bundesstaat in Brasilien (z.B. Amazonas) ist 
größer als Mitteleuropa! Wir brauchen die exakte physische Distanz.

Geospatial Analytics nutzt Koordinaten (Latitude & Longitude), um räumliche Muster 
zu analysieren. Da die Erde eine Kugel ist, können wir nicht einfach den Satz des 
Pythagoras für Distanzen nutzen. Stattdessen verwenden wir die "Haversine-Formel", 
die die Großkreis-Entfernung (kürzeste Strecke auf einer Kugeloberfläche) zwischen 
zwei Punkten exakt in Kilometern berechnet.

Business Case in diesem Skript:
Wir wollen wissen: Wie stark treibt die exakte Luftlinien-Entfernung zwischen dem 
Verkäufer (Seller) und dem Kunden (Customer) die Frachtkosten (Freight Value) in die Höhe? 
Und ab wie vielen Kilometern kippt das System und die Lieferungen kommen zu spät? 
Dazu verknüpfen wir die Postleitzahlen mit den Geo-Koordinaten von Olist und 
berechnen die Haversine-Distanz für jede einzelne Bestellung.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# -------------------------------------------------------------------
# 1. Setup & Pfade
# -------------------------------------------------------------------
ORDERS_PATH = "/Users/stefanbechthold/statistik_business_analytics/data/olist_orders_dataset.csv"
ITEMS_PATH = "/Users/stefanbechthold/statistik_business_analytics/data/olist_order_items_dataset.csv"
CUSTOMERS_PATH = "/Users/stefanbechthold/statistik_business_analytics/data/olist_customers_dataset.csv"
SELLERS_PATH = "/Users/stefanbechthold/statistik_business_analytics/data/olist_sellers_dataset.csv"
GEO_PATH = "/Users/stefanbechthold/statistik_business_analytics/data/olist_geolocation_dataset.csv"

GRAPHIC_DIR = "/Users/stefanbechthold/statistik_business_analytics/Grafiken/12_Advanced_Analytics_Spezial"
GRAPHIC_PATH = os.path.join(GRAPHIC_DIR, "59_geospatial_haversine_distance.png")

os.makedirs(GRAPHIC_DIR, exist_ok=True)

def haversine_distance(lat1, lon1, lat2, lon2):
    """
    Berechnet die Distanz zwischen zwei Punkten auf der Erde in Kilometern.
    Erwartet Arrays oder Pandas Series mit Dezimalgraden.
    """
    R = 6371.0 # Erdradius in Kilometern

    # Grad in Bogenmaß (Radians) umrechnen
    lat1_rad, lon1_rad = np.radians(lat1), np.radians(lon1)
    lat2_rad, lon2_rad = np.radians(lat2), np.radians(lon2)

    dlon = lon2_rad - lon1_rad
    dlat = lat2_rad - lat1_rad

    # Haversine Formel
    a = np.sin(dlat / 2.0)**2 + np.cos(lat1_rad) * np.cos(lat2_rad) * np.sin(dlon / 2.0)**2
    c = 2 * np.arcsin(np.sqrt(a))
    distance = R * c
    
    return distance

def main():
    print("🚀 Starte Griff 59: Geospatial Analytics & Haversine Distance...")
    
    # -------------------------------------------------------------------
    # 2. Daten & Geo-Koordinaten laden
    # -------------------------------------------------------------------
    df_orders = pd.read_csv(ORDERS_PATH)
    df_items = pd.read_csv(ITEMS_PATH)
    df_customers = pd.read_csv(CUSTOMERS_PATH)
    df_sellers = pd.read_csv(SELLERS_PATH)
    df_geo = pd.read_csv(GEO_PATH)
    
    # Geo-Daten bereinigen (Eine PLZ kann mehrere Einträge haben -> Wir nehmen den Mittelpunkt)
    geo_clean = df_geo.groupby('geolocation_zip_code_prefix').agg({
        'geolocation_lat': 'mean',
        'geolocation_lng': 'mean'
    }).reset_index()
    
    # -------------------------------------------------------------------
    # 3. Master-Tabelle mit Start- und Ziel-Koordinaten bauen
    # -------------------------------------------------------------------
    # Frachtkosten aggregieren
    order_freight = df_items.groupby('order_id').agg({'freight_value': 'sum', 'seller_id': 'first'}).reset_index()
    
    df_merged = pd.merge(df_orders[['order_id', 'customer_id', 'order_status', 
                                    'order_delivered_customer_date', 'order_estimated_delivery_date']], 
                         order_freight, on='order_id', how='inner')
    
    # Kunden PLZ anspielen
    df_merged = pd.merge(df_merged, df_customers[['customer_id', 'customer_zip_code_prefix']], on='customer_id')
    
    # Seller PLZ anspielen
    df_merged = pd.merge(df_merged, df_sellers[['seller_id', 'seller_zip_code_prefix']], on='seller_id')
    
    # Kunden-Koordinaten (Ziel) mergen
    df_merged = pd.merge(df_merged, geo_clean, left_on='customer_zip_code_prefix', right_on='geolocation_zip_code_prefix', how='left')
    df_merged.rename(columns={'geolocation_lat': 'cust_lat', 'geolocation_lng': 'cust_lng'}, inplace=True)
    
    # Seller-Koordinaten (Start) mergen
    df_merged = pd.merge(df_merged, geo_clean, left_on='seller_zip_code_prefix', right_on='geolocation_zip_code_prefix', how='left')
    df_merged.rename(columns={'geolocation_lat': 'seller_lat', 'geolocation_lng': 'seller_lng'}, inplace=True)
    
    df_clean = df_merged.dropna(subset=['cust_lat', 'cust_lng', 'seller_lat', 'seller_lng', 'order_delivered_customer_date']).copy()

    # -------------------------------------------------------------------
    # 4. Feature Engineering: Haversine Distanz & Verspätung
    # -------------------------------------------------------------------
    # 1. Distanz berechnen (vektorisiert für Performance)
    df_clean['distance_km'] = haversine_distance(
        df_clean['seller_lat'], df_clean['seller_lng'],
        df_clean['cust_lat'], df_clean['cust_lng']
    )
    
    # 2. Verspätungs-Logik
    df_clean['delivered'] = pd.to_datetime(df_clean['order_delivered_customer_date'])
    df_clean['estimated'] = pd.to_datetime(df_clean['order_estimated_delivery_date'])
    df_clean['is_late'] = (df_clean['delivered'] > df_clean['estimated']).astype(int)
    
    # Extreme Ausreißer (Datenfehler) filtern (z.B. > 5000km innerhalb Brasiliens oder Fracht > 300)
    df_clean = df_clean[(df_clean['distance_km'] < 4000) & (df_clean['freight_value'] < 200)]
    
    # Sample ziehen für Visualisierung
    df_sample = df_clean.sample(n=15000, random_state=42).reset_index(drop=True)

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

    # --- Plot 1: Hexbin-Plot (Distanz vs. Frachtkosten) ---
    # Bei vielen Datenpunkten sind Scatterplots unlesbar ("Overplotting"). Hexbins zeigen die Dichte.
    hb = ax1.hexbin(df_sample['distance_km'], df_sample['freight_value'], gridsize=40, cmap='magma', mincnt=1)
    
    # Eine einfache Trendlinie zur Orientierung
    z = np.polyfit(df_sample['distance_km'], df_sample['freight_value'], 1)
    p = np.poly1d(z)
    ax1.plot(df_sample['distance_km'], p(df_sample['distance_km']), color='#2ecc71', linestyle='--', linewidth=2, label='Linearer Trend')
    
    cb = fig.colorbar(hb, ax=ax1)
    cb.set_label('Anzahl der Bestellungen', color='white')
    cb.ax.yaxis.set_tick_params(color='white')
    plt.setp(plt.getp(cb.ax.axes, 'yticklabels'), color='white')
    
    ax1.set_title("1. Pricing Dynamics: Frachtkosten vs. Luftlinien-Distanz", fontsize=14, fontweight='bold', color='white')
    ax1.set_xlabel("Haversine Distanz (Kilometer)")
    ax1.set_ylabel("Frachtkosten (BRL)")
    
    legend1 = ax1.legend(loc='upper right', facecolor='#1e1e1e', edgecolor='#333333')
    plt.setp(legend1.get_texts(), color='white')

    # Info-Box Plot 1
    info_1 = (
        "Geospatial Pricing Insight:\n"
        "---------------------------\n"
        "Der Hexbin-Plot zeigt den 'Hotspot' (helle Farben):\n"
        "Die meisten Pakete reisen nur sehr kurze Strecken (< 500km).\n"
        "Die Trendlinie (Grün) beweist: Jeder zusätzliche Kilometer\n"
        "kostet Geld. Aber wir sehen auch eine massive vertikale Streuung,\n"
        "was bedeutet, dass die Distanz allein nicht alles erklärt\n"
        "(Volumen/Gewicht spielen ebenfalls eine große Rolle)."
    )
    props1 = dict(boxstyle='round,pad=1', facecolor='#1e1e1e', alpha=0.9, edgecolor='#f1c40f', linewidth=1.5)
    ax1.text(0.45, 0.35, info_1, transform=ax1.transAxes, fontsize=11,
             verticalalignment='center', bbox=props1, color='white')

    # --- Plot 2: Logistik-Risiko (KDE Plot Distanzen nach Pünktlichkeit) ---
    sns.kdeplot(df_sample[df_sample['is_late'] == 0]['distance_km'], 
                color='#3498db', fill=True, alpha=0.5, label='Pünktlich (0)', ax=ax2, edgecolor='none')
    sns.kdeplot(df_sample[df_sample['is_late'] == 1]['distance_km'], 
                color='#e74c3c', fill=True, alpha=0.5, label='Verspätet (1)', ax=ax2, edgecolor='none')
    
    # Mediane einzeichnen
    median_ontime = df_sample[df_sample['is_late'] == 0]['distance_km'].median()
    median_late = df_sample[df_sample['is_late'] == 1]['distance_km'].median()
    
    ax2.axvline(median_ontime, color='#3498db', linestyle=':', linewidth=3, label=f'Median Pünktlich ({median_ontime:.0f} km)')
    ax2.axvline(median_late, color='#e74c3c', linestyle=':', linewidth=3, label=f'Median Verspätet ({median_late:.0f} km)')

    ax2.set_title("2. Delivery Risk: Verteilung der Entfernungen nach Pünktlichkeit", fontsize=14, fontweight='bold', color='white')
    ax2.set_xlabel("Haversine Distanz (Kilometer)")
    ax2.set_ylabel("Dichte (Relative Häufigkeit)")
    ax2.set_xlim(0, 3500)
    
    legend2 = ax2.legend(loc='upper right', facecolor='#1e1e1e', edgecolor='#333333')
    plt.setp(legend2.get_texts(), color='white')

    # Info-Box Plot 2
    info_2 = (
        "Business Action (Risk Management):\n"
        "----------------------------------\n"
        "Pünktliche Pakete (Blau) stauen sich stark im Nahbereich an.\n"
        "Die Verteilung der verspäteten Pakete (Rot) ist deutlich flacher\n"
        "und zieht sich weiter nach rechts aus (Fat Tail).\n\n"
        "Konsequenz: Ab ca. 1000 km steigt die Wahrscheinlichkeit für\n"
        "Logistik-Ausfälle massiv. Für diese Long-Distance-Routen sollten\n"
        "wir die Puffer-Zeiten in den Liefer-Schätzungen künstlich erhöhen!"
    )
    props2 = dict(boxstyle='round,pad=1', facecolor='#1e1e1e', alpha=0.9, edgecolor='#e74c3c', linewidth=1.5)
    ax2.text(0.35, 0.45, info_2, transform=ax2.transAxes, fontsize=11,
             verticalalignment='center', bbox=props2, color='white')

    plt.suptitle("Geospatial Analytics: Haversine-Distanz auf dem Erdball berechnen", 
                 fontsize=18, fontweight='bold', color='white', y=1.05)
    plt.tight_layout()
    
    # Speichern der Grafik
    plt.savefig(GRAPHIC_PATH, dpi=300, bbox_inches='tight', facecolor=fig.get_facecolor())
    plt.close()
    
    print(f"✅ Grafik erfolgreich gespeichert unter:\n{GRAPHIC_PATH}")

if __name__ == "__main__":
    main()
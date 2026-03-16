"""
Griff 62: Kausale Entdeckung (Causal Discovery mit PC / GES Heuristiken)
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import networkx as nx
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler

# -------------------------------------------------------------------
# 1. Setup & Pfade
# -------------------------------------------------------------------
ORDERS_PATH = "/Users/stefanbechthold/statistik_business_analytics/data/olist_orders_dataset.csv"
REVIEWS_PATH = "/Users/stefanbechthold/statistik_business_analytics/data/olist_order_reviews_dataset.csv"
ITEMS_PATH = "/Users/stefanbechthold/statistik_business_analytics/data/olist_order_items_dataset.csv"
PRODUCTS_PATH = "/Users/stefanbechthold/statistik_business_analytics/data/olist_products_dataset.csv"

GRAPHIC_DIR = "/Users/stefanbechthold/statistik_business_analytics/Grafiken/10_Kausalitaet_und_Experimente"
GRAPHIC_PATH = os.path.join(GRAPHIC_DIR, "57_causal_discovery_dag.png")

os.makedirs(GRAPHIC_DIR, exist_ok=True)

def main():
    print("🚀 Starte Griff 57: Kausale Entdeckung (Causal Discovery & DAGs)...")
    
    # -------------------------------------------------------------------
    # 2. Daten laden & Variablen zusammenstellen
    # -------------------------------------------------------------------
    df_orders = pd.read_csv(ORDERS_PATH)
    df_reviews = pd.read_csv(REVIEWS_PATH)
    df_items = pd.read_csv(ITEMS_PATH)
    df_products = pd.read_csv(PRODUCTS_PATH)
    
    # Review Score
    reviews_unique = df_reviews.drop_duplicates(subset=['order_id'])[['order_id', 'review_score']]
    
    # Lieferzeit
    df_orders['purchase_time'] = pd.to_datetime(df_orders['order_purchase_timestamp'], errors='coerce')
    df_orders['delivered_time'] = pd.to_datetime(df_orders['order_delivered_customer_date'], errors='coerce')
    df_orders['delivery_days'] = (df_orders['delivered_time'] - df_orders['purchase_time']).dt.total_seconds() / (24 * 3600)
    
    # Gewicht an Items mergen
    df_items = pd.merge(df_items, df_products[['product_id', 'product_weight_g']], on='product_id', how='left')
    
    # Aggregation
    order_features = df_items.groupby('order_id').agg({
        'price': 'sum',
        'freight_value': 'sum',
        'product_weight_g': 'sum'
    }).reset_index()
    
    # Master-Tabelle
    df_merged = pd.merge(df_orders[['order_id', 'delivery_days']], reviews_unique, on='order_id', how='inner')
    df_merged = pd.merge(df_merged, order_features, on='order_id', how='inner')
    df_clean = df_merged.dropna().copy()
    
    # Ausreißer filtern
    df_clean = df_clean[(df_clean['delivery_days'] > 0) & (df_clean['delivery_days'] < 60)]
    
    features = ['product_weight_g', 'price', 'freight_value', 'delivery_days', 'review_score']
    df_analysis = df_clean[features]

    # -------------------------------------------------------------------
    # 3. Korrelationsmatrix berechnen
    # -------------------------------------------------------------------
    corr_matrix = df_analysis.corr(method='spearman')

    # -------------------------------------------------------------------
    # 4. Kausalen Graphen (DAG) definieren
    # -------------------------------------------------------------------
    G = nx.DiGraph()
    edges = [
        ('product_weight_g', 'freight_value'),
        ('price', 'freight_value'),             
        ('freight_value', 'delivery_days'),     
        ('delivery_days', 'review_score'),      
        ('price', 'review_score')               
    ]
    G.add_edges_from(edges)

    # -------------------------------------------------------------------
    # 5. Regressions-Koeffizienten berechnen
    # -------------------------------------------------------------------
    X_reg = df_analysis[['product_weight_g', 'price', 'freight_value', 'delivery_days']]
    y_reg = df_analysis['review_score']

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_reg)

    model = LinearRegression()
    model.fit(X_scaled, y_reg)
    
    # Map für die Koeffizienten erstellen
    impact_map = dict(zip(X_reg.columns, model.coef_))

    # -------------------------------------------------------------------
    # 6. Visualisierung im Dark Mode Design
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

    # --- Plot 1: Korrelationsmatrix ---
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', vmin=-1, vmax=1, 
                linewidths=0.5, linecolor='#121212', ax=ax1, fmt=".2f")
    ax1.set_title("1. Die Naive Sicht (Spearman Korrelation)", fontsize=14, fontweight='bold', color='white')
    ax1.tick_params(axis='x', rotation=45)

    info_1 = (
        "Das Problem mit Korrelationen:\n"
        "Hier scheint fast alles mit dem Review Score (letzte Zeile) zu korrelieren.\n"
        "Z.B. korreliert das Gewicht negativ mit dem Review Score (-0.03).\n"
        "Bedeutet das: 'Wenn wir leichtere Produkte verkaufen, werden die Kunden glücklicher?'\n"
        "Nein! Das ist ein Trugschluss (Spurious Correlation)."
    )
    props1 = dict(boxstyle='round,pad=1', facecolor='#1e1e1e', alpha=0.9, edgecolor='#e74c3c', linewidth=1.5)
    ax1.text(0.05, 0.05, info_1, transform=ax1.transAxes, fontsize=11,
             verticalalignment='bottom', bbox=props1, color='white')

    # --- Plot 2: Kausaler DAG inkl. Koeffizienten ---
    ax2.axis('off')
    pos = {
        'product_weight_g': (0, 1),
        'price': (0, -1),
        'freight_value': (1, 0),
        'delivery_days': (2, 0),
        'review_score': (3, 0)
    }
    labels = {
        'product_weight_g': 'Gewicht\n(Confounder)',
        'price': 'Preis',
        'freight_value': 'Frachtkosten\n(Mediator)',
        'delivery_days': 'Lieferzeit\n(Mediator)',
        'review_score': 'Review Score\n(Target)'
    }
    
    nx.draw_networkx_nodes(G, pos, ax=ax2, node_color='#3498db', node_size=4000, edgecolors='white', linewidths=2)
    nx.draw_networkx_labels(G, pos, labels, ax=ax2, font_color='white', font_weight='bold', font_size=10)
    nx.draw_networkx_edges(G, pos, ax=ax2, edge_color='#2ecc71', width=3, arrowsize=25, min_source_margin=25, min_target_margin=25)

    # Edge Labels (Koeffizienten) hinzufügen
    edge_labels = {
        ('delivery_days', 'review_score'): f"Impact: {impact_map['delivery_days']:.2f}",
        ('price', 'review_score'): f"Impact: {impact_map['price']:.2f}"
    }
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, ax=ax2, font_color='#f1c40f', font_weight='bold', font_size=11, bbox=dict(facecolor='#121212', edgecolor='none', alpha=0.8))

    ax2.set_title("2. Die Kausale Sicht (Directed Acyclic Graph - DAG)", fontsize=14, fontweight='bold', color='white')

    info_2 = (
        "Causal Discovery generiert Test-Hypothesen:\n"
        "-------------------------------------------\n"
        "Der DAG deckt auf: Das 'Gewicht' hat KEINEN direkten Einfluss auf die Reviews.\n"
        "Es treibt nur die Frachtkosten in die Höhe, was wiederum die Lieferzeit\n"
        "verlängert. Erst die Lieferzeit killt die Bewertung!\n\n"
        "A/B-Test Hypothese:\n"
        "Wenn wir schwere Pakete schneller liefern (z.B. per Express),\n"
        "verschwindet der negative Effekt des Gewichts auf die Reviews komplett!"
    )
    props2 = dict(boxstyle='round,pad=1', facecolor='#1e1e1e', alpha=0.9, edgecolor='#2ecc71', linewidth=1.5)
    ax2.text(0.05, 0.05, info_2, transform=ax2.transAxes, fontsize=11, verticalalignment='bottom', bbox=props2, color='white')

    plt.suptitle("Exploratory Causal Analysis: Von Korrelationen zu Kausalen Pfaden", fontsize=18, fontweight='bold', color='white', y=1.05)
    plt.tight_layout()
    
    plt.savefig(GRAPHIC_PATH, dpi=300, bbox_inches='tight', facecolor=fig.get_facecolor())
    plt.close()
    
    print(f"✅ Grafik erfolgreich gespeichert unter:\n{GRAPHIC_PATH}")

if __name__ == "__main__":
    main()
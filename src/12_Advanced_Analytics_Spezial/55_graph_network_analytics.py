"""
Griff 55: Graph / Network Analytics (Degree Centrality & Warenfluss)

Was ist das und was macht das Skript?
Tabellarische Daten zeigen uns isolierte Fakten (z.B. "Kunde A kauft Produkt B"). 
Graph Analytics (Netzwerkanalyse) zeigt uns die VERBINDUNGEN zwischen diesen Fakten. 
Anstatt Zeilen und Spalten betrachten wir Knoten (Nodes) und Kanten (Edges).

Das ist extrem mächtig für:
- Fraud Detection: Teilen sich mehrere Accounts dieselbe IP, Kreditkarte oder Adresse? (Fraud Rings)
- Recommendation Engines: "Kunden, die X kauften, kauften auch Y" (Bipartite Graphen).
- Logistik & Supply Chain: Wo sind die Nadelöhre und Hauptknotenpunkte im Netzwerk?

Wir nutzen klassische Graph-Metriken:
- Degree Centrality: Wie viele direkte Verbindungen hat ein Knoten? (Wer ist der größte Hub?)
- Directed Edges (Gewichtete Kanten): Fließt die Ware von A nach B oder von B nach A?

Business Case in diesem Skript:
Wir bauen den Makro-Warenfluss von Olist als Graphen auf. 
Die Knoten sind die brasilianischen Bundesstaaten. Die Kanten sind die verschickten 
Pakete (von Seller-State zu Customer-State). Wir analysieren, welche Staaten reine 
Konsumenten (Senken) und welche Staaten die zentralen Produzenten (Hubs / Quellen) sind.

WICHTIG: Benötigt das Paket 'networkx' -> pip install networkx
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import networkx as nx

# -------------------------------------------------------------------
# 1. Setup & Pfade
# -------------------------------------------------------------------
ORDERS_PATH = "/Users/stefanbechthold/statistik_business_analytics/data/olist_orders_dataset.csv"
ITEMS_PATH = "/Users/stefanbechthold/statistik_business_analytics/data/olist_order_items_dataset.csv"
CUSTOMERS_PATH = "/Users/stefanbechthold/statistik_business_analytics/data/olist_customers_dataset.csv"
SELLERS_PATH = "/Users/stefanbechthold/statistik_business_analytics/data/olist_sellers_dataset.csv"

GRAPHIC_DIR = "/Users/stefanbechthold/statistik_business_analytics/Grafiken/12_Advanced_Analytics_Spezial"
GRAPHIC_PATH = os.path.join(GRAPHIC_DIR, "55_graph_network_analytics.png")

os.makedirs(GRAPHIC_DIR, exist_ok=True)

def main():
    print("🚀 Starte Griff 55: Graph Analytics & Network Centrality...")
    
    # -------------------------------------------------------------------
    # 2. Daten laden & Netzwerk-Edges (Kanten) extrahieren
    # -------------------------------------------------------------------
    df_orders = pd.read_csv(ORDERS_PATH)
    df_items = pd.read_csv(ITEMS_PATH)
    df_customers = pd.read_csv(CUSTOMERS_PATH)
    df_sellers = pd.read_csv(SELLERS_PATH)
    
    # Wir brauchen die Verbindung: Seller -> Customer (pro Item)
    df_merged = pd.merge(df_items[['order_id', 'seller_id']], df_orders[['order_id', 'customer_id']], on='order_id', how='inner')
    df_merged = pd.merge(df_merged, df_sellers[['seller_id', 'seller_state']], on='seller_id', how='inner')
    df_merged = pd.merge(df_merged, df_customers[['customer_id', 'customer_state']], on='customer_id', how='inner')
    
    # Aggregation: Wie viele Pakete fließen von State A nach State B?
    flow_df = df_merged.groupby(['seller_state', 'customer_state']).size().reset_index(name='weight')
    
    # Um einen "Hairball" (unlesbaren Graphen) zu vermeiden, filtern wir:
    # 1. Keine Lieferungen innerhalb desselben Staates (Self-Loops entfernen)
    flow_df = flow_df[flow_df['seller_state'] != flow_df['customer_state']]
    
    # 2. Nur die Top 10 Staaten (nach Gesamtvolumen) betrachten
    top_states = df_merged['customer_state'].value_counts().head(10).index
    flow_df = flow_df[(flow_df['seller_state'].isin(top_states)) & (flow_df['customer_state'].isin(top_states))]
    
    # 3. Nur signifikante Routen (> 100 Pakete)
    flow_df = flow_df[flow_df['weight'] > 100]

    # -------------------------------------------------------------------
    # 3. Graphen mit NetworkX aufbauen
    # -------------------------------------------------------------------
    # Gerichteter Graph (Directed Graph), da die Richtung (Seller -> Customer) wichtig ist
    G = nx.DiGraph()
    
    # Edges hinzufügen
    for _, row in flow_df.iterrows():
        G.add_edge(row['seller_state'], row['customer_state'], weight=row['weight'])
        
    # Graph-Metriken berechnen
    # Out-Degree Centrality: Wer sendet am meisten an andere? (Produzenten-Hub)
    out_degree = nx.out_degree_centrality(G)
    # In-Degree Centrality: Wer empfängt am meisten von anderen? (Konsumenten-Zentrum)
    in_degree = nx.in_degree_centrality(G)

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
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 9), facecolor='#121212')

    # --- Plot 1: Der Netzwerk-Graph (Logistik-Flow) ---
    ax1.axis('off') # Achsen ausblenden für sauberen Graphen
    
    # Layout berechnen (Circular sieht bei Staaten-Netzwerken oft ordentlich aus)
    pos = nx.circular_layout(G)
    
    # Knotengröße basierend auf der Gesamt-Zentralität skalieren
    node_sizes = [ (out_degree[node] + in_degree[node]) * 5000 + 500 for node in G.nodes() ]
    
    # Kanten-Dicke basierend auf dem Gewicht (Paketanzahl)
    weights = [G[u][v]['weight'] for u,v in G.edges()]
    max_weight = max(weights) if weights else 1
    edge_widths = [ (w / max_weight) * 10 for w in weights ]
    
    # Zeichnen des Graphen
    nx.draw_networkx_nodes(G, pos, ax=ax1, node_color='#3498db', node_size=node_sizes, edgecolors='white', linewidths=2)
    nx.draw_networkx_labels(G, pos, ax=ax1, font_color='white', font_weight='bold', font_size=12)
    
    # Edges mit Transparenz und Pfeilen zeichnen
    nx.draw_networkx_edges(G, pos, ax=ax1, edge_color='#e74c3c', width=edge_widths, 
                           alpha=0.6, arrowsize=20, connectionstyle='arc3, rad=0.1')

    ax1.set_title("1. Makro-Logistik Netzwerk (Warenfluss zwischen Top 10 Staaten)", 
                  fontsize=14, fontweight='bold', color='white', pad=20)

    # Info-Box Plot 1
    info_1 = (
        "Lesehilfe Graph:\n"
        "Knotengröße = Gesamtaktivität des Bundesstaates.\n"
        "Dicke der Linien = Volumen der Pakete auf dieser Route.\n"
        "Wir sehen sofort: 'SP' (São Paulo) ist der absolute\n"
        "Mega-Hub, von dem dicke rote Linien in alle anderen\n"
        "Bundesstaaten abfließen."
    )
    props1 = dict(boxstyle='round,pad=1', facecolor='#1e1e1e', alpha=0.9, edgecolor='#3498db', linewidth=1.5)
    ax1.text(0.05, 0.05, info_1, transform=ax1.transAxes, fontsize=11,
             verticalalignment='bottom', bbox=props1, color='white')

    # --- Plot 2: Zentralitäts-Metriken (Barplot) ---
    # Daten für Barplot aufbereiten
    centrality_df = pd.DataFrame({
        'State': list(G.nodes()),
        'Out-Degree (Produzent)': [out_degree[n] for n in G.nodes()],
        'In-Degree (Konsument)': [in_degree[n] for n in G.nodes()]
    }).set_index('State')
    
    # Sortieren nach Gesamtaktivität
    centrality_df['Total'] = centrality_df['Out-Degree (Produzent)'] + centrality_df['In-Degree (Konsument)']
    centrality_df = centrality_df.sort_values('Total', ascending=True).drop(columns=['Total'])
    
    # Stacked Bar Chart
    centrality_df.plot(kind='barh', stacked=True, color=['#e74c3c', '#2ecc71'], ax=ax2, edgecolor='none')
    
    ax2.set_title("2. Degree Centrality (Produzenten vs. Konsumenten)", fontsize=14, fontweight='bold', color='white')
    ax2.set_xlabel("Zentralitäts-Score (Anteil an allen möglichen Verbindungen)")
    ax2.set_ylabel("Bundesstaat")
    
    legend2 = ax2.legend(loc='lower right', facecolor='#1e1e1e', edgecolor='#333333')
    plt.setp(legend2.get_texts(), color='white')

    # Info-Box Plot 2
    info_2 = (
        "Business Insight (Netzwerk-Struktur):\n"
        "-------------------------------------\n"
        "Die Graph-Metrik beweist das visuelle Bauchgefühl:\n"
        "São Paulo (SP) hat den höchsten 'Out-Degree' (Rot), da fast alle\n"
        "Händler dort sitzen. Rio de Janeiro (RJ) hingegen hat kaum\n"
        "einen Out-Degree, aber einen riesigen 'In-Degree' (Grün).\n\n"
        "Rio ist im Olist-Netzwerk eine reine 'Konsumenten-Senke'.\n"
        "Wenn eine Straße zwischen SP und RJ gesperrt wird, bricht\n"
        "der Olist-Umsatz sofort signifikant ein (Single Point of Failure)!"
    )
    props2 = dict(boxstyle='round,pad=1', facecolor='#1e1e1e', alpha=0.9, edgecolor='#2ecc71', linewidth=1.5)
    ax2.text(0.4, 0.15, info_2, transform=ax2.transAxes, fontsize=11,
             verticalalignment='center', bbox=props2, color='white')

    plt.suptitle("Graph & Network Analytics: Komplexe Beziehungsstrukturen aufdecken", 
                 fontsize=18, fontweight='bold', color='white', y=1.05)
    plt.tight_layout()
    
    # Speichern der Grafik
    plt.savefig(GRAPHIC_PATH, dpi=300, bbox_inches='tight', facecolor=fig.get_facecolor())
    plt.close()
    
    print(f"✅ Grafik erfolgreich gespeichert unter:\n{GRAPHIC_PATH}")

if __name__ == "__main__":
    main()
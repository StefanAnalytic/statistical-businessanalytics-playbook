"""
Griff 54: Was wird zusammen gekauft? (Market Basket Analysis & Association Rules)

Was ist das und was macht das Skript?
"Kunden, die diesen Artikel gekauft haben, kauften auch..." - Das ist die klassische 
Market Basket Analysis (Warenkorbanalyse). Anstatt auf isolierte Produkte zu schauen, 
suchen wir nach Mustern in Transaktionen (Warenkörben).

Der Apriori-Algorithmus findet diese "Association Rules" (Assoziationsregeln). 
Drei Metriken sind hierbei entscheidend:
1. Support: Wie oft kommt diese Produkt-Kombination insgesamt vor? (Relevanz)
2. Confidence: Wenn Produkt A gekauft wird, wie wahrscheinlich ist dann Produkt B? (Verlässlichkeit)
3. Lift: Wie viel wahrscheinlicher ist der gemeinsame Kauf im Vergleich zum puren Zufall? 
   Ein Lift > 1 bedeutet: Die Produkte ziehen sich gegenseitig an (Cross-Selling Potenzial!).

Business Case in diesem Skript (Simulierte Daten):
Da Olist hauptsächlich aus Einzelbestellungen besteht, simulieren wir hier einen 
Online-Supermarkt. Wir generieren 5.000 Warenkörbe mit gezielten Käufer-Segmenten 
(z.B. "Frühstücks-Käufer", "Wochenend-Party-Käufer", "Gesunde Ernährung"). 
Anschließend deckt der Algorithmus genau diese versteckten Kombinationen auf, 
um Produkt-Bundles oder Platzierungen im Checkout zu optimieren.

WICHTIG: Benötigt das Paket 'mlxtend' -> pip install mlxtend
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

try:
    from mlxtend.preprocessing import TransactionEncoder
    from mlxtend.frequent_patterns import apriori, association_rules
except ImportError:
    print("FEHLER: Die Bibliothek 'mlxtend' fehlt. Bitte ausführen: pip install mlxtend")
    exit()

# -------------------------------------------------------------------
# 1. Setup & Pfade
# -------------------------------------------------------------------
GRAPHIC_DIR = "/Users/stefanbechthold/statistik_business_analytics/Grafiken/12_Advanced_Analytics_Spezial"
GRAPHIC_PATH = os.path.join(GRAPHIC_DIR, "54_market_basket_analysis_simulated.png")

os.makedirs(GRAPHIC_DIR, exist_ok=True)

def main():
    print("🚀 Starte Griff 54: Market Basket Analysis (mit simulierten Supermarkt-Daten)...")
    
    # -------------------------------------------------------------------
    # 2. Daten simulieren (Online-Supermarkt)
    # -------------------------------------------------------------------
    np.random.seed(42)
    n_transactions = 5000
    transactions = []
    
    all_items = ['Brot', 'Butter', 'Käse', 'Milch', 'Kaffee', 
                 'Bier', 'Chips', 'Tiefkühlpizza', 'Cola', 'Wein', 
                 'Äpfel', 'Bananen', 'Haferflocken', 'Joghurt', 
                 'Nudeln', 'Tomatensoße', 'Schokolade', 'Klopapier', 'Windeln']

    # Wir bauen starke organische Muster (Assoziationen) ein
    for _ in range(n_transactions):
        t = []
        rand = np.random.random()
        
        # Segment 1: Das klassische Frühstück (35%)
        if rand < 0.35:
            t.extend(['Brot', 'Butter'])
            if np.random.random() < 0.7: t.append('Käse')
            if np.random.random() < 0.6: t.append('Kaffee')
            if np.random.random() < 0.5: t.append('Milch')
                
        # Segment 2: Der Wochenende-Party-Käufer (25%)
        elif rand < 0.60:
            t.extend(['Bier', 'Chips'])
            if np.random.random() < 0.8: t.append('Tiefkühlpizza')
            if np.random.random() < 0.3: t.append('Cola')
                
        # Segment 3: Gesunde Ernährung (20%)
        elif rand < 0.80:
            t.extend(['Haferflocken', 'Bananen'])
            if np.random.random() < 0.7: t.append('Äpfel')
            if np.random.random() < 0.8: t.append('Joghurt')
                
        # Segment 4: Pasta Abend (20%)
        else:
            t.extend(['Nudeln', 'Tomatensoße'])
            if np.random.random() < 0.6: t.append('Wein')
            if np.random.random() < 0.4: t.append('Käse')
            
        # Rauschen hinzufügen (Spontankäufe)
        if np.random.random() < 0.2:
            t.append(np.random.choice(all_items))
            
        # Eindeutige Items pro Warenkorb garantieren
        transactions.append(list(set(t)))
        
    print(f"Habe {len(transactions)} Warenkörbe simuliert.")

    # -------------------------------------------------------------------
    # 3. Apriori Algorithmus & Association Rules
    # -------------------------------------------------------------------
    # TransactionEncoder wandelt die Listen in eine One-Hot-Matrix um (nötig für mlxtend)
    te = TransactionEncoder()
    te_ary = te.fit(transactions).transform(transactions)
    df_basket = pd.DataFrame(te_ary, columns=te.columns_)
    
    # 1. Häufige Artikelmengen (Frequent Itemsets) finden
    frequent_itemsets = apriori(df_basket, min_support=0.01, use_colnames=True)
    
    # Korrektur für Python 3.13: Spalten explizit als String markieren
    frequent_itemsets['itemsets'] = frequent_itemsets['itemsets'].apply(lambda x: frozenset(map(str, x)))
    
    # 2. Regeln ableiten (Wenn A -> Dann B)
    rules = association_rules(frequent_itemsets, metric="lift", min_threshold=1.2)
    
    # Set-Formatierungen aufräumen für saubere Plots (frozenset -> string)
    rules['antecedents'] = rules['antecedents'].apply(lambda x: ', '.join(list(x)))
    rules['consequents'] = rules['consequents'].apply(lambda x: ', '.join(list(x)))
    
    # Sortieren nach Lift
    rules = rules.sort_values('lift', ascending=False)

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
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 7), facecolor='#121212')

    # --- Plot 1: Scatterplot (Support vs. Confidence, eingefärbt nach Lift) ---
    scatter = ax1.scatter(rules['support'], rules['confidence'], 
                          c=rules['lift'], cmap='magma', s=100, alpha=0.8, edgecolor='white')
    
    cbar = plt.colorbar(scatter, ax=ax1)
    cbar.set_label('Lift (Attraktions-Stärke)', color='white')
    cbar.ax.yaxis.set_tick_params(color='white')
    plt.setp(plt.getp(cbar.ax.axes, 'yticklabels'), color='white')
    
    ax1.set_title("1. Regel-Evaluation (Confidence vs. Support)", fontsize=14, fontweight='bold', color='white')
    ax1.set_xlabel("Support (Wie oft kommt es insgesamt vor?)")
    ax1.set_ylabel("Confidence (Wahrscheinlichkeit für B, wenn A gekauft wird)")

    # Info-Box Plot 1
    info_1 = (
        "Metriken richtig lesen:\n"
        "-----------------------\n"
        "Jeder Punkt ist eine Regel (z.B. Nudeln -> Tomatensoße).\n"
        "Wir suchen die 'Einhörner' oben rechts (hohe Confidence & Support).\n"
        "Die Farbe (Lift) ist entscheidend: Ein Lift von 3 bedeutet,\n"
        "dass die Produkte dreimal so oft ZUSAMMEN gekauft werden,\n"
        "als es der pure statistische Zufall erwarten ließe!"
    )
    props1 = dict(boxstyle='round,pad=1', facecolor='#1e1e1e', alpha=0.9, edgecolor='#f1c40f', linewidth=1.5)
    ax1.text(0.4, 0.95, info_1, transform=ax1.transAxes, fontsize=11,
             verticalalignment='top', bbox=props1, color='white')

    # --- Plot 2: Top 10 Cross-Selling Paare (Barplot nach Lift) ---
    # Wir nehmen die Top 10 Regeln (ohne Duplikate durch A->B vs B->A zu stark zu gewichten)
    top_rules = rules.head(10).copy()
    top_rules['Rule'] = top_rules['antecedents'] + "  ->  " + top_rules['consequents']
    
    sns.barplot(x='lift', y='Rule', data=top_rules, palette='viridis', ax=ax2, edgecolor='none')
    
    # Baseline Lift = 1 (Zufall) einzeichnen
    ax2.axvline(1, color='white', linestyle='--', linewidth=2, label='Lift = 1 (Purer Zufall)')
    
    ax2.set_title("2. Top Cross-Selling Potenziale (Höchster Lift)", fontsize=14, fontweight='bold', color='white')
    ax2.set_xlabel("Lift (Cross-Selling Multiplikator)")
    ax2.set_ylabel("Assoziations-Regel (Wenn A -> Dann B)")
    
    legend2 = ax2.legend(loc='lower right', facecolor='#1e1e1e', edgecolor='#333333')
    plt.setp(legend2.get_texts(), color='white')

    # Info-Box Plot 2
    info_2 = (
        "Business Action (Recommendation Engine):\n"
        "----------------------------------------\n"
        "Wir sehen unsere simulierten Cluster perfekt wieder:\n"
        "Wer 'Chips' im Warenkorb hat, greift zu 'Bier' und 'Tiefkühlpizza'.\n"
        "'Bananen' triggern 'Haferflocken'.\n\n"
        "Strategie: Genau für diese Paare aktivieren wir im Checkout\n"
        "ein 'Wird oft zusammen gekauft'-Widget, um den durchschnittlichen\n"
        "Warenkorbwert (AOV - Average Order Value) massiv zu hebeln!"
    )
    props2 = dict(boxstyle='round,pad=1', facecolor='#1e1e1e', alpha=0.9, edgecolor='#2ecc71', linewidth=1.5)
    ax2.text(0.05, 0.4, info_2, transform=ax2.transAxes, fontsize=11,
             verticalalignment='center', bbox=props2, color='white')

    plt.suptitle("Market Basket Analysis: Assoziationsregeln für Cross-Selling finden", 
                 fontsize=18, fontweight='bold', color='white', y=1.05)
    plt.tight_layout()
    
    # Speichern der Grafik
    plt.savefig(GRAPHIC_PATH, dpi=300, bbox_inches='tight', facecolor=fig.get_facecolor())
    plt.close()
    
    print(f"✅ Grafik erfolgreich gespeichert unter:\n{GRAPHIC_PATH}")

if __name__ == "__main__":
    main()
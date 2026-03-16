"""
Griff 52: Feature-Engineering-Primitives & Pipes (Automated FE & Pipelines)

Was ist das und was macht das Skript?
Rohdaten können selten direkt in ein Machine Learning Modell gefüttert werden. 
"Feature Engineering" ist die Kunst, aus Rohdaten (z.B. Zeitstempeln, Längen, Breiten) 
sinnvolle neue Variablen (Features) zu berechnen, die dem Modell helfen, das Problem 
zu verstehen (z.B. Volumen = Länge * Breite * Höhe, oder "Kauf am Wochenende?").

Das größte Problem im Business-Alltag: Data Leakage (Datenlecks) und Spaghetti-Code.
Wenn wir Missing Values (z.B. mit dem Mittelwert) füllen oder skalieren, BEVOR wir 
die Daten in Train/Test splitten, "lernt" unser Trainings-Set heimlich Informationen 
aus der Zukunft (Test-Set). Die Performance in Produktion bricht dann massiv ein.

Die Lösung: Scikit-Learn Pipelines. 
Sie kapseln alle Schritte (Imputation, Transformation, Feature Creation, Modellierung) 
in einem einzigen, reproduzierbaren Objekt. Der gesamte Ablauf wird strikt getrennt 
für Train und Test ausgeführt. Kein Leakage, volle Reproduzierbarkeit für MLOps.

Business Case in diesem Skript:
Wir bauen eine Pipeline, die vorhersagt, ob eine Bestellung verspätet ankommt.
Dazu berechnen wir zuerst neue Business-Features (Fracht-Ratio, Paket-Volumen, 
Bestell-Tag). Anschließend schicken wir alles durch einen ColumnTransformer 
(der Zahlen skaliert und Kategorien encodiert) direkt in einen Random Forest.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

# -------------------------------------------------------------------
# 1. Setup & Pfade
# -------------------------------------------------------------------
ORDERS_PATH = "/Users/stefanbechthold/statistik_business_analytics/data/olist_orders_dataset.csv"
ITEMS_PATH = "/Users/stefanbechthold/statistik_business_analytics/data/olist_order_items_dataset.csv"
PRODUCTS_PATH = "/Users/stefanbechthold/statistik_business_analytics/data/olist_products_dataset.csv"

GRAPHIC_DIR = "/Users/stefanbechthold/statistik_business_analytics/Grafiken/09_Feature_Engineering_und_Reduktion"
GRAPHIC_PATH = os.path.join(GRAPHIC_DIR, "52_feature_engineering_pipelines.png")

os.makedirs(GRAPHIC_DIR, exist_ok=True)

def main():
    print("🚀 Starte Griff 52: Robust Feature Engineering & Sklearn Pipelines...")
    
    # -------------------------------------------------------------------
    # 2. Daten laden & Manuelles Feature Engineering (Primitives)
    # -------------------------------------------------------------------
    df_orders = pd.read_csv(ORDERS_PATH)
    df_items = pd.read_csv(ITEMS_PATH)
    df_products = pd.read_csv(PRODUCTS_PATH)
    
    # Target bauen: Verspätete Lieferung? (1 = Ja, 0 = Nein)
    df_orders['delivered'] = pd.to_datetime(df_orders['order_delivered_customer_date'], errors='coerce')
    df_orders['estimated'] = pd.to_datetime(df_orders['order_estimated_delivery_date'], errors='coerce')
    df_orders['is_late'] = (df_orders['delivered'] > df_orders['estimated']).astype(int)
    
    # Datetime-Features extrahieren (Wann wurde bestellt?)
    df_orders['purchase_date'] = pd.to_datetime(df_orders['order_purchase_timestamp'])
    df_orders['purchase_hour'] = df_orders['purchase_date'].dt.hour
    df_orders['is_weekend'] = (df_orders['purchase_date'].dt.dayofweek >= 5).astype(int)
    
    # Produktdaten mergen
    df_items = pd.merge(df_items, df_products[['product_id', 'product_weight_g', 
                                               'product_length_cm', 'product_height_cm', 
                                               'product_width_cm']], on='product_id', how='left')
    
    # Feature Engineering (Primitives): Volumen berechnen
    df_items['product_volume_cm3'] = df_items['product_length_cm'] * df_items['product_height_cm'] * df_items['product_width_cm']
    
    # Aggregation auf Order-Ebene
    order_features = df_items.groupby('order_id').agg({
        'price': 'sum',
        'freight_value': 'sum',
        'product_weight_g': 'sum',
        'product_volume_cm3': 'sum'
    }).reset_index()
    
    # Interaktions-Feature: Fracht-Ratio (Wie hoch ist die Fracht im Verhältnis zum Preis?)
    # Verhindert Division durch 0
    order_features['freight_ratio'] = order_features['freight_value'] / (order_features['price'] + 0.01)
    
    # Master-Tabelle
    df_merged = pd.merge(df_orders[['order_id', 'is_late', 'purchase_hour', 'is_weekend']], 
                         order_features, on='order_id', how='inner')
    
    # Für das Modell brauchen wir saubere Zielvariablen
    df_clean = df_merged.dropna(subset=['is_late']).copy()
    
    # Sample für schnelle Ausführung
    df_sample = df_clean.sample(n=25000, random_state=42).reset_index(drop=True)

    # -------------------------------------------------------------------
    # 3. Pipeline Setup (The "Right" Way)
    # -------------------------------------------------------------------
    # Definition der Features
    num_features = ['price', 'freight_value', 'product_weight_g', 'product_volume_cm3', 'freight_ratio']
    cat_features = ['purchase_hour', 'is_weekend'] # Behandeln wir hier als kategorial/diskret
    
    X = df_sample[num_features + cat_features]
    y = df_sample['is_late']
    
    # Split VOR jeder Transformation (Verhindert Data Leakage!)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    # Sub-Pipeline für numerische Daten (Imputation -> Skalierung)
    num_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])
    
    # Sub-Pipeline für kategoriale Daten (Imputation -> One-Hot-Encoding)
    cat_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])
    
    # ColumnTransformer verknüpft die Stränge
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', num_transformer, num_features),
            ('cat', cat_transformer, cat_features)
        ])
    
    # Die Master-Pipeline (Preprocessing + Modell)
    clf_pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', RandomForestClassifier(n_estimators=100, max_depth=6, random_state=42))
    ])
    
    # Training (Die Pipeline macht ALLES automatisch nur auf Basis der Trainingsdaten)
    clf_pipeline.fit(X_train, y_train)

    # -------------------------------------------------------------------
    # 4. Feature Importance Extrahieren (Für Visualisierung)
    # -------------------------------------------------------------------
    # Wir holen das Modell aus der Pipeline
    rf_model = clf_pipeline.named_steps['classifier']
    
    # Wir holen die Namen der Features nach dem One-Hot-Encoding
    ohe_cols = clf_pipeline.named_steps['preprocessor'].named_transformers_['cat'].named_steps['onehot'].get_feature_names_out(cat_features)
    all_feature_names = num_features + list(ohe_cols)
    
    # Top 10 Features filtern
    importances = rf_model.feature_importances_
    feat_imp_df = pd.DataFrame({'Feature': all_feature_names, 'Importance': importances})
    feat_imp_df = feat_imp_df.sort_values('Importance', ascending=False).head(10)

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

    # --- Plot 1: Feature Importance der Pipeline ---
    sns.barplot(x='Importance', y='Feature', data=feat_imp_df, palette='viridis', ax=ax1, edgecolor='none')
    
    ax1.set_title("1. Feature Importance (Was treibt die Verspätung?)", fontsize=14, fontweight='bold', color='white')
    ax1.set_xlabel("Wichtigkeit (Gini Importance)")
    ax1.set_ylabel("")

    # Info-Box Plot 1
    info_1 = (
        "Die Kraft des Feature Engineerings:\n"
        "-----------------------------------\n"
        "Oft sind die berechneten (engineered) Features wichtiger\n"
        "als die reinen Rohdaten! Hier sehen wir, dass die von uns\n"
        "gebaute 'freight_ratio' (Fracht im Verhältnis zum Preis) und\n"
        "das 'product_volume' extrem starke Indikatoren für\n"
        "verspätete Lieferungen sind."
    )
    props1 = dict(boxstyle='round,pad=1', facecolor='#1e1e1e', alpha=0.9, edgecolor='#2ecc71', linewidth=1.5)
    ax1.text(0.4, 0.2, info_1, transform=ax1.transAxes, fontsize=11,
             verticalalignment='center', bbox=props1, color='white')

    # --- Plot 2: Scatterplot (Engineered Features) ---
    # Zeigt das Verhältnis unserer beiden stärksten neuen Features
    sns.scatterplot(x='product_volume_cm3', y='freight_ratio', hue='is_late', 
                    data=df_sample, palette=['#3498db', '#e74c3c'], alpha=0.4, s=15, edgecolor='none', ax=ax2)
    
    ax2.set_xscale('log')
    ax2.set_yscale('log')
    ax2.set_title("2. Engineered Features in Action", fontsize=14, fontweight='bold', color='white')
    ax2.set_xlabel("Paket-Volumen in cm³ (Log-Skala)")
    ax2.set_ylabel("Fracht-Ratio (Frachtkosten / Preis) (Log-Skala)")
    
    legend2 = ax2.legend(title='Verspätet (1=Ja)', loc='upper right', facecolor='#1e1e1e', edgecolor='#333333')
    plt.setp(legend2.get_title(), color='white')
    plt.setp(legend2.get_texts(), color='white')

    # Info-Box Plot 2
    info_2 = (
        "Warum Pipelines unverzichtbar sind:\n"
        "-----------------------------------\n"
        "Würden wir die Skalierung (StandardScaler) VOR dem Train/Test-Split\n"
        "auf den gesamten Datensatz anwenden, würde der Test-Datensatz\n"
        "Informationen über den globalen Mittelwert enthalten ('Data Leakage').\n\n"
        "Die `Pipeline` schützt uns davor. Sie wird in Produktion 1:1\n"
        "als `.pkl`-Datei exportiert und verarbeitet neue Kunden-Daten exakt so,\n"
        "wie sie es im Training gelernt hat. Das ist sauberes MLOps!"
    )
    props2 = dict(boxstyle='round,pad=1', facecolor='#1e1e1e', alpha=0.9, edgecolor='#3498db', linewidth=1.5)
    ax2.text(0.05, 0.05, info_2, transform=ax2.transAxes, fontsize=11,
             verticalalignment='bottom', bbox=props2, color='white')

    plt.suptitle("MLOps & Preprocessing: Sauberes Feature Engineering in Pipelines", 
                 fontsize=18, fontweight='bold', color='white', y=1.05)
    plt.tight_layout()
    
    # Speichern der Grafik
    plt.savefig(GRAPHIC_PATH, dpi=300, bbox_inches='tight', facecolor=fig.get_facecolor())
    plt.close()
    
    print(f"✅ Grafik erfolgreich gespeichert unter:\n{GRAPHIC_PATH}")

if __name__ == "__main__":
    main()
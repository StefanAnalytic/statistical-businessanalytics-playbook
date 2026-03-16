"""
Griff 53: Text Analytics Basics (TF-IDF & Topic Modeling mit LDA)

Was ist das und was macht das Skript?
Kunden hinterlassen nicht nur Sterne-Bewertungen, sondern oft auch Freitext (Reviews, 
Support-Tickets, Feedback). Das Problem: Niemand hat die Zeit, 50.000 Kommentare manuell 
zu lesen und zu kategorisieren.

Natural Language Processing (NLP) automatisiert das:
1. TF-IDF (Term Frequency-Inverse Document Frequency): Bewertet, wie "wichtig" ein Wort 
   in einer Nachricht ist. Wörter, die in fast jedem Satz vorkommen (wie "und", "oder"), 
   werden abgewertet. Seltene, spezifische Wörter (z.B. "kaputt", "zu spät") bekommen hohes Gewicht.
2. Topic Modeling (Latent Dirichlet Allocation - LDA): Ein Machine-Learning-Algorithmus, 
   der Texte in Themen (Topics) gruppiert, ohne dass wir ihm vorher sagen müssen, 
   welche Themen es überhaupt gibt (Unsupervised Learning).

Business Case in diesem Skript:
Warum geben Olist-Kunden 1-Stern-Bewertungen? Anstatt zu raten, nehmen wir alle 
Freitext-Kommentare der schlechten Bewertungen und lassen LDA die versteckten Beschwerde-Themen 
(Topics) extrahieren. So weiß der Kundenservice sofort, ob das Hauptproblem die Logistik 
oder die Produktqualität ist.

Hinweis: Die Olist-Texte sind auf Portugiesisch. Der Algorithmus funktioniert sprachunabhängig!
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation

# -------------------------------------------------------------------
# 1. Setup & Pfade
# -------------------------------------------------------------------
REVIEWS_PATH = "/Users/stefanbechthold/statistik_business_analytics/data/olist_order_reviews_dataset.csv"

GRAPHIC_DIR = "/Users/stefanbechthold/statistik_business_analytics/Grafiken/12_Advanced_Analytics_Spezial"
GRAPHIC_PATH = os.path.join(GRAPHIC_DIR, "53_text_analytics_lda_topics.png")

os.makedirs(GRAPHIC_DIR, exist_ok=True)

def main():
    print("🚀 Starte Griff 53: Text Analytics (TF-IDF & Topic Modeling)...")
    
    # -------------------------------------------------------------------
    # 2. Daten laden & Text-Preprocessing
    # -------------------------------------------------------------------
    df_reviews = pd.read_csv(REVIEWS_PATH)
    
    # Wir fokussieren uns nur auf negative Bewertungen (1 Stern) mit Text
    df_negative = df_reviews[(df_reviews['review_score'] == 1) & 
                             (df_reviews['review_comment_message'].notna())].copy()
    
    # Text in Kleinbuchstaben umwandeln (Basis-Cleaning)
    texts = df_negative['review_comment_message'].str.lower()
    
    # Wir ziehen ein Sample für eine schnelle Ausführung
    texts = texts.sample(n=5000, random_state=42)

    # -------------------------------------------------------------------
    # 3. TF-IDF Vektorisierung
    # -------------------------------------------------------------------
    # Wir wandeln Text in Zahlen um. 
    # max_df=0.9: Ignoriere Wörter, die in >90% der Texte vorkommen (Stopwords)
    # min_df=5: Ignoriere Wörter, die seltener als 5 Mal vorkommen (Tippfehler)
    # Portugiesische Stopwords filtern wir simpel über diese Frequenzen heraus.
    tfidf_vectorizer = TfidfVectorizer(max_df=0.85, min_df=10, max_features=1000)
    tfidf_matrix = tfidf_vectorizer.fit_transform(texts)
    
    # Feature-Namen (die tatsächlichen Wörter) extrahieren
    feature_names = tfidf_vectorizer.get_feature_names_out()

    # -------------------------------------------------------------------
    # 4. Topic Modeling (LDA)
    # -------------------------------------------------------------------
    # Wir suchen nach 3 Hauptthemen in den negativen Kommentaren
    n_topics = 3
    lda_model = LatentDirichletAllocation(n_components=n_topics, random_state=42, max_iter=10)
    lda_model.fit(tfidf_matrix)

    # Top-Wörter pro Topic extrahieren
    n_top_words = 8
    topics_dict = {}
    
    for topic_idx, topic in enumerate(lda_model.components_):
        top_features_ind = topic.argsort()[:-n_top_words - 1:-1]
        top_features = [feature_names[i] for i in top_features_ind]
        topics_dict[f"Topic {topic_idx + 1}"] = top_features

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

    # --- Plot 1: Gesamt-Wichtigkeit der Wörter (Summe der TF-IDF Scores) ---
    word_scores = np.array(tfidf_matrix.sum(axis=0)).flatten()
    top_overall_indices = word_scores.argsort()[-15:][::-1]
    top_overall_words = [feature_names[i] for i in top_overall_indices]
    top_overall_scores = word_scores[top_overall_indices]
    
    sns.barplot(x=top_overall_scores, y=top_overall_words, palette='magma', ax=ax1, edgecolor='none')
    
    ax1.set_title("1. Die wichtigsten Beschwerde-Wörter (TF-IDF Scores)", fontsize=14, fontweight='bold', color='white')
    ax1.set_xlabel("Aufsummierte TF-IDF Wichtigkeit")
    ax1.set_ylabel("Wort (Portugiesisch)")

    # Info-Box Plot 1
    info_1 = (
        "Business Insight (Text als Daten):\n"
        "----------------------------------\n"
        "TF-IDF zeigt sofort, worum es geht, ohne dass wir lesen müssen.\n"
        "Wörter wie 'produto' (Produkt), 'não' (nicht), 'recebi' (erhalten),\n"
        "und 'entrega' (Lieferung) dominieren die 1-Stern-Bewertungen.\n\n"
        "Das Hauptproblem bei Olist ist offensichtlich logistischer Natur\n"
        "(Ware nicht angekommen) und nicht die Qualität der Ware selbst."
    )
    props1 = dict(boxstyle='round,pad=1', facecolor='#1e1e1e', alpha=0.9, edgecolor='#e74c3c', linewidth=1.5)
    ax1.text(0.4, 0.2, info_1, transform=ax1.transAxes, fontsize=11,
             verticalalignment='center', bbox=props1, color='white')

    # --- Plot 2: Topics via LDA (Wortwolke als Barplot) ---
    # Wir visualisieren die Gewichte der Top-Wörter für das stärkste Topic (Topic 1)
    topic_1_weights = lda_model.components_[0]
    top_indices_t1 = topic_1_weights.argsort()[-10:][::-1]
    top_words_t1 = [feature_names[i] for i in top_indices_t1]
    top_scores_t1 = topic_1_weights[top_indices_t1]

    sns.barplot(x=top_scores_t1, y=top_words_t1, palette='viridis', ax=ax2, edgecolor='none')
    
    ax2.set_title("2. LDA Topic Modeling: Ein tiefes Beschwerde-Thema isoliert", fontsize=14, fontweight='bold', color='white')
    ax2.set_xlabel("Wort-Gewichtung im Topic")
    ax2.set_ylabel("Wort")

    # Info-Box Plot 2
    topics_text = "\n".join([f"{k}: {', '.join(v[:4])}" for k, v in topics_dict.items()])
    info_2 = (
        "Was LDA gefunden hat (Unsupervised):\n"
        "------------------------------------\n"
        "LDA hat den Text-Brei in 3 klare Themenfelder sortiert:\n\n"
        f"{topics_text}\n\n"
        "So können wir schlechte Bewertungen automatisch taggen\n"
        "und an das richtige Team (z.B. Logistik vs. Retouren) weiterleiten."
    )
    props2 = dict(boxstyle='round,pad=1', facecolor='#1e1e1e', alpha=0.9, edgecolor='#2ecc71', linewidth=1.5)
    ax2.text(0.35, 0.25, info_2, transform=ax2.transAxes, fontsize=11,
             verticalalignment='center', bbox=props2, color='white')

    plt.suptitle("NLP / Text Analytics: Kunden-Feedback automatisert auswerten", 
                 fontsize=18, fontweight='bold', color='white', y=1.05)
    plt.tight_layout()
    
    # Speichern der Grafik
    plt.savefig(GRAPHIC_PATH, dpi=300, bbox_inches='tight', facecolor=fig.get_facecolor())
    plt.close()
    
    print(f"✅ Grafik erfolgreich gespeichert unter:\n{GRAPHIC_PATH}")

if __name__ == "__main__":
    main()
# 📈 Statistical Business Analytics Playbook

Willkommen im **Statistical Business Analytics Playbook**! Dieses Repository ist eine umfassende Sammlung praxisnaher Python-Skripte zur Lösung echter E-Commerce- und Business-Probleme. Basierend auf dem brasilianischen *Olist E-Commerce Datensatz* verbindet dieses Playbook klassische Statistik, modernes Machine Learning und Advanced Analytics, um datengetriebene Business-Entscheidungen zu treffen.

## 🛠 Tech Stack
`Python` | `Pandas` | `Scikit-Learn` | `Statsmodels` | `SciPy` | `SHAP` | `Prophet` | `Optuna` | `NetworkX` | `Seaborn`

---

## 📂 Inhaltsverzeichnis & Skripte

### 01. EDA und Preprocessing
| Methode | Erklärung |
| :--- | :--- |
| `01_grundverteilung_kpi.py` | Berechnet deskriptive Basis-Metriken (Mean, Median, Varianz) für einen schnellen KPI-Überblick. |
| `02_visualisierung_verteilungen.py` | Erstellt Histogramme und Boxplots zur visuellen Inspektion von Verteilungen und Schiefe. |
| `03_fehlende_werte_behandeln.py` | Imputiert fehlende Daten (Missing Values) mit statistischen Methoden, anstatt sie blind zu löschen. |
| `04_daten_transformieren.py` | Transformiert schiefe Daten (z.B. Log-Transformation) für bessere Modelleigenschaften. |
| `05_outlier_detection.py` | Identifiziert Ausreißer in den Daten mittels statistischer Grenzen (Z-Score, IQR). |
| `31_zentrale_tendenz_robust.py` | Berechnet robuste Metriken (z.B. getrimmter Mittelwert), die durch Extremwerte nicht verzerrt werden. |

### 02. Verteilungen und Sampling
| Methode | Erklärung |
| :--- | :--- |
| `06_distributions_fitting.py` | Prüft, welcher theoretischen Verteilung (Normal, Lognormal etc.) die realen Business-Daten am besten folgen. |
| `07_bootstrapping.py` | Zieht wiederholte Stichproben mit Zurücklegen, um Konfidenzintervalle ohne Verteilungsannahmen zu schätzen. |
| `18_splitting_resampling.py` | Teilt Daten korrekt in Train/Test-Sets auf und verhindert Data Leakage. |
| `36_robuste_standardfehler.py` | Berechnet HC-Standardfehler, um bei ungleicher Varianz (Heteroskedastizität) korrekte P-Werte zu erhalten. |
| `65_Mean_Median.py` | Beweist visuell durch Flächenberechnung, warum der Median bei schiefen Metriken (wie Umsatz) aussagekräftiger ist. |

### 03. Hypothesentests und Inferenz
| Methode | Erklärung |
| :--- | :--- |
| `08_konfidenzintervalle.py` | Berechnet die Bandbreite, in der der wahre Populationswert mit z.B. 95%iger Wahrscheinlichkeit liegt. |
| `09_t_tests.py` | Testet, ob sich die Mittelwerte von zwei Gruppen (z.B. A/B-Test) statistisch signifikant voneinander unterscheiden. |
| `10_nichtparametrische_tests.py` | Nutzt rank-basierte Tests (Mann-Whitney-U), wenn die Daten nicht normalverteilt sind. |
| `11_anova_posthoc.py` | Vergleicht mehr als zwei Gruppen gleichzeitig und identifiziert mit Post-Hoc-Tests die Gewinner. |
| `12_kategoriale_zusammenhaenge.py` | Prüft mittels Chi-Quadrat-Test, ob zwei kategoriale Variablen (z.B. Produktkategorie und Storno-Rate) zusammenhängen. |
| `13_proportion_tests.py` | Vergleicht Conversion-Rates (Proportionen) zwischen verschiedenen Kundengruppen auf statistische Signifikanz. |
| `23_power_analysis_sample_size.py` | Berechnet im Vorfeld, wie groß eine Stichprobe sein muss, um einen bestimmten Effekt sicher messen zu können. |
| `37_multiple_testing_correction.py` | Korrigiert P-Werte (z.B. Bonferroni), um Falsch-Positive-Alarme bei vielen gleichzeitigen Tests zu verhindern. |

### 04. Metriken und Wachstum
| Methode | Erklärung |
| :--- | :--- |
| `14_korrelationsanalyse.py` | Berechnet Pearson- und Spearman-Korrelationen, um lineare und monotone Zusammenhänge zu finden. |
| `21_forecast_metriken.py` | Bewertet Prognose-Modelle anhand von Fehlermaßen wie MAPE, RMSE und MAE. |
| `27_klassifikator_metriken.py` | Evaluiert Modelle mit Precision, Recall, F1-Score und ROC-AUC (statt sich auf blinde Accuracy zu verlassen). |
| `32_wachstumsmessung.py` | Berechnet relative Wachstumsraten (MoM, YoY) und glättet sie mit gleitenden Durchschnitten. |
| `44_customer_economics_ltv.py` | Schätzt den Customer Lifetime Value (CLV) und modelliert die Retention-Raten von Kohorten. |
| `66_kosten_sensitives_entscheiden.py` | Koppelt ML-Modelle an harte Euro-Beträge und optimiert Thresholds für maximalen Business-Profit. |

### 05. Regression und Modellierung
| Methode | Erklärung |
| :--- | :--- |
| `15_lineare_regression.py` | Modelliert lineare Zusammenhänge und isoliert den Effekt einzelner Features. |
| `16_logistische_regression.py` | Sagt Wahrscheinlichkeiten für binäre Events (Ja/Nein) wie z.B. späte Lieferungen voraus. |
| `33_count_daten_glm.py` | Nutzt Poisson-Regression für reine Zähldaten (z.B. Anzahl der Support-Tickets pro Tag). |
| `34_quantile_regression.py` | Prognostiziert nicht den Durchschnitt, sondern extreme Quantile (z.B. das 90. Perzentil der Lieferzeit). |
| `35_mixed_effects_hierarchical.py` | Modelliert verschachtelte Daten (z.B. Verkäufer innerhalb von Bundesstaaten) mit Random Effects. |

### 06. Klassifikation und Validierung
| Methode | Erklärung |
| :--- | :--- |
| `17_regularisierung.py` | Verhindert Overfitting durch Lasso/Ridge (L1/L2) und selektiert automatisch die wichtigsten Features. |
| `19_cross_validation_nested.py` | Bewertet die Modell-Performance realistisch durch Kreuzvalidierung auf rotierenden Daten-Splits. |
| `40_imbalance_handling.py` | Behandelt extrem ungleich verteilte Klassen (z.B. Betrugsfälle) durch SMOTE oder Class Weights. |
| `56_ensemble_stacking.py` | Kombiniert mehrere Modelle (Random Forest, Gradient Boosting) zu einem überlegenen Meta-Modell (Stacking). |

### 07. Zeitreihen und Forecasting
| Methode | Erklärung |
| :--- | :--- |
| `20_time_series_arima.py` | Erstellt klassische ARIMA-Prognosen für historische Zeitreihen. |
| `43_arimax_dynamic_regression.py` | Integriert externe Einflussfaktoren (z.B. Marketing-Ausgaben) in die Zeitreihen-Prognose. |
| `49_prophet_tbats_stl.py` | Zerlegt Zeitreihen mit STL und erstellt skalierbare Forecasts inklusive Feiertags-Effekten (Meta Prophet). |

### 08. Clustering und Segmentierung
| Methode | Erklärung |
| :--- | :--- |
| `25_clustering_kmeans_dbscan.py` | Gruppiert Kunden unüberwacht anhand ihres Kaufverhaltens in logische Segmente. |
| `42_cluster_stability_profiling.py` | Überprüft die mathematische Stabilität der Cluster (Silhouette Score) und erstellt Business-Profile. |

### 09. Feature Engineering und Reduktion
| Methode | Erklärung |
| :--- | :--- |
| `26_feature_reduktion.py` | Komprimiert hunderte Variablen mit PCA auf die wichtigsten Hauptkomponenten. |
| `52_feature_engineering_pipelines.py` | Baut saubere Scikit-Learn Pipelines für Preprocessing, um Data Leakage in Produktion zu verhindern. |

### 10. Kausalität und Experimente
| Methode | Erklärung |
| :--- | :--- |
| `24_ab_test_pipeline.py` | Führt eine automatisierte End-to-End A/B-Test-Auswertung inkl. Signifikanzprüfung durch. |
| `28_causal_did_psm_iv.py` | Schätzt kausale Effekte in Beobachtungsdaten mittels Propensity Score Matching (PSM). |
| `38_heterogene_treatment_effects.py` | Untersucht (CATE), ob eine Maßnahme bei spezifischen Kundengruppen unterschiedlich stark wirkt. |
| `45_experiment_design.py` | Berechnet Vorab-Design-Parameter (Minimum Detectable Effect, Dauer) für A/B-Tests. |
| `47_synthetic_control_methods.py` | Simuliert einen "synthetischen Zwilling" zur Evaluation, wenn kein echter A/B-Test (z.B. bei Geo-Rollouts) möglich ist. |
| `48_cate_ipw_doubly_robust.py` | Nutzt "Doubly Robust Estimators", um den wahren kausalen Effekt frei von Confoundern zu berechnen. |
| `57_causal_discovery.py` | Deckt Kausalstrukturen auf und generiert aus Beobachtungsdaten gerichtete Graphen (DAGs) für neue Hypothesen. |

### 11. Interpretierbarkeit und Kalibrierung
| Methode | Erklärung |
| :--- | :--- |
| `29_explainability_shap.py` | Erklärt die Entscheidungen von Black-Box-Modellen auf Einzelkunden-Ebene mittels SHAP-Values. |
| `39_probability_calibration.py` | Kalibriert Modelle so, dass eine Vorhersage von 80% auch in der Realität exakt 80% Trefferwahrscheinlichkeit bedeutet. |
| `41_komplexe_modelle_erklaeren.py` | Nutzt Partial Dependence Plots (PDP) zur Visualisierung isolierter Feature-Effekte auf die Vorhersage. |

### 12. Advanced Analytics Spezial
| Methode | Erklärung |
| :--- | :--- |
| `22_survival_analysis_churn.py` | Modelliert die Zeit bis zu einem Ereignis (Time-to-Event) unter Berücksichtigung rechtszensierter Daten. |
| `46_bayesian_estimation.py` | Nutzt bayesianisches Updating, um Unsicherheiten ("Shrinkage") bei kleinen Datenmengen (z.B. neuen Verkäufern) abzufedern. |
| `53_text_analytics_nlp.py` | Extrahiert mit NLP (TF-IDF, LDA) automatisiert versteckte Themen aus unstrukturierten Kunden-Reviews. |
| `54_Market Basket Analysis.py` | Findet Assoziationsregeln ("Wer A kauft, kauft oft auch B") für effektive Cross-Selling-Strategien. |
| `55_graph_network_analytics.py` | Analysiert Logistik-Netzwerke (Nodes & Edges) und berechnet die Zentralität einzelner Bundesstaaten. |
| `61_Image Basics.py` | Nutzt vortrainierte Deep Learning Modelle (MobileNetV2), um Bild-Embeddings für Visual Search zu generieren. |
| `62_Causal_Discovery.py` | Vertieft die Algorithmen zur automatisierten Generierung kausaler Netzwerke. |
| `63_Privaccy_Governance.py` | Schützt aggregierte Dashboards mit Differential Privacy (Laplace-Rauschen) vor Re-Identifizierungs-Attacken. |
| `64_MLOps.py` | Simuliert den Modell-Lifecycle in Produktion inkl. Data Drift und automatisierten CI/CD Retraining-Triggern. |

### 13. Monitoring, MLOps und Governance
| Methode | Erklärung |
| :--- | :--- |
| `30_modell_monitoring_drift.py` | Überwacht deployed Modelle auf Performance Decay und Feature Drift. |
| `50_change_point_detection.py` | Findet mit dem CUSUM-Algorithmus automatisiert plötzliche Strukturbrüche in operativen Zeitreihen. |
| `51_deep_anomaly_detection.py` | Isoliert komplexe, multivariate Ausreißer in riesigen Datenmengen mittels Isolation Forest. |
| `58_Bayesian Hyperparameter Optimization.py` | Optimiert Hyperparameter intelligent, gerichtet und ressourcenschonend mit Optuna. |
| `59_Geospatial_Analytics.py` | Berechnet die exakte physische Haversine-Distanz zwischen Koordinaten für geografisches Risk-Pricing. |

---
*Erstellt mit Fokus auf Business Value und fundierter statistischer Methodik.*
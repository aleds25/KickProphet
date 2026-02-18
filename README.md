# KickProphet 🚀
**Un Sistema di Machine Learning per la Predizione del Successo su Kickstarter**

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## 📖 Introduzione
**KickProphet** è uno strumento avanzato di Machine Learning progettato per stimare la probabilità di successo di una campagna Kickstarter *prima* del suo lancio. 

Analizzando parametri chiave come titolo, descrizione, categoria, obiettivo economico e durata, il sistema fornisce ai creator un indice di **"Fattibilità"** e feedback azionabili per ottimizzare la loro campagna. Il progetto affronta il problema con un approccio di **Classificazione Binaria Supervisionata** (Successo/Fallimento), utilizzando tecniche di NLP all'avanguardia e modelli ensemble.

### 👥 Autori
- **Alessio Del Sorbo**
- **Gianni Policola**

---

## ✨ Caratteristiche Principali
- **Ensemble Ibrido & Ottimizzato**: Combina la potenza di **LightGBM** e **XGBoost**, con iperparametri ottimizzati via **Optuna** per massima accuratezza e robustezza.
- **NLP Avanzato**: Analisi semantica dei testi (titolo e blurb) utilizzando Transformer (**all-MiniLM-L6-v2**) e riduzione dimensionale (**PCA**), oltre a metriche classiche come leggibilità e sentiment.
- **Feature Engineering Intelligente**: Encoding ciclico per le date, Target Encoding con smoothing per categorie ad alta cardinalità, e trasformazioni logaritmiche per variabili asimmetriche.
- **Pipeline Trustworthy**: 
    - **Guardrails**: Filtri euristici pre-imputazione per scartare input di bassa qualità (spam, gibberish).
    - **Semantic Gating**: Controllo di coerenza semantica tra descrizione e categoria.
    - **Calibrazione**: Probabilità calibrate (Isotonic Regression) per fornire stime di successo realistiche.
- **Produzione Ready**: Gestione in tempo reale dei tassi di cambio (API esterna) e robusta pipeline di inferenza.

---

## 📂 Struttura del Progetto
```
KickProphet/
├── app/
│   ├── data/               # Dataset grezzi e processati (non inclusi nel repo)
│   ├── models/             # Modelli addestrati (.joblib) e artefatti
│   ├── notebooks/          # Jupyter Notebooks per analisi ed esperimenti
│   │   ├── 1_Analisi_Esplorativa.ipynb
│   │   ├── 2_Preprocessing.ipynb
│   │   ├── 3_Model_Selection.ipynb
│   │   └── 4_Model_Evaluation.ipynb
│   ├── src/                # Codice sorgente Python
│   │   ├── config.py           # Configurazioni globali
│   │   ├── raw_dataset.py      # Unificazione dataset grezzi
│   │   ├── clean_dataset.py    # Pulizia e deduplicazione
│   │   ├── preprocessing.py    # Feature engineering pipeline
│   │   ├── train_model.py      # Script di addestramento
│   │   ├── predict.py          # Pipeline di inferenza
│   │   ├── tune_lgbm.py        # Tuning LightGBM
│   │   └── tune_xgb.py         # Tuning XGBoost
│   └── web/                # (Opzionale) Interfaccia Web
├── docs/                   # Documentazione aggiuntiva
├── requirements.txt        # Dipendenze del progetto
└── README.md               # Questo file
```

---

## 🚀 Installazione

1. **Clona il repository**:
   ```bash
   git clone https://github.com/aleds25/KickProphet.git
   cd KickProphet
   ```

2. **Crea un ambiente virtuale (opzionale ma consigliato)**:
   ```bash
   python -m venv venv
   source venv/bin/activate  # Su Windows: venv\Scripts\activate
   ```

3. **Installa le dipendenze**:
   ```bash
   pip install -r requirements.txt
   ```

---

## 🛠️ Utilizzo

L'intera pipeline può essere eseguita sequenzialmente tramite gli script nella cartella `app/src`. Assicurati di avere i file CSV dei dati nella cartella corretta (configurata in `app/src/config.py`).

### 1. Preparazione dei Dati
Unisci i file grezzi e pulisci il dataset:
```bash
python app/src/raw_dataset.py
python app/src/clean_dataset.py
```

### 2. Feature Engineering
Genera le feature per il training:
```bash
python app/src/preprocessing.py
```

### 3. Addestramento del Modello
Addestra l'ensemble LightGBM + XGBoost e salva gli artefatti:
```bash
python app/src/train_model.py
```
*Nota: Per rieseguire il tuning degli iperparametri, usa `tune_lgbm.py` e `tune_xgb.py`.*

### 4. Inferenza (Predizione)
Per stimare il successo di un nuovo progetto:
```bash
python app/src/predict.py
```
*Modifica lo script o usa le funzioni importate per passare i dati del tuo progetto.*

---

## 📊 Performance e Risultati

Il modello è stato valutato su un Test Set (20% dei dati) mai visto durante il training. I risultati confermano l'efficacia dell'approccio ibrido:

| Metrica | Valore | Descrizione |
| :--- | :--- | :--- |
| **Accuracy** | **81.4%** | Percentuale di previsioni corrette. |
| **ROC-AUC** | **0.895** | Eccellente capacità di discriminazione tra successo e fallimento. |
| **Precision** | **86.1%** | Basso tasso di falsi positivi (progetti predetti come successo che falliscono). |
| **Recall** | **83.7%** | Capacità di identificare la maggior parte dei progetti di successo reali. |
| **Brier Score** | **0.126** | Basso errore nelle probabilità calibrate. |

### Fattori Chiave di Successo
L'analisi dell'importanza delle feature ha rivelato che oltre al **Goal** (obiettivo economico), sono determinanti:
- **Qualità della presentazione**: Lunghezza e ricchezza semantica della descrizione.
- **Tempismo e Pianificazione**: Giorni di preparazione pre-lancio (`preparation_days`).
- **Coinvolgimento**: Presenza di video promozionali e aggiornamenti.

---

## 📄 Licenza
Distribuito sotto licenza **MIT**. Vedi il file `LICENSE` per maggiori informazioni.

---
*Università degli Studi di Salerno - Dipartimento di Informatica*

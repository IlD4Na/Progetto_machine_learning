# Progetto Diabete – Modello di Regressione per la Progressione della Malattia

Questo repository contiene il file `Progetto_Diabete_Machine_Learning.ipynb`, sviluppato come progetto di Machine Learning per la previsione della **progressione del diabete** a partire da dati clinici dei pazienti.

L’idea è simulare un caso reale in cui un’azienda sanitaria vuole integrare un modello predittivo nei propri sistemi per supportare i medici nelle decisioni cliniche.

---

## 🧠 Obiettivo del progetto

L’obiettivo è costruire un **modello di regressione** in grado di prevedere la progressione della malattia diabetica utilizzando variabili cliniche, tra cui:

- Età del paziente  
- Sesso  
- BMI (indice di massa corporea)  
- Pressione sanguigna media  
- Parametri ematici (colesterolo, LDL, HDL, trigliceridi, glicemia, ecc.)

Il modello finale viene ottimizzato, valutato e infine esportato per un utilizzo futuro in contesti applicativi.

---

## 📊 Dataset

Il progetto utilizza il **Diabetes dataset** fornito da `scikit-learn` tramite:

- `from sklearn.datasets import load_diabetes`  
- `diabetes = load_diabetes(as_frame=True, scaled=False)`

Caratteristiche del dataset:

- 442 pazienti  
- 10 variabili indipendenti (feature cliniche numeriche)  
- 1 variabile target (valore quantitativo che rappresenta la progressione della malattia)

Le feature includono:

1. `age` – età  
2. `sex` – genere  
3. `bmi` – indice di massa corporea  
4. `bp` – pressione sanguigna media  
5. `s1` – colesterolo sierico totale  
6. `s2` – lipoproteine a bassa densità (LDL)  
7. `s3` – lipoproteine ad alta densità (HDL)  
8. `s4` – rapporto colesterolo totale / HDL  
9. `s5` – trigliceridi  
10. `s6` – livello di glicemia  

Il target è una misura quantitativa della **progressione del diabete**.

---

## 🚀 Flusso di lavoro

Il file `progetto_diabete_machine_learning_.py` segue una pipeline ben strutturata.

### 1. Caricamento del dataset

- Caricamento del dataset con `load_diabetes(as_frame=True, scaled=False)`.  
- Creazione di:
  - `df` → DataFrame con le feature  
  - `target` → Serie con la variabile di output  
- Verifica di:
  - tipi di dato con `df.info()`  
  - statistiche descrittive con `df.describe()`  

---

### 2. Analisi esplorativa dei dati (EDA)

- Boxplot per le principali variabili cliniche (`age`, `bmi`, `bp`, `s1`–`s6`) per:
  - analizzare la distribuzione  
  - individuare outlier  

- Analisi della distribuzione del sesso (`sex`) tramite:
  - `value_counts()`  
  - grafico a barre  

- Identificazione degli outlier con filtri logici su:
  - `s6` (glicemia)  
  - `bmi`  
  - `s3` (HDL)  
  - `s2` (LDL)  
  - `s1` (colesterolo totale)  

- Calcolo della **matrice di correlazione** (feature + target) e visualizzazione tramite **heatmap**:
  - identificazione delle variabili più correlate al target (in particolare `bmi` e `s5`)  
  - analisi di collinearità tra feature (es. alta correlazione tra `s1` e `s2`)  

- Scatter plot tra:
  - `bmi` e `target`  
  - `s5` e `target`  
  - `s3` e `target`  

---

### 3. Pre-processing e divisione in train/test

- Creazione di:
  - `X` → tutte le feature  
  - `y` → target  

- Suddivisione in train e test (80% / 20%) con `train_test_split` (es. `random_state = 0`).  

- Standardizzazione delle feature con `StandardScaler`:
  - `fit` sul solo training set  
  - `transform` su `X_train` e `X_test`  

---

### 4. Modello base: Regressione Lineare Multipla

- Addestramento di un modello di `LinearRegression` sulle feature standardizzate.  
- Valutazione con funzione personalizzata `valutazione_modello` che stampa:
  - MSE (Mean Squared Error)  
  - R² (coefficiente di determinazione)  

- Valutazione su:
  - dati di training  
  - dati di test  

---

### 5. Regressione Polinomiale

- Creazione di feature polinomiali con `PolynomialFeatures` di grado 2, 3 e 4.  
- Per ogni grado:
  - trasformazione del training set nello spazio polinomiale  
  - addestramento di `LinearRegression`  
  - valutazione con MSE e R² su train e test  

---

### 6. Selezione delle variabili e Lasso Regression

- Applicazione di **Lasso Regression** su feature polinomiali.  
- Variazione del parametro `alpha` nel range `np.arange(0, 4, 0.1)`.  
- Valutazione con `valutazione_modello` su train e test.  
- Migliore compromesso ottenuto con Lasso di grado 2.  

---

### 7. Ottimizzazione con GridSearchCV

- Pipeline con `PolynomialFeatures` + `Lasso`.  
- Ricerca su griglia `alpha` tramite `GridSearchCV`.  
- Valutazione con `cv=5`, metrica `r2`.  
- Recupero di `best_params_` e `best_score_`.

---

### 8. Modello finale e Learning Curve

- Addestramento finale su tutto il training set.  
- Valutazione sul test set.  
- Tracciamento della **Learning Curve** per analisi del modello.

---

### 9. Esportazione del modello

- Esportazione con `pickle` in `diabetes_progression_model.pkl`.  
- Caricamento successivo con `pickle.load()`.

---

## 🧩 Competenze sviluppate

Questo progetto dimostra e rafforza competenze in:

### 🔍 Data Analysis & EDA

- Analisi di distribuzioni, outlier, correlazioni  
- Utilizzo di `pandas`, `matplotlib`, `seaborn`  

### 🧪 Machine Learning (Regressione)

- Regressione lineare multipla  
- Regressione polinomiale  
- Lasso Regression con regolarizzazione  

### ⚙️ Model Selection & Tuning

- Utilizzo di `GridSearchCV` per ottimizzare iperparametri  
- Valutazione con MSE e R²  
- Analisi delle Learning Curve  

### 🔄 Preprocessing

- Standardizzazione con `StandardScaler`  
- Suddivisione train/test evitando data leakage  

### 🔐 MLOps di base

- Esportazione e riutilizzo del modello con `pickle`  

---

## 🛠️ Requisiti

L’ambiente di esecuzione richiede:

- Python 3.x  

Librerie principali:

- `pandas`  
- `numpy`  
- `scikit-learn`  
- `matplotlib`  
- `seaborn`  

Installazione delle dipendenze:

```
pip install pandas numpy scikit-learn matplotlib seaborn
```

---

## ▶️ Come eseguire il progetto

### 1. Clona il repository

```
git clone https://github.com/<tuo-username>/<nome-repo>.git
cd <nome-repo>
```

### 2. Esegui lo script

Da terminale:

```
python progetto_diabete_machine_learning_.py
```

Oppure:

- Jupyter Notebook / JupyterLab  
- Google Colab  
- IDE (PyCharm, VS Code)

---

## 📁 Struttura del repository

```
/nome-repo/
├── progetto_diabete_machine_learning_.py
├── diabetes_progression_model.pkl  # (opzionale, generato dallo script)
└── README.md
```

---

## 📌 Note finali

Questo progetto è un esercizio completo di regressione e ottimizzazione in ambito sanitario e può servire da base per:

- applicazioni cliniche reali  
- deployment di modelli predittivi  
- approfondimenti su regolarizzazione e selezione delle feature  

---

© 2025 – Progetto didattico

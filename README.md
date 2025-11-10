# Analisi Predittiva del Successo Accademico: Un Approccio di Machine Learning

La presente repository contiene il codice, i dati e i risultati relativi al progetto di tesi di **Simone Magli** che esplora la capacità dei modelli di Machine Learning di predire il rendimento accademico (GPA) degli studenti universitari basandosi sulle loro abitudini di vita.

***

## Panoramica del Progetto

L'obiettivo della tesi è confrontare modelli di complessità crescente per determinare l'approccio più efficace e stabile nella predizione del **Grade Point Average (GPA)**.

Gli algoritmi considerati per la predizione sono:
1.  **Regressione Lineare Multipla** (Modello di base)
2.  **Random Forest Regressor** (Algoritmo Ensemble)
3.  **Support Vector Regression (SVR)** con kernel RBF (Modello Kernel-based)

I modelli 2 e 3 sono stati ottimizzati tramite **Hyperparameter Tuning (Grid Search)**.

***

## Risultati Chiave

Il risultato più significativo è che la **Regressione Lineare Multipla** (il modello più semplice) si è dimostrata il più **stabile e performante**.

* **Forte Predittore:** L'unica variabile a mostrare una correlazione forte e positiva con il GPA è stata l'**Ore di Studio Giornaliere (`Study_Hours_Per_Day`)**.
* **Correlazioni Deboli:** Il sonno e l'attività fisica non hanno mostrato una correlazione significativa con il GPA.

### Metriche Finali (Test Set)
I tre modelli migliori sono stati testati su un set di dati inedito.

| Modello | $R^2$ (Test Set) | MSE (Test Set) |
| :--- | :--- | :--- |
| **Linear Regression (Multivariata)** | **0.5317** | **0.0391** |
| SVR (tuned) | 0.5302 | 0.0393 |
| Random Forest (tuned) | 0.5073 | 0.0412 |
***

## Analisi Statistica e Stabilità

Un'analisi statistica condotta su **30 ripetizioni** (k-fold) ha confermato la maggiore stabilità e l'accuratezza media della Regressione Lineare rispetto all'SVR ottimizzato.

* **Regressione Lineare (Media):** $R^2$ medio: $0.5288 \pm 0.0186$
* **SVR (tuned) (Media):** $R^2$ medio: $0.5146 \pm 0.0192$

**Conclusione:** Per questo dataset, la regressione lineare è risultata preferibile per la sua maggiore semplicità, interpretabilità e per le performance predittive sostanzialmente equivalenti, se non leggermente superiori, a quelle di un modello più complesso come SVR.

Simone_Magli_Tesi_Informatica_Per_Il_Management/ ├── Tesi/ │ ├── ProgettoNew.py # Codice sorgente del progetto in Python │ ├── student_lifestyle_dataset.csv # Dataset utilizzato │ ├── simone.magli.presentazione.tesi.pdf # Presentazione della tesi │ ├── tesi_magli_FINAL.pdf # Testo integrale della tesi │ └── image/ # Cartella contenente tutti i grafici e i risultati └── README.md



# -*- coding: utf-8 -*-
"""
Created on Tue Aug 19 19:16:48 2025

@author: simon
"""


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
import numpy as np
from scipy import stats

# ===============================
# FASE 1 - CARICAMENTO
# ===============================
file_path = 'student_lifestyle_dataset.csv'
df = pd.read_csv(file_path)

# ===============================
# FASE 2 - PULIZIA
# ===============================
# Rimuovo righe con valori mancanti
df_cleaned = df.dropna()

# Rimuovo ID (inutile per la predizione)
df_cleaned = df_cleaned.drop(columns=['Student_ID'])

# One-hot encoding per Stress_Level (variabile categorica)
df_cleaned = pd.get_dummies(df_cleaned, columns=['Stress_Level'], drop_first=False)

print(f"Numero di righe dopo la pulizia: {df_cleaned.shape[0]}")

# ===============================
# FASE 3 - EDA AVANZATA
# ===============================

# --- Statistiche descrittive ---
print("\n=== Statistiche descrittive ===")
print(df_cleaned.describe())

# --- Distribuzioni univariate (istogrammi + boxplot) ---
num_vars = ['GPA', 'Study_Hours_Per_Day', 'Sleep_Hours_Per_Day', 'Physical_Activity_Hours_Per_Day']

for col in num_vars:
    plt.figure(figsize=(12,5))

    # Istogramma con KDE
    plt.subplot(1,2,1)
    sns.histplot(df_cleaned[col], kde=True, bins=20)
    plt.title(f"Distribuzione di {col}")

    # Boxplot per outlier detection
    plt.subplot(1,2,2)
    sns.boxplot(x=df_cleaned[col])
    plt.title(f"Boxplot di {col}")

    plt.show()

# --- GPA in funzione del livello di stress ---
plt.figure(figsize=(8,5))
sns.boxplot(x='Stress_Level_High', y='GPA', data=df_cleaned)
plt.title("Distribuzione GPA in base allo Stress (High=1, Low=0)")
plt.show()

# --- GPA in funzione dell’attività fisica ---
plt.figure(figsize=(8,5))
sns.scatterplot(x='Physical_Activity_Hours_Per_Day', y='GPA', data=df_cleaned, hue='Stress_Level_High')
plt.title("GPA vs Ore di attività fisica (colorato per Stress)")
plt.show()

# --- Distribuzioni condizionate ---
plt.figure(figsize=(8,5))
sns.violinplot(x=pd.cut(df_cleaned['Study_Hours_Per_Day'], bins=[0,2,4,6,8,12]), y='GPA', data=df_cleaned)
plt.title("Distribuzione GPA per fasce di studio")
plt.xticks(rotation=45)
plt.show()

plt.figure(figsize=(8,5))
sns.violinplot(x=pd.cut(df_cleaned['Sleep_Hours_Per_Day'], bins=[0,4,6,8,10,12]), y='GPA', data=df_cleaned)
plt.title("Distribuzione GPA per fasce di sonno")
plt.xticks(rotation=45)
plt.show()

# --- Pairplot con colore per stress ---
sns.pairplot(df_cleaned[['GPA','Study_Hours_Per_Day','Sleep_Hours_Per_Day',
                         'Physical_Activity_Hours_Per_Day','Stress_Level_High']],
             hue='Stress_Level_High')
plt.show()

# --- MATRICE DI CORRELAZIONE ---
plt.figure(figsize=(10, 6))
corr_matrix = df_cleaned.corr()
sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Matrice di correlazione tra variabili")
plt.show()
# ===============================
# FASE 4 - SPLITTING
# ===============================
target = 'GPA'
X = df_cleaned.drop(columns=[target])
y = df_cleaned[target]

X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

print(f"Training set: {X_train.shape[0]} campioni")
print(f"Validation set: {X_val.shape[0]} campioni")
print(f"Test set: {X_test.shape[0]} campioni")
# ===============================
# FASE 5 - REGRESSIONE LINEARE MULTIVARIATA
# ===============================

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Modello multivariato con tutte le feature
linreg_model = LinearRegression()
linreg_model.fit(X_train, y_train)

# Predizione sul validation set
y_lin_pred = linreg_model.predict(X_val)

print("\n=== Regressione Lineare Multivariata ===")
print("Coefficiente/i:")
for col, coef in zip(X_train.columns, linreg_model.coef_):
    print(f"  {col}: {coef:.4f}")
print(f"Intercetta: {linreg_model.intercept_:.4f}")

# Metriche
r2_lin = r2_score(y_val, y_lin_pred)
mse_lin = mean_squared_error(y_val, y_lin_pred)
print(f"R^2 (Validation): {r2_lin:.4f}")
print(f"MSE (Validation): {mse_lin:.4f}")

# Scatter plot Predicted vs Actual
plt.figure(figsize=(10, 6))
plt.scatter(y_val, y_lin_pred, color='blue', alpha=0.6, label='Predizioni')
plt.plot([y_val.min(), y_val.max()], [y_val.min(), y_val.max()], 'r--', label='Perfetta predizione (y=x)')
plt.xlabel('Valori reali GPA')
plt.ylabel('Valori predetti GPA')
plt.title('Regressione Lineare Multivariata: Predicted vs Actual')
plt.legend()
plt.show()

# Analisi residui
residui_lin = y_val - y_lin_pred
plt.figure(figsize=(10, 6))
sns.histplot(residui_lin, kde=True, bins=20, color="blue")
plt.title("Distribuzione residui - Regressione Lineare Multivariata")
plt.xlabel("Residui")
plt.show()


# ===============================
# FASE 6 - ALTRI MODELLI PREDITTIVI
# ===============================

# -------------------------------
# Random Forest Regressor
# -------------------------------
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

y_rf_pred = rf_model.predict(X_val)

print("\n=== Random Forest Regressor ===")
print(f"R^2: {r2_score(y_val, y_rf_pred):.4f}")
print(f"MSE: {mean_squared_error(y_val, y_rf_pred):.4f}")

plt.figure(figsize=(10, 6))
plt.scatter(y_val, y_rf_pred, color='green', alpha=0.6, label="Predizioni")
plt.plot([y_val.min(), y_val.max()], [y_val.min(), y_val.max()], 'r--', label="Perfetta predizione (y=x)")
plt.xlabel('Valori reali GPA')
plt.ylabel('Predizioni GPA')
plt.title('Random Forest: Valori reali vs Predetti')
plt.legend()
plt.show()

# --- Analisi residui RF ---
residui_rf = y_val - y_rf_pred


plt.figure(figsize=(10, 6))
sns.histplot(residui_rf, kde=True, bins=20, color="blue")
plt.title("Distribuzione residui - Random Forest")
plt.xlabel("Residui")
plt.show()


# -------------------------------
# Support Vector Regression (SVR con kernel RBF)
# -------------------------------
svr_model = make_pipeline(StandardScaler(), SVR(kernel='rbf', C=100, epsilon=0.1))
svr_model.fit(X_train, y_train)

y_svr_pred = svr_model.predict(X_val)

print("\n=== Support Vector Regression (RBF Kernel) ===")
print(f"R^2: {r2_score(y_val, y_svr_pred):.4f}")
print(f"MSE: {mean_squared_error(y_val, y_svr_pred):.4f}")

plt.figure(figsize=(10, 6))
plt.scatter(y_val, y_svr_pred, color='purple', alpha=0.6, label="Predizioni")
plt.plot([y_val.min(), y_val.max()], [y_val.min(), y_val.max()], 'r--', label="Perfetta predizione (y=x)")
plt.xlabel('Valori reali GPA')
plt.ylabel('Predizioni GPA')
plt.title('SVR: Valori reali vs Predetti')
plt.legend()
plt.show()

# --- Analisi residui SVR ---
residui_svr = y_val - y_svr_pred

plt.figure(figsize=(10, 6))
sns.histplot(residui_svr, kde=True, bins=20, color="orange")
plt.title("Distribuzione residui - SVR")
plt.xlabel("Residui")
plt.show()

from sklearn.model_selection import GridSearchCV

# ===============================
# PARAMETER TUNING - RANDOM FOREST
# ===============================
param_grid_rf = {
    'n_estimators': [100, 200, 500],
    'max_depth': [None, 10, 20, 30],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'max_features': [None, 'sqrt', 'log2']
}

rf = RandomForestRegressor(random_state=42)

grid_search_rf = GridSearchCV(
    estimator=rf,
    param_grid=param_grid_rf,
    cv=5,                # 5-fold cross-validation
    scoring='r2',        # valuta con R^2
    n_jobs=-1,           # usa tutti i core
    verbose=2
)

grid_search_rf.fit(X_train, y_train)

print("\n=== Best Parameters Random Forest ===")
print(grid_search_rf.best_params_)
print(f"Best R^2 (CV): {grid_search_rf.best_score_:.4f}")

best_rf = grid_search_rf.best_estimator_

# Valutazione sul validation set
y_rf_best = best_rf.predict(X_val)
print(f"R^2 (Validation): {r2_score(y_val, y_rf_best):.4f}")
print(f"MSE (Validation): {mean_squared_error(y_val, y_rf_best):.4f}")


# ===============================
# PARAMETER TUNING - SVR
# ===============================
param_grid_svr = {
    'svr__C': [0.1, 1, 10, 100],
    'svr__epsilon': [0.01, 0.1, 0.2, 0.5],
    'svr__kernel': ['rbf', 'poly'],
    'svr__gamma': ['scale', 'auto']
}

svr_pipeline = make_pipeline(StandardScaler(), SVR())

grid_search_svr = GridSearchCV(
    estimator=svr_pipeline,
    param_grid=param_grid_svr,
    cv=5,
    scoring='r2',
    n_jobs=-1,
    verbose=2
)

grid_search_svr.fit(X_train, y_train)

print("\n=== Best Parameters SVR ===")
print(grid_search_svr.best_params_)
print(f"Best R^2 (CV): {grid_search_svr.best_score_:.4f}")

best_svr = grid_search_svr.best_estimator_

# Valutazione sul validation set
y_svr_best = best_svr.predict(X_val)
print(f"R^2 (Validation): {r2_score(y_val, y_svr_best):.4f}")
print(f"MSE (Validation): {mean_squared_error(y_val, y_svr_best):.4f}")

from sklearn.metrics import r2_score, mean_squared_error

# ===============================
# CONFRONTO MODELLI TUNED
# ===============================
results = {}

# Random Forest tuned
results["Random Forest (tuned)"] = {
    "y_true": y_val,
    "y_pred": y_rf_best,
    "R2": r2_score(y_val, y_rf_best),
    "MSE": mean_squared_error(y_val, y_rf_best)
}

# SVR tuned
results["SVR (tuned)"] = {
    "y_true": y_val,
    "y_pred": y_svr_best,
    "R2": r2_score(y_val, y_svr_best),
    "MSE": mean_squared_error(y_val, y_svr_best)
}

# ===============================
# 1. Predicted vs Actual
# ===============================
plt.figure(figsize=(12,6))
for model_name, res in results.items():
    plt.scatter(res["y_true"], res["y_pred"], alpha=0.6, label=model_name)

plt.plot([y_val.min(), y_val.max()], [y_val.min(), y_val.max()], "k--", lw=2, label="Perfetta predizione (y=x)")
plt.xlabel("Valori reali GPA")
plt.ylabel("Valori predetti GPA")
plt.title("Predicted vs Actual (Validation Set)")
plt.legend()
plt.show()

# ===============================
# 2. Residual Plot
# ===============================
plt.figure(figsize=(12,6))
for model_name, res in results.items():
    residui = res["y_true"] - res["y_pred"]
    plt.scatter(res["y_pred"], residui, alpha=0.6, label=model_name)

plt.axhline(0, color="red", linestyle="--")
plt.xlabel("Valori predetti GPA")
plt.ylabel("Residui")
plt.title("Residual Plot (Validation Set)")
plt.legend()
plt.show()

# ===============================
# 3. Distribuzione residui
# ===============================
plt.figure(figsize=(12,6))
for model_name, res in results.items():
    residui = res["y_true"] - res["y_pred"]
    sns.kdeplot(residui, label=model_name, fill=True, alpha=0.3)

plt.axvline(0, color="red", linestyle="--")
plt.xlabel("Residui")
plt.title("Distribuzione residui")
plt.legend()
plt.show()

# ===============================
# 4. Barplot metriche (senza warning)
# ===============================
metrics_df = pd.DataFrame({
    model: {"R2": res["R2"], "MSE": res["MSE"]}
    for model, res in results.items()
}).T.reset_index().rename(columns={"index": "Model"})

fig, ax = plt.subplots(1, 2, figsize=(14,6))

# --- R² ---
sns.barplot(
    x="Model", y="R2", hue="Model",
    data=metrics_df, ax=ax[0], palette="viridis", legend=False
)
ax[0].set_title("R² dei modelli")
ax[0].set_ylim(0, 1)
ax[0].set_xlabel("")
ax[0].set_ylabel("R²")

# --- MSE ---
sns.barplot(
    x="Model", y="MSE", hue="Model",
    data=metrics_df, ax=ax[1], palette="magma", legend=False
)
ax[1].set_title("MSE dei modelli")
ax[1].set_xlabel("")
ax[1].set_ylabel("MSE")

plt.suptitle("Confronto Random Forest (tuned) vs SVR (tuned)", fontsize=14)
plt.show()

# ===============================
# FASE FINALE - VALUTAZIONE SUL TEST SET
# ===============================

print("\n=== VALUTAZIONE FINALE SU TEST SET ===")

test_results = {}

# --- Regressione Lineare Multivariata ---
y_lin_test = linreg_model.predict(X_test)
test_results["Linear Regression (Multivariata)"] = {
    "R2": r2_score(y_test, y_lin_test),
    "MSE": mean_squared_error(y_test, y_lin_test)
}


# --- Random Forest (tuned) ---
y_rf_test = best_rf.predict(X_test)
test_results["Random Forest (tuned)"] = {
    "R2": r2_score(y_test, y_rf_test),
    "MSE": mean_squared_error(y_test, y_rf_test)
}

# --- SVR (tuned) ---
y_svr_test = best_svr.predict(X_test)
test_results["SVR (tuned)"] = {
    "R2": r2_score(y_test, y_svr_test),
    "MSE": mean_squared_error(y_test, y_svr_test)
}

# --- Stampa i risultati ---
for model, metrics in test_results.items():
    print(f"\n{model}")
    print(f"  R²  = {metrics['R2']:.4f}")
    print(f"  MSE = {metrics['MSE']:.4f}")

# --- Grafico Predicted vs Actual (Test set) ---
# --- Grafico Predicted vs Actual (Test set) ---
plt.figure(figsize=(12,6))
plt.scatter(y_test, y_lin_test, alpha=0.6, label="Linear Regression (Multivariata)")
plt.scatter(y_test, y_rf_test, alpha=0.6, label="Random Forest (tuned)")
plt.scatter(y_test, y_svr_test, alpha=0.6, label="SVR (tuned)")

plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], "k--", lw=2, label="Perfetta predizione (y=x)")
plt.xlabel("Valori reali GPA (Test set)")
plt.ylabel("Valori predetti GPA")
plt.title("Predicted vs Actual - Test Set")
plt.legend()
plt.show()

# ===============================
# ANALISI STATISTICA SU K RIPETIZIONI (Linear Regression vs SVR)
# ===============================

k = 30  # numero di ripetizioni (k >= 10, meglio 30 o 50)
alpha = 0.05  # livello di confidenza (95%)

results_stats = {}

for model_name, model in [("Linear Regression", LinearRegression()), 
                          ("SVR (tuned)", best_svr)]:
    
    r2_scores = []
    mse_scores = []
    
    for seed in range(k):
        # nuovo split casuale ogni volta
        X_train, X_test_split, y_train, y_test_split = train_test_split(
            X, y, test_size=0.3, random_state=seed
        )
        
        # addestramento
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test_split)
        
        # metriche
        r2_scores.append(r2_score(y_test_split, y_pred))
        mse_scores.append(mean_squared_error(y_test_split, y_pred))
    
    # conversione in array numpy
    r2_array = np.array(r2_scores)
    mse_array = np.array(mse_scores)
    
    # intervalli di confidenza
    conf_r2 = stats.t.interval(
        1 - alpha, len(r2_array)-1, loc=np.mean(r2_array), scale=stats.sem(r2_array)
    )
    conf_mse = stats.t.interval(
        1 - alpha, len(mse_array)-1, loc=np.mean(mse_array), scale=stats.sem(mse_array)
    )
    
    results_stats[model_name] = {
        "R2_mean": r2_array.mean(),
        "R2_std": r2_array.std(),
        "R2_conf": conf_r2,
        "MSE_mean": mse_array.mean(),
        "MSE_std": mse_array.std(),
        "MSE_conf": conf_mse,
        "R2_all": r2_array,
        "MSE_all": mse_array
    }
    
    # stampa riepilogo
    print(f"\n=== {model_name} ===")
    print(f"R² medio: {r2_array.mean():.4f} ± {r2_array.std():.4f}")
    print(f"IC 95% R²: {conf_r2}")
    print(f"MSE medio: {mse_array.mean():.4f} ± {mse_array.std():.4f}")
    print(f"IC 95% MSE: {conf_mse}")

# ===============================
# GRAFICI COMPARATIVI
# ===============================
plt.figure(figsize=(14,6))

# Istogramma R²
plt.subplot(1,2,1)
sns.histplot(results_stats["Linear Regression"]["R2_all"], bins=10, color="blue", kde=True, label="Linear Regression", alpha=0.5)
sns.histplot(results_stats["SVR (tuned)"]["R2_all"], bins=10, color="orange", kde=True, label="SVR (tuned)", alpha=0.5)
plt.title("Distribuzione R² su k ripetizioni")
plt.xlabel("R²")
plt.legend()

# Boxplot R²
plt.subplot(1,2,2)
sns.boxplot(data=[results_stats["Linear Regression"]["R2_all"], 
                  results_stats["SVR (tuned)"]["R2_all"]],
            palette=["blue","orange"])
plt.xticks([0,1], ["Linear Regression", "SVR (tuned)"])
plt.title("Boxplot R² su k ripetizioni")

plt.tight_layout()
plt.show()

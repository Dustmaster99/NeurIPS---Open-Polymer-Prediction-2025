# -*- coding: utf-8 -*-
"""
Created on Fri Jul 25 13:51:30 2025

@author: eosjo
"""
import os
os.chdir(r'C:\Kegle_Jojo\NeurIPS---Open-Polymer-Prediction-2025\Kegle_Competition')

import numpy as np
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from scikeras.wrappers import KerasRegressor
from tensorflow.keras.optimizers import Adam, SGD
import click
import logging
from pathlib import Path
from dotenv import find_dotenv, load_dotenv
import pandas as pd
import re
import yaml
from pathlib import Path

from scipy.stats import shapiro
from rdkit import Chem
import networkx as nx
import matplotlib.pyplot as plt
from networkx.algorithms import isomorphism
import warnings
from src.data.pre_processing import load_config
from src.data.pre_processing import load_input
from src.data.pre_processing import save_csv
import pickle
from src.data.pre_processing import remover_colunas_zeros
from scipy.stats import normaltest
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import pearsonr
import os
import matplotlib.pyplot as plt
import seaborn as sns
import os
from pathlib import Path
from scipy.stats import boxcox
from src.features.statistic_tests_to_features import importar_lista_pickle
import statsmodels.api as sm # estimação de modelos
from statstests.process import stepwise
import mlflow
import mlflow.sklearn
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error
from statsmodels.stats.stattools import durbin_watson
from src.models.box_cox_model import remover_caracteres_especiais
from src.models.box_cox_model import limpar_colunas_dataframe
import mlflow
import mlflow.sklearn
from scikeras.wrappers import KerasRegressor
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.metrics import r2_score, mean_squared_error


import os
import matplotlib.pyplot as plt
import mlflow
#%%
def plot_y_real_vs_pred(y_test, y_pred, nome_arquivo):
    """
    Gera e salva um gráfico de y_real vs y_predito, e loga como artefato no MLflow.

    Parâmetros:
    - y_test: valores reais
    - y_pred: valores preditos
    - nome_arquivo: nome do arquivo (sem extensão .png)

    Salva o arquivo em ../../data/plot/<nome_arquivo>.png
    """
    # Criação do gráfico
    plt.figure(figsize=(6, 6))
    plt.scatter(y_test, y_pred, alpha=0.6, color='blue')
    plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], '--r')
    plt.xlabel('Valor Real')
    plt.ylabel('Valor Predito')
    plt.title('Real vs Predito (Amostra de Teste)')
    plt.grid(True)

    # Caminho relativo para salvar o plot
    pasta_plot = os.path.join("..", "..", "data", "plot")
    os.makedirs(pasta_plot, exist_ok=True)

    caminho_plot = os.path.join(pasta_plot, f"{nome_arquivo}.png")
    plt.savefig(caminho_plot)
    plt.close()

    # Logar no MLflow
    mlflow.log_artifact(caminho_plot)

#%%
def main():
    
    mlflow.set_experiment("Neural_Network with grid search")
    
    
    # 1. Carregar o arquivo de configuração
    config = load_config('config.yaml')
    
    #Paths
    path_interim = config['paths']['interim']
    pickle_path =  config['paths']['element_count_column_list']
    path_processed = config['paths']['processed']
    path_plot = config['paths']['feature_plot']
    
    #data
    train_name = config['data']['train']
    test_name = config['data']['test']
    smile_name = config['data']['smile']
    pickle_name = config['data']['element_count_column_list']
    
    #targets
    
    targets = []
    y_TG = config ['data'] ['target_col_TG']
    targets.append(y_TG)
    y_FFV = config ['data'] ['target_col_FFV']
    targets.append(y_FFV)
    y_Tc = config ['data'] ['target_col_Tc']
    targets.append(y_Tc)
    y_density = config ['data'] ['target_col_Density']
    targets.append(y_density)
    y_Rg = config ['data'] ['target_col_Rg']
    targets.append(y_Rg)

    #modelParameters
    scaler = config ['Neural_network']['scaler']
    scoring = config ['Neural_network']['scoring']
    hidden_layers = config ['Neural_network']['hidden_layers']
    neurons = config ['Neural_network']['neurons']
    activation = config ['Neural_network']['activation']
    optimizer = config ['Neural_network']['optimizer']
    epochs = config ['Neural_network']['epochs']
    batch_size = config ['Neural_network']['batch_size']
    t_size = config ['Neural_network']['t_size']
    folder_n = config ['Neural_network']['folder_n']
    
    # 2. Carregar os dados
    df_train = load_input(relative_path=path_processed , filename=train_name)
    df_test = load_input(relative_path=path_processed , filename=test_name)
    Features = importar_lista_pickle(pickle_path,pickle_name)
    #Features = substituir_espacos_lista(Features)
    #Features = remover_parenteses(Features)
    Features = remover_caracteres_especiais(Features)
    
    
    #return Features
    # 3. criar dataframe de features
    #df_y_TG = df_train.dropna(subset=[y_TG])
    #df_y_TG = substituir_espacos_colunas(df_y_TG)
    
    df_y_FFV = df_train.dropna(subset=[y_FFV])
    df_y_FFV = limpar_colunas_dataframe(df_y_FFV)
    
    #df_y_Tc = df_train.dropna(subset=[y_Tc])
    #df_y_Tc = substituir_espacos_colunas(y_Tc)

    #df_y_density = df_train.dropna(subset=[y_density])
    #df_y_density = substituir_espacos_colunas(df_y_density)
    
    #df_y_Rg = df_train.dropna(subset=[y_Rg])
    #df_y_Rg = substituir_espacos_colunas(df_y_Rg)


    # Variáveis X e Y
    X = df_y_FFV[Features]
    y = df_y_FFV[y_FFV]
    
    if(scaler == True):
        scaler = StandardScaler()
        X = scaler.fit_transform(X)
        
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size= t_size)
    
    #função de constru~~ao do modelo
    def create_model(hidden_layers=1, neurons=64, activation='relu', optimizer='adam'):
        model = Sequential()
        model.add(Dense(neurons, input_dim=X.shape[1], activation=activation))
        for _ in range(hidden_layers - 1):
            model.add(Dense(neurons, activation=activation))
        model.add(Dense(1))  # camada de saída
        model.compile(loss='mse', optimizer=optimizer)
        return model
    
    
    # 3. Wrap do modelo com KerasRegressor
    model = KerasRegressor(build_fn=create_model, verbose=0)
    
    # 4. Definir a grade de parâmetros
    param_grid = {
        'model__hidden_layers': hidden_layers,
        'model__neurons': neurons,
        'model__activation': activation,
        'optimizer': optimizer,
        'epochs': epochs,
        'batch_size': batch_size
    }
    
    with mlflow.start_run():
    
        grid = GridSearchCV(estimator=model, param_grid=param_grid, scoring= scoring, cv= folder_n)
        
        # 5. Buscar melhores parâmetros
        grid_result = grid.fit(X_train, y_train)
        
        # Mostrar melhores resultados
        print("Melhor score :", -grid_result.best_score_)
        print("Melhores parâmetros:", grid_result.best_params_)
        
        best_score = -grid_result.best_score_
        
        for param_name, param_value in grid_result.best_params_.items():
            mlflow.log_param(param_name, param_value)
        
        
        # Avaliar no teste
        # Melhor modelo (KerasRegressor) após GridSearchCV
        best_model = grid.best_estimator_

        # Acessar o modelo Keras real treinado
        keras_model = best_model.model_
        
        # Logar o modelo corretamente
        mlflow.keras.log_model(keras_model, artifact_path="keras_model")
        
        y_pred = best_model.predict(X_test)
        print("MSE no teste:", mean_squared_error(y_test, y_pred))
       
        # Métricas
        r2 = r2_score(y_test, y_pred)
        mse = mean_squared_error(y_test, y_pred)
        
        # 🔹 Logar métricas
        mlflow.log_metric("r2_score", r2)
        mlflow.log_metric("mse", mse)
        
        plot_y_real_vs_pred(y_test,y_pred,"comparação.png")
        
    return y_pred
  
#%%
if __name__ == '__main__':


# Oculta todos os DeprecationWarnings
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    y_pred = main()    
        
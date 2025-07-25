# -*- coding: utf-8 -*-
"""
Created on Fri Jul 18 16:58:03 2025

@author: eosjo
"""
import os
os.chdir(r'C:\Kegle_Jojo\NeurIPS---Open-Polymer-Prediction-2025\Kegle_Competition')

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

#%%
def substituir_espacos_colunas(df):
    """
    Substitui espaços por underscores nos nomes das colunas de um DataFrame.

    Parâmetros:
    - df: pandas.DataFrame - DataFrame cujas colunas serão renomeadas.

    Retorna:
    - pandas.DataFrame - DataFrame com nomes de colunas ajustados.
    """
    df.columns = df.columns.str.replace(' ', '_')
    return df
#%%
def substituir_espacos_lista(colunas):
    """
    Substitui espaços por underscores em cada elemento de uma lista de strings.

    Parâmetros:
    - colunas: list[str] - Lista de nomes de colunas.

    Retorna:
    - list[str] - Lista com espaços substituídos por underscores.
    """
    return [col.replace(' ', '_') for col in colunas]

#%%
def remover_parenteses(lista):
    """
    Remove os caracteres '(' e ')' de cada string em uma lista.

    Parâmetros:
    - lista: list[str] - Lista de strings.

    Retorna:
    - list[str] - Lista com parênteses removidos.
    """
    return [s.replace('(', '').replace(')', '') for s in lista]

#%%
def remover_caracteres_especiais(lista):
    """
    Remove os caracteres '-', '_', ' ', '.' de cada string em uma lista.

    Parâmetros:
    - lista: list[str]

    Retorna:
    - list[str]
    """
    return [s.replace('-', '').replace('_', '').replace(' ', '').replace('.', '').replace('(', '').replace(')', '') for s in lista]
#%%
def limpar_colunas_dataframe(df):
    """
    Remove '-', '_', ' ', '.' dos nomes das colunas de um DataFrame.

    Parâmetros:
    - df: pandas.DataFrame

    Retorna:
    - pandas.DataFrame com nomes de colunas limpos.
    """
    df.columns = [re.sub(r'[-_ .()]', '', col) for col in df.columns]
    return df


#%%
def main():
    
    mlflow.set_experiment("meu_experimento_box_cox")
    
    
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
    p_value_sig = config ['box-cox'] ['p_value_sig']
    p_value_sig_str = config ['box-cox'] ['p_value_sig_str']
    apply_stepwise = config ['box-cox'] ['stepwise']
    stepwise_str = config ['box-cox'] ['stepwise_str']
    r2 = config ['box-cox'] ['r2']
    r2_adj = config ['box-cox'] ['r2_adj']
    f_stat = config ['box-cox'] ['f_stat']
    f_pvalue = config ['box-cox'] ['f_pvalue']
    rmse_str = config ['box-cox'] ['rmse_str']
    mae_str = config ['box-cox'] ['mae_str']
    durbin_watson_str = config ['box-cox'] ['durbin_watson_str']
    modelo = config ['box-cox'] ['modelo']
    modelo_str = config ['box-cox'] ['modelo_str']
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


    with mlflow.start_run():
        
        formula = y_FFV + " ~ " + ' + '.join(Features)
        modelo_box_cox_FFV = sm.OLS.from_formula(formula,df_y_FFV).fit()
        if(apply_stepwise == True):
            modelo_box_cox_FFV = stepwise(modelo_box_cox_FFV, pvalue_limit=p_value_sig)
        # print model parameters
        print(modelo_box_cox_FFV.summary(alpha=p_value_sig))
 
        # Log de parâmetros
        mlflow.log_param(modelo_str, modelo)
        mlflow.log_param(p_value_sig_str, p_value_sig)
        mlflow.log_param(stepwise_str, apply_stepwise)
        
        
        # Log de métricas
        
        # R² e R² ajustado
        
        mlflow.log_metric(r2, modelo_box_cox_FFV.rsquared)
        mlflow.log_metric(r2_adj, modelo_box_cox_FFV.rsquared_adj)
    
        # Estatística F e p-valor global
        mlflow.log_metric(f_stat, modelo_box_cox_FFV.fvalue)
        mlflow.log_metric(f_pvalue, modelo_box_cox_FFV.f_pvalue)
    
        # RMSE e MAE
        rmse = np.sqrt(mean_squared_error(modelo_box_cox_FFV.model.endog, y_pred = modelo_box_cox_FFV.fittedvalues))
        mae = mean_absolute_error(modelo_box_cox_FFV.model.endog,modelo_box_cox_FFV.fittedvalues)
       
        mlflow.log_metric(rmse_str, rmse)
        mlflow.log_metric(mae_str, mae)
    
        # Durbin-Watson (autocorrelação dos resíduos)
        dw = durbin_watson(modelo_box_cox_FFV.resid)
        mlflow.log_metric(durbin_watson_str, dw)

        
        
    
    return modelo_box_cox_FFV
    

#%%
if __name__ == '__main__':


# Oculta todos os DeprecationWarnings
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    modelo_box_cox_FFV = main()    
    
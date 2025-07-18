# -*- coding: utf-8 -*-
"""
Created on Fri Jul 18 17:03:40 2025

@author: eosjo
"""


import click
import logging
from pathlib import Path
from dotenv import find_dotenv, load_dotenv
import pandas as pd

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
#%%
def apply_normal_test(df, colunas):
    """
    Aplica o teste de Shapiro-Wilk para verificar normalidade
    em colunas numéricas de um DataFrame.

    Parâmetros:
    - df: pandas.DataFrame
        O DataFrame contendo os dados
    - colunas: list[str]
        Lista com os nomes das colunas a serem testadas

    Resultado:
    - Print do nome da coluna, estatística do teste e p-valor
    """
    for coluna in colunas:
        dados = df[coluna].dropna()  # remover valores ausentes
        estatistica, p_valor = normaltest(dados)

        print(f"🧪 Coluna: {coluna}")
        print(f"   Estatística : {estatistica:.4f} | p-valor: {p_valor:.4f}")

        if p_valor > 0.05:
            print("   ✅ Dados parecem normalmente distribuídos.\n")
        else:
            print("   ⚠️ Dados NÃO seguem uma distribuição normal.\n")
   

#%%
def importar_lista_pickle(relative_path: str, filename: str):
    """
    Salva uma lista em um arquivo .pkl (pickle).

    Parâmetros:
    - lista: list
        A lista que será exportada.
    - caminho_arquivo: str
        Caminho e nome do arquivo de saída (ex: 'saida.pkl').
    """
    # Resolve the path to the current file
    base_path = Path(__file__).resolve()
    print(base_path )
    # Navigate two directories up
    grandparent_dir = base_path.parent.parent.parent
    print(grandparent_dir)
    
    # Combine with the relative path and file name
    full_path = grandparent_dir / relative_path / filename
    
    with open(full_path, 'rb') as f:
        objeto = pickle.load(f)
    return objeto

#%%
def main():
    # 1. Carregar o arquivo de configuração
    config = load_config('config.yaml')
    path_interim = config['paths']['interim']
    pickle_path =  config['paths']['element_count_column_list']
    path_processed = config['paths']['processed']
    train_name = config['data']['train']
    test_name = config['data']['test']
    smile_name = config['data']['smile']
    pickle_name = config['data']['element_count_column_list']
    

    # 2. Carregar os dados
    df_train = load_input(relative_path=path_processed , filename=train_name)
    df_test = load_input(relative_path=path_processed , filename=test_name)
    Features = importar_lista_pickle(pickle_path,pickle_name)
    
    
    # 3. Aplicar teste de shapiro wilkes
    apply_normal_test(df_train,Features)
    
    
    
#%%
if __name__ == '__main__':


# Oculta todos os DeprecationWarnings
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    main()    

    



# -*- coding: utf-8 -*-
import os
import click
import logging
from pathlib import Path
from dotenv import find_dotenv, load_dotenv
import pandas as pd

import yaml
from pathlib import Path

from rdkit import Chem
import networkx as nx
import matplotlib.pyplot as plt
from networkx.algorithms import isomorphism
import warnings



#%%
def remover_colunas_zeros(df):
    """
    Remove colunas do DataFrame que possuem todos os valores iguais a zero.

    Parâmetros:
    - df: pandas.DataFrame
        DataFrame a ser processado.

    Retorna:
    - df_filtrado: pandas.DataFrame
        DataFrame sem as colunas onde todos os valores são zero.
    """
    df_filtrado = df.loc[:, (df != 0).any(axis=0)]
    return df_filtrado



#%%
def load_config(filename: str):
    """
    Loads a YAML file located in the parent directory of the script calling this function.
    
    Args:
        filename (str): The name of the YAML file (e.g., 'config.yaml')
    
    Returns:
        dict: Parsed YAML content
    """
    # Get the path of the script that called this function
    caller_path = Path(__file__).resolve()
    
    # Move to the parent directory
    parent_dir = caller_path.parent.parent
    
    # Build full path to the YAML file
    yaml_path = parent_dir / filename
    
    # Load and return YAML content
    with open(yaml_path, 'r',encoding='utf-8') as f:
        data = yaml.safe_load(f)
        
    return data
#%%
def load_input(relative_path: str, filename: str) -> pd.DataFrame:
    """
    Loads a CSV file located two directories above the current script,
    in a subpath defined by `relative_path`, with the specified `filename`.

    Args:
        relative_path (str): Subfolder path from the grandparent directory (e.g., 'data/raw')
        filename (str): Name of the CSV file (e.g., 'dataset.csv')

    Returns:
        pd.DataFrame: The loaded CSV as a DataFrame
    """
    # Resolve the path to the current file
    base_path = Path(__file__).resolve()
    
    # Navigate two directories up
    grandparent_dir = base_path.parent.parent.parent
    
    # Combine with the relative path and file name
    full_path = grandparent_dir / relative_path / filename
    # Load and return the CSV as a DataFrame
    return pd.read_csv(full_path)


#%%
def save_csv(df:object, relative_path: str, filename: str):
    """
    Saves a DataFrame to a CSV file at the specified path.

    Parameters:
    - df: pandas.DataFrame - DataFrame to be saved.
    - caminho: str - Path to the directory where the file will be saved.
    - nome_arquivo: str - Name of the CSV file (default: 'dataset.csv').
    """
    # Resolve the path to the current file
    base_path = Path(__file__).resolve()
    
    # Navigate two directories up
    grandparent_dir = base_path.parent.parent.parent
    
    # Combine with the relative path and file name
    full_path = grandparent_dir / relative_path / filename

    # Salva o DataFrame como CSV
    df.to_csv(full_path, index=False)

    print(f"Arquivo salvo em: {full_path}")


#%%
def main():
    # 1. Carregar o arquivo de configuração
    config = load_config('config.yaml')
    
    path_raw = config['paths']['raw']
    path_interim = config['paths']['interim']
    train_name = config['data']['train']
    test_name = config['data']['test']

    # 2. Carregar os dados
    df_train = load_input(relative_path=path_raw , filename=train_name)
    df_test = load_input(relative_path=path_raw , filename=test_name)
    
    # 3. Aplicar tratamento inicial a base de dados: To be done........
    
    # 4. exporta a base de dados tratada como CSV para o caminho
    save_csv(df_train,path_interim, train_name)
    save_csv(df_test,path_interim, test_name)
    
   

#%%
if __name__ == '__main__':


# Oculta todos os DeprecationWarnings
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    df = main()

# -*- coding: utf-8 -*-
"""
Created on Fri Jul 18 17:03:40 2025

@author: eosjo
"""

import os
os.chdir(r'C:\Kegle_Jojo\NeurIPS---Open-Polymer-Prediction-2025\Kegle_Competition')

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
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import pearsonr
import os
import matplotlib.pyplot as plt
import seaborn as sns
import os
from pathlib import Path



#%%
def pearson_correlation_table(df, targets, features):
    """
    Calcula a correlação de Pearson entre múltiplas variáveis target e várias features.
    
    Parâmetros:
    - df: pd.DataFrame com os dados
    - targets: lista de 5 strings, nomes das colunas target
    - features: lista de strings, nomes das colunas features
    
    Retorna:
    - pd.DataFrame com índice targets, colunas features, valores de correlação
    - pd.DataFrame com p-valores correspondentes (mesma estrutura)
    """
    corr_matrix = pd.DataFrame(index=targets, columns=features, dtype=float)
    pval_matrix = pd.DataFrame(index=targets, columns=features, dtype=float)
    
    for target in targets:
        for feature in features:
            # Remove NaNs em ambas as colunas para cálculo correto
            valid = df[[target, feature]].dropna()
            if len(valid) > 1:
                corr, pval = pearsonr(valid[target], valid[feature])
                corr_matrix.loc[target, feature] = corr
                pval_matrix.loc[target, feature] = pval
            else:
                corr_matrix.loc[target, feature] = None
                pval_matrix.loc[target, feature] = None
    
    return corr_matrix, pval_matrix

#%%



def plot_corr_heatmaps(corr_matrix, relative_path, n_features_por_plot=5, title_prefix='Heatmap Parte'):
    """
    Plota e salva heatmaps de correlação em partes, dividindo as colunas em grupos.

    Parâmetros:
    - corr_matrix: pd.DataFrame - matriz de correlação (targets x features)
    - relative_path: str - caminho relativo onde os arquivos .jpg serão salvos
    - n_features_por_plot: int - número máximo de features por plot
    - title_prefix: str - prefixo do título de cada gráfico
    """
    # Resolve base path e constrói caminho completo
    base_path = Path(__file__).resolve()
    grandparent_dir = base_path.parent.parent.parent
    output_dir = grandparent_dir / relative_path
    output_dir.mkdir(parents=True, exist_ok=True)

    total_features = corr_matrix.shape[1]
    features = corr_matrix.columns.tolist()

    for i in range(0, total_features, n_features_por_plot):
        subset = features[i:i + n_features_por_plot]
        corr_subset = corr_matrix[subset]

        plt.figure(figsize=(1.5 * len(subset), 6))
        sns.heatmap(corr_subset, annot=True, fmt=".2f", cmap='coolwarm', center=0)
        titulo = f"{title_prefix} {i // n_features_por_plot + 1}"
        plt.title(titulo)
        plt.xlabel('Features')
        plt.ylabel('Targets')
        plt.tight_layout()

        filename = f"{title_prefix.replace(' ', '_')}_{i // n_features_por_plot + 1}.jpg"
        full_path = output_dir / filename
        plt.savefig(full_path, dpi=300)
        print(f"Salvo em: {full_path}")

        plt.close()

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
def plot_target_vs_features(df, y_target, X_cols, caminho , n_cols=5, figsize=(15, 10), ):
    """
    Plota gráficos de dispersão da variável target em função de todas as colunas especificadas.
    
    Parâmetros:
    - df: pd.DataFrame - DataFrame com os dados
    - y_target: str - nome da coluna target (eixo Y)
    - X_cols: list - lista com os nomes das colunas (features, eixo X)
    - n_cols: int - número de colunas por linha na grade de subplots
    - figsize: tuple - tamanho da figura
    """
    
    # Resolve the path to the current file
    base_path = Path(__file__).resolve()
    
    # Navigate two directories up
    grandparent_dir = base_path.parent.parent.parent
    
    filename = str(y_target) + '.jpg'
    
    # Combine with the relative path and file name
    full_path = grandparent_dir /  caminho / filename
    
    
    n_plots = len(X_cols)
    n_rows = -(-n_plots // n_cols)  # ceil divisão inteira

    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
    axes = axes.flatten()

    for i, col in enumerate(X_cols):
        sns.scatterplot(data=df, x=col, y=y_target, ax=axes[i])
        axes[i].set_title(f'{y_target} vs {col}')
        axes[i].set_xlabel(col)
        axes[i].set_ylabel(y_target)

    # Remove subplots vazios
    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])

    plt.tight_layout()
    plt.savefig(full_path, format='jpg', dpi=300)  # dpi é a resolução
    plt.show()
    # Salva o gráfico no arquivo JPG
    
    
#%%
def main():
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

    

    # 2. Carregar os dados
    df_train = load_input(relative_path=path_processed , filename=train_name)
    df_test = load_input(relative_path=path_processed , filename=test_name)
    Features = importar_lista_pickle(pickle_path,pickle_name)
    #return Features
    # 3. criar dataframe de features
    df_y_TG = df_train.dropna(subset=[y_TG])
    df_y_FFV = df_train.dropna(subset=[y_FFV])
    df_y_Tc = df_train.dropna(subset=[y_Tc])
    df_y_density = df_train.dropna(subset=[y_density])
    df_y_Rg = df_train.dropna(subset=[y_Rg])
    
    
    # 4. Aplicar teste de shapiro wilkes
    apply_normal_test(df_train,Features)
    
    
    # 5.Caminho para salvar graficos
    
    
    # 6. Imprimir target x features
    plot_target_vs_features(df_y_TG ,y_TG, Features,path_plot,)
    plot_target_vs_features(df_y_FFV, y_FFV, Features,path_plot)
    plot_target_vs_features(df_y_Tc, y_Tc, Features,path_plot)
    plot_target_vs_features(df_y_density,y_density, Features,path_plot)
    plot_target_vs_features(df_y_Rg, y_Rg, Features,path_plot)
    
    
    corr_matrix, pval_matrix = pearson_correlation_table(df_train, targets, Features)
    plot_corr_heatmaps(corr_matrix, path_plot)

    
    
#%%
if __name__ == '__main__':


# Oculta todos os DeprecationWarnings
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    main()    
    

    



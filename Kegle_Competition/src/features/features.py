# -*- coding: utf-8 -*-
"""
Created on Fri Jul 18 15:59:17 2025

@author: eosjo
"""

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
from src.data.pre_processing import load_config
from src.data.pre_processing import load_input
from src.data.pre_processing import save_csv



#%%
def mol_to_nx(smiles):
    """
    Converte uma molécula em SMILES para um grafo NetworkX.
    - Nós: átomos com rótulo 'índice:elemento'
    - Arestas: ligações codificadas como:
        0 = Aromática
        1 = Simples
        2 = Dupla
       -1 = Outro
    """
    mol = Chem.MolFromSmiles(smiles)
    G = nx.Graph()

    # Adiciona os nós (átomos)
    for atom in mol.GetAtoms():
        idx = atom.GetIdx()
        symbol = atom.GetSymbol()
        G.add_node(idx, element=symbol, label=f"{idx}:{symbol}")

    # Mapeamento de tipos de ligação
    bond_type_map = {
    'AROMATIC': 0,
    'SINGLE': 1,
    'DOUBLE': 2,
    'TRIPLE': 3  # ligação tripla
}

    # Adiciona as arestas (ligações)
    for bond in mol.GetBonds():
        a1 = bond.GetBeginAtomIdx()
        a2 = bond.GetEndAtomIdx()
        bond_type_str = str(bond.GetBondType()).upper()
        bond_label = bond_type_map.get(bond_type_str, -1)
        G.add_edge(a1, a2, bond_label=bond_label)

    return G
#%%
def plot_molecule_graph(G, title="Grafo Molecular com Rótulos Codificados"):
    """
    Plota o grafo molecular com:
    - Rótulos dos nós: 'índice:elemento'
    - Rótulos das arestas: códigos inteiros de tipo de ligação
    """
    pos = nx.spring_layout(G, seed=42)
    node_labels = nx.get_node_attributes(G, 'label')
    edge_labels = {(u, v): d['bond_label'] for u, v, d in G.edges(data=True)}

    plt.figure(figsize=(8, 6))
    nx.draw_networkx_nodes(G, pos, node_color='lightblue', node_size=800)
    nx.draw_networkx_edges(G, pos)
    nx.draw_networkx_labels(G, pos, labels=node_labels, font_size=10, font_weight='bold')
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=9)
    plt.title(title)
    plt.axis('off')
    plt.tight_layout()
    plt.show()


#%%
def get_node_edge_match(group_data,group_name):
    def normalize(text):
        return (text.lower()
                .replace('á', 'a').replace('ó', 'o').replace('í', 'i')
                .replace('é', 'e').replace('ú', 'u').replace('ç', 'c')
                .replace('ã', 'a').replace('õ', 'o').replace('ê', 'e')
                .replace('â', 'a').replace('ô', 'o').strip())

    group = normalize(group_name)

    

    if group not in group_data:
        raise ValueError(f"Grupo funcional ou substância desconhecida: {group_name}")

    smile, allowed_elements, allowed_bonds = group_data[group]
    H = mol_to_nx(smile)
    

    node_match = lambda n1, n2: (n1['element'] == n2['element']) and (n1['element'] in allowed_elements)
    edge_match = lambda e1, e2: (e1['bond_label'] == e2['bond_label']) and (e1['bond_label'] in allowed_bonds)

    return H, smile, node_match, edge_match
#%% 

def generate_components_count(df_train:object , column_name:str):
# 1. Iterar sobre as amostras (limitado ao head(1) como no original)
    for idx, smile_input in df_train[column_name].items():
         G = mol_to_nx(smile_input)
         #plot_molecule_graph(G)
    
         for nome, (smile, _, _) in group_data.items():
             try:
                 # Obter padrão e grafo do grupo funcional
                 H, smile_to_compare, node_match, edge_match = get_node_edge_match(group_data, nome)
                 #plot_molecule_graph(H)
    
                 #print(f"\nAnalisando grupo: {nome}")
                 #print(f"Molécula alvo: {smile_input}")
                 #print(f"Padrão SMARTS/SMILES: {smile_to_compare}")
    
                 mol = Chem.MolFromSmiles(smile_input)
                 pattern = Chem.MolFromSmarts(smile_to_compare)
    
                 if mol is None or pattern is None:
                     #print(f"[AVISO] Falha ao gerar Mol para '{nome}'. Pulando.")
                     continue
    
                 # Buscar subestruturas que correspondem ao padrão
                 matches = mol.GetSubstructMatches(pattern)
                 #print(f"Matches encontrados: {matches}")
    
                 # Nome da nova coluna
                 new_column = nome + ' matches'
                 df_train.at[idx, new_column] = len(matches)
    
             except Exception as e:
                 print(f"[ERRO] Falha ao processar '{nome}': {e}")
    
     # 4. (Opcional) Salvar ou retornar df_train modificado
    print("\nProcessamento concluído para a primeira molécula do conjunto.")
    print(df_train.head(1))
    return df_train





#%% Data containing Smile and restrictions for chemical components

group_data = {
    'metano':                ('C',              ['C', 'H'],            [1]),
    'etano':                 ('CC',             ['C', 'H'],            [1]),
    'eteno':                 ('C=C',            ['C', 'H'],            [1, 2]),
    'etino':                 ('C#C',            ['C', 'H'],            [1, 3]),
    'metanol':              ('CO',             ['C', 'O', 'H'],       [1]),
    'etanol':               ('CCO',            ['C', 'O', 'H'],       [1]),
    'acido acetico':        ('CC(=O)O',        ['C', 'O', 'H'],       [1, 2]),
    'formaldeido':          ('C=O',            ['C', 'O', 'H'],       [1, 2]),
    'acetona':              ('CC(=O)C',        ['C', 'O', 'H'],       [1, 2]),
    'formamida':            ('C(=O)N',         ['C', 'O', 'N', 'H'],  [1, 2]),
    'dimetilamina':         ('CN(C)C',         ['C', 'N', 'H'],       [1]),
    'fenol':                ('c1ccccc1O',      ['C', 'O', 'H'],       [1, 2]),
    'anilina':              ('c1ccccc1N',      ['C', 'N', 'H'],       [1, 2]),
    'acido benzoico':       ('c1ccccc1C(=O)O', ['C', 'O', 'H'],       [1, 2]),
    'tolueno':              ('Cc1ccccc1',      ['C', 'H'],            [1]),
    'benzeno':              ('c1ccccc1',       ['C', 'H'],            [1, 2]),
    'acido formico':        ('O=CO',           ['C', 'O', 'H'],       [1, 2]),
    'acetato de etila':     ('CC(=O)OCC',      ['C', 'O', 'H'],       [1, 2]),
    'etilamina':            ('CCN',            ['C', 'N', 'H'],       [1]),
    'glicerol':             ('C(C(CO)O)O',     ['C', 'O', 'H'],       [1]),

    # Inorgânicas simples
    'agua':                 ('O',              ['O', 'H'],            [1]),
    'amonia':               ('N',              ['N', 'H'],            [1]),
    'acido cloridrico':     ('[H]Cl',          ['H', 'Cl'],           [1]),
    'acido sulfurico':      ('OS(=O)(=O)O',    ['O', 'S', 'H'],       [1, 2]),
    'acido nitrico':        ('O=N(=O)O',       ['O', 'N', 'H'],       [1, 2]),
    'dioxido de carbono':   ('O=C=O',          ['C', 'O'],            [2]),
    'gas hidrogenio':       ('[H][H]',         ['H'],                 [1]),
    'gas oxigenio':         ('O=O',            ['O'],                 [2]),
    'gas nitrogenio':       ('N#N',            ['N'],                 [3]),
    'cloro gas':            ('ClCl',           ['Cl'],                [1]),
    'metanoato de sodio':   ('[Na+].[O-]C=O',  ['Na', 'O', 'C', 'H'], [1, 2]),
    'cloreto de sodio':     ('[Na+].[Cl-]',    ['Na', 'Cl'],          [1]),
    'acido fosforico':      ('OP(=O)(O)O',     ['O', 'P', 'H'],       [1, 2]),
    'hidroxido de sodio':   ('[Na+].[OH-]',    ['Na', 'O', 'H'],      [1])
}

#%%
def main():
    # 1. Carregar o arquivo de configuração
    config = load_config('config.yaml')
    path_interim = config['paths']['interim']
    path_processed = config['paths']['processed']
    train_name = config['data']['train']
    test_name = config['data']['test']
    smile_name = config['data']['smile']

    # 2. Carregar os dados
    df_train = load_input(relative_path=path_interim , filename=train_name)
    df_test = load_input(relative_path=path_interim , filename=test_name)
    
    df_train = generate_components_count(df_train,smile_name)
    
    save_csv(df_train,path_processed, train_name)
    save_csv(df_test,path_processed, test_name)

     

#%%
if __name__ == '__main__':


# Oculta todos os DeprecationWarnings
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    df = main()


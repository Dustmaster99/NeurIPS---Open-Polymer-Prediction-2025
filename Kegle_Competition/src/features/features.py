# -*- coding: utf-8 -*-
"""
Created on Fri Jul 18 15:59:17 2025

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
from rdkit.Chem import Descriptors

#%%
def exportar_lista_pickle(lista:list, relative_path: str, filename: str):
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
    
    # Navigate two directories up
    grandparent_dir = base_path.parent.parent.parent
    
    # Combine with the relative path and file name
    full_path = grandparent_dir / relative_path / filename

    
    with open(full_path, 'wb') as f:
        pickle.dump(lista, f)
    print(f"✅ Lista exportada com sucesso para: {full_path}")

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

    smile, allowed_elements, allowed_bonds, categoria = group_data[group]
    H = mol_to_nx(smile)
    

    node_match = lambda n1, n2: (n1['element'] == n2['element']) and (n1['element'] in allowed_elements)
    edge_match = lambda e1, e2: (e1['bond_label'] == e2['bond_label']) and (e1['bond_label'] in allowed_bonds)

    return H, smile, node_match, edge_match


#%%
def generate_components_count_categoria(df_train: object, column_name: str, group: dict):
    import pandas as pd
    import numpy as np

    # Guarda resultados parciais
    match_counts = {}       # (idx, nome_match) -> n_matches
    category_counts = {}    # (idx, categoria_count) -> total

    for idx, smile_input in df_train[column_name].items():
        for nome, (smile, _, _, categoria) in group.items():
            try:
                H, smile_to_compare, node_match, edge_match = get_node_edge_match(group, nome)

                mol = Chem.MolFromSmiles(smile_input)
                pattern = Chem.MolFromSmarts(smile_to_compare)

                if mol is None or pattern is None:
                    continue

                matches = mol.GetSubstructMatches(pattern)
                n_matches = len(matches)

                match_col = f"{nome} matches"
                match_counts[(idx, match_col)] = n_matches

                if n_matches > 0:
                    cat_col = f"{categoria}_count"
                    current_val = category_counts.get((idx, cat_col), 0)
                    category_counts[(idx, cat_col)] = current_val + 1

            except Exception as e:
                print(f"[ERRO] Falha ao processar '{nome}': {e}")

    # Converte os resultados acumulados em DataFrames
    df_matches = pd.DataFrame([
        {"index": idx, "column": col, "value": val}
        for (idx, col), val in match_counts.items()
    ])
    df_categorias = pd.DataFrame([
        {"index": idx, "column": col, "value": val}
        for (idx, col), val in category_counts.items()
    ])

    def pivot_results(df_partial):
        return df_partial.pivot(index="index", columns="column", values="value")

    df_matches_pivot = pivot_results(df_matches)
    df_categorias_pivot = pivot_results(df_categorias)

    # Junta tudo com o df original
    df_result = df_train.copy()
    if not df_matches_pivot.empty:
        df_result = df_result.join(df_matches_pivot, how='left')
    if not df_categorias_pivot.empty:
        df_result = df_result.join(df_categorias_pivot, how='left')

    df_result.fillna(0, inplace=True)
    df_result = remover_colunas_zeros(df_result)

    # Lista de colunas que realmente ficaram no resultado
    remaining_columns = list(df_matches_pivot.columns) if not df_matches_pivot.empty else []

    print("\nProcessamento concluído para a primeira molécula do conjunto (versão otimizada).")
    return df_result, remaining_columns

#%%
def generate_components_count(df_train:object , column_name:str, group:dict):
# 1. Iterar sobre as amostras
    for idx, smile_input in df_train[column_name].items():
         G = mol_to_nx(smile_input)
         #plot_molecule_graph(G)
         generated_columns_list =[]
         for nome, (smile, _, _, categoria) in group.items():
             try:
                 # Obter padrão e grafo do grupo funcional
                 H, smile_to_compare, node_match, edge_match = get_node_edge_match(group, nome)
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
                 generated_columns_list.append(new_column)
             except Exception as e:
                 print(f"[ERRO] Falha ao processar '{nome}': {e}")
    
     # 4. (Opcional) Salvar ou retornar df_train modificado
    print("\nProcessamento concluído para a primeira molécula do conjunto.")
     # 5. Remover colunas em que todos os valores de contagem são igual a zero
    df_train = remover_colunas_zeros(df_train)
    remaining_columns = [col for col in generated_columns_list if col in df_train.columns]
    
    
    return df_train,remaining_columns
#%% Data containing Smile and restrictions for chemical components


alcanos_data = {
    'metano':       ('C',          ['C', 'H'],    [1]),
    'etano':        ('CC',         ['C', 'H'],    [1]),
    'propano':      ('CCC',        ['C', 'H'],    [1]),
    'butano':       ('CCCC',       ['C', 'H'],    [1]),
    'pentano':      ('CCCCC',      ['C', 'H'],    [1]),
    'hexano':       ('CCCCCC',     ['C', 'H'],    [1]),
    'heptano':      ('CCCCCCC',    ['C', 'H'],    [1]),
    'octano':       ('CCCCCCCC',   ['C', 'H'],    [1]),
    'nonano':       ('CCCCCCCCC',  ['C', 'H'],    [1]),
    'decano':       ('CCCCCCCCCC', ['C', 'H'],    [1]),
}


alcohols_data = {
    'metanol':       ('CO',          ['C', 'O', 'H'], [1]),
    'etanol':        ('CCO',         ['C', 'O', 'H'], [1]),
    'propanol-1':    ('CCCO',        ['C', 'O', 'H'], [1]),
    'propanol-2':    ('CC(O)C',      ['C', 'O', 'H'], [1]),
    'butanol-1':     ('CCCCO',       ['C', 'O', 'H'], [1]),
    'butanol-2':     ('CCC(O)C',     ['C', 'O', 'H'], [1]),
    'isobutanol':    ('CC(C)CO',     ['C', 'O', 'H'], [1]),
    'terc-butanol':  ('CC(C)(C)O',   ['C', 'O', 'H'], [1]),
    'pentanol-1':    ('CCCCCO',      ['C', 'O', 'H'], [1]),
    'hexanol-1':     ('CCCCCCO',     ['C', 'O', 'H'], [1]),
    'glicerol':      ('C(C(CO)O)O',  ['C', 'O', 'H'], [1])  # poliol, mas incluído por conter função álcool
}

fenois_data = {
    'fenol':               ('c1ccccc1O',       ['C', 'O', 'H'],      [1, 2]),
    'cresol o-':           ('Oc1ccccc1C',      ['C', 'O', 'H'],      [1, 2]),
    'cresol m-':           ('Oc1ccc(cc1)C',    ['C', 'O', 'H'],      [1, 2]),
    'cresol p-':           ('Oc1ccc(cc1)C',    ['C', 'O', 'H'],      [1, 2]),
    'resorcinol':          ('c1cc(O)ccc1O',    ['C', 'O', 'H'],      [1, 2]),
    'hidroquinona':        ('c1cc(O)cc(O)c1',  ['C', 'O', 'H'],      [1, 2]),
    'catecol':             ('c1cc(O)cc(O)c1',  ['C', 'O', 'H'],      [1, 2]),
    'naftol-1':            ('Oc1cccc2c1cccc2', ['C', 'O', 'H'],      [1, 2]),
    'naftol-2':            ('Oc1ccc2ccccc2c1', ['C', 'O', 'H'],      [1, 2]),
    'pirocatecol':         ('c1cc(O)cc(O)c1',  ['C', 'O', 'H'],      [1, 2]),
}

eteres_data = {
    'éter dimetílico':        ('COC',           ['C', 'O', 'H'],       [1]),
    'éter etílico (éter dietílico)': ('CCOCC',    ['C', 'O', 'H'],       [1]),
    'éter metil fenílico':    ('COc1ccccc1',    ['C', 'O', 'H'],       [1, 2]),
    'éter fenil vinílico':    ('c1ccccc1OC=C',  ['C', 'O', 'H'],       [1, 2]),
    'éter tetra-hidrofurano': ('C1COCC1',       ['C', 'O', 'H'],       [1]),
    'éter dimetílico cíclico': ('C1COC1',       ['C', 'O', 'H'],       [1]),
    'éter dietílico cíclico': ('C1CCOCC1',      ['C', 'O', 'H'],       [1]),
}


aldeidos_data = {
    'formaldeído':       ('C=O',           ['C', 'O', 'H'],       [1, 2]),
    'acetaldeído':       ('CC=O',          ['C', 'O', 'H'],       [1, 2]),
    'propionaldeído':    ('CCC=O',         ['C', 'O', 'H'],       [1, 2]),
    'butiraldeído':      ('CCCC=O',        ['C', 'O', 'H'],       [1, 2]),
    'valeraldeído':      ('CCCCC=O',       ['C', 'O', 'H'],       [1, 2]),
    'benzaldeído':       ('c1ccccc1C=O',   ['C', 'O', 'H'],       [1, 2]),
    'cinnamaldeído':     ('C=CCc1ccccc1',  ['C', 'O', 'H'],       [1, 2]),
}

cetonas_data = {
    'acetona':            ('CC(=O)C',          ['C', 'O', 'H'],       [1, 2]),
    'propanona':          ('CC(=O)C',          ['C', 'O', 'H'],       [1, 2]),  # sinônimo de acetona
    'butanona (metiletilcetona)': ('CCC(=O)C',   ['C', 'O', 'H'],       [1, 2]),
    'pentanona-2':        ('CCCC(=O)C',        ['C', 'O', 'H'],       [1, 2]),
    'hexanona-2':         ('CCCCC(=O)C',       ['C', 'O', 'H'],       [1, 2]),
    'ciclopentanona':     ('C1CCC(=O)C1',      ['C', 'O', 'H'],       [1, 2]),
    'ciclohexanona':      ('C1CCCCC(=O)C1',    ['C', 'O', 'H'],       [1, 2]),
    'benzofenona':        ('c1ccc(cc1)C(=O)c2ccccc2', ['C', 'O', 'H'], [1, 2]),
    'fluorenona':         ('c1ccc2c(c1)C(=O)c3ccccc23', ['C', 'O', 'H'], [1, 2]),
}


acidos_carboxilicos_data = {
    'ácido fórmico':           ('O=CO',             ['C', 'O', 'H'],       [1, 2]),
    'ácido acético':           ('CC(=O)O',          ['C', 'O', 'H'],       [1, 2]),
    'ácido propanoico':        ('CCC(=O)O',          ['C', 'O', 'H'],       [1, 2]),
    'ácido butanoico':         ('CCCC(=O)O',         ['C', 'O', 'H'],       [1, 2]),
    'ácido pentanoico':        ('CCCCC(=O)O',        ['C', 'O', 'H'],       [1, 2]),
    'ácido benzoico':          ('c1ccccc1C(=O)O',    ['C', 'O', 'H'],       [1, 2]),
    'ácido oxálico':           ('O=C(O)C(=O)O',      ['C', 'O', 'H'],       [1, 2]),
    'ácido cítrico':           ('C(C(=O)O)C(CC(=O)O)C(=O)O', ['C', 'O', 'H'],  [1, 2]),
    'ácido málico':            ('C(C(=O)O)CC(O)C(=O)O', ['C', 'O', 'H'],     [1, 2]),
    'ácido succínico':         ('C(CC(=O)O)C(=O)O',  ['C', 'O', 'H'],       [1, 2]),
}

esteres_data = {
    'acetato de etila':        ('CC(=O)OCC',          ['C', 'O', 'H'],       [1, 2]),
    'acetato de metila':       ('CC(=O)OC',           ['C', 'O', 'H'],       [1, 2]),
    'benzoato de metila':      ('c1ccccc1C(=O)OC',    ['C', 'O', 'H'],       [1, 2]),
    'benzoato de etila':       ('c1ccccc1C(=O)OCC',   ['C', 'O', 'H'],       [1, 2]),
    'formiato de metila':      ('O=COC',              ['C', 'O', 'H'],       [1, 2]),
    'formiato de etila':       ('O=COCC',             ['C', 'O', 'H'],       [1, 2]),
    'propionato de metila':    ('CCC(=O)OC',          ['C', 'O', 'H'],       [1, 2]),
    'propionato de etila':     ('CCC(=O)OCC',         ['C', 'O', 'H'],       [1, 2]),
    'butirato de metila':      ('CCCC(=O)OC',         ['C', 'O', 'H'],       [1, 2]),
    'butirato de etila':       ('CCCC(=O)OCC',        ['C', 'O', 'H'],       [1, 2]),
    'laurato de metila':       ('CCCCCCCCCCCC(=O)OC', ['C', 'O', 'H'],       [1, 2]),
    'laurato de etila':        ('CCCCCCCCCCCC(=O)OCC',['C', 'O', 'H'],       [1, 2])
}

anidridos_acido_data = {
    'anidrido acético':        ('CC(=O)OC(=O)C',        ['C', 'O', 'H'],       [1, 2]),
    'anidrido maleico':        ('O=C\\C=C\\C(=O)OCC(=O)O', ['C', 'O', 'H'],    [1, 2]),
    'anidrido ftálico':        ('OC(=O)C1=CC=CC=C1C(=O)O', ['C', 'O', 'H'],    [1, 2]),
    'anidrido benzoico':       ('c1ccccc1C(=O)OC(=O)c2ccccc2', ['C', 'O', 'H'], [1, 2]),
}

aminas_data = {
    'metilamina':           ('CN',            ['C', 'N', 'H'],       [1]),
    'dimetilamina':         ('CN(C)C',        ['C', 'N', 'H'],       [1]),
    'trimetilamina':        ('N(C)(C)C',      ['C', 'N', 'H'],       [1]),
    'anilina':              ('c1ccccc1N',     ['C', 'N', 'H'],       [1, 2]),
    'etilamina':            ('CCN',           ['C', 'N', 'H'],       [1]),
}

amidas_data = {
    'formamida':            ('C(=O)N',        ['C', 'O', 'N', 'H'],  [1, 2]),
    'acetamida':            ('CC(=O)N',       ['C', 'O', 'N', 'H'],  [1, 2]),
    'benzamida':            ('c1ccccc1C(=O)N',['C', 'O', 'N', 'H'],  [1, 2]),
}

nitrilas_data = {
    'acetonitrila':         ('CC#N',          ['C', 'N', 'H'],       [1, 3]),
    'benzonitrila':         ('c1ccccc1C#N',   ['C', 'N', 'H'],       [1, 3]),
    'propionitrila':        ('CCC#N',         ['C', 'N', 'H'],       [1, 3]),
}

nitrocompostos_data = {
    'nitrobenzeno':         ('c1ccccc1[N+](=O)[O-]', ['C', 'N', 'O', 'H'], [1, 2]),
    'nitrometano':          ('C[N+](=O)[O-]',       ['C', 'N', 'O', 'H'], [1, 2]),
}


tiol_data = {
    'metanotiol':    ('CS',           ['C', 'S', 'H'],      [1]),
    'etanotiol':     ('CCS',          ['C', 'S', 'H'],      [1]),
    'benzenotiol':   ('c1ccccc1S',    ['C', 'S', 'H'],      [1, 2]),
}


sulfeto_data = {
    'dimetilsulfeto': ('CSC',         ['C', 'S', 'H'],      [1]),
    'dietilsulfeto':  ('CCSCC',       ['C', 'S', 'H'],      [1]),
    'tiofeno':        ('c1ccsc1',     ['C', 'S', 'H'],      [1, 2]),
}


acido_sulfonico_data = {
    'ácido benzenossulfônico': ('c1ccc(cc1)S(=O)(=O)O', ['C', 'S', 'O', 'H'], [1, 2]),
    'ácido metanossulfônico':  ('CS(=O)(=O)O',          ['C', 'S', 'O', 'H'], [1, 2]),
}

# Haletos (Cl⁻, Br⁻, I⁻, F⁻)
haletos_data = {
    'cloreto de sódio':       ('[Na+].[Cl-]',        ['Na', 'Cl'],           [1]),
    'cloreto de metila':      ('CCl',                ['C', 'Cl', 'H'],       [1]),
    'brometo de etila':       ('CCBr',               ['C', 'Br', 'H'],       [1]),
    'iodeto de metila':       ('CI',                 ['C', 'I', 'H'],        [1]),
    'fluoreto de hidrogênio': ('[H][F]',                 ['H', 'F'],             [1]),
}

# Óxidos (metal + O ou não-metal + O)
oxidos_data = {
    'dióxido de carbono':     ('O=C=O',              ['C', 'O'],             [2]),
    'óxido de sódio':         ('[Na+] [O-]',          ['Na', 'O'],            [1]),
    'óxido de ferro (III)':   ('[Fe+3].[O-].[O-]',   ['Fe', 'O'],             [1]),
    'monóxido de carbono':    ('[C-]#[O+]',          ['C', 'O'],             [3]),
    'óxido de enxofre (IV)':  ('O=S=O',              ['S', 'O'],             [2]),
}

# Hidretos (H⁻ ligado a metal ou não-metal)
hidretos_data = {
    'hidreto de lítio':       ('[LiH]',              ['Li', 'H'],            [1]),
    'amônia':                 ('N',                  ['N', 'H'],             [1]),  # NH3 SMILES simplificado
    'hidreto de sódio':       ('[NaH]',              ['Na', 'H'],            [1]),
    'água':                  ('O',                   ['O', 'H'],             [1]),  # H2O simplificado
}

# Ácidos inorgânicos
acidos_inorganicos_data = {
    'ácido clorídrico':       ('[H+].[Cl-]',         ['H', 'Cl'],            [1]),
    'ácido sulfúrico':        ('OS(=O)(=O)O',        ['H', 'S', 'O'],        [1, 2]),
    'ácido nítrico':          ('O=N(=O)O',           ['H', 'N', 'O'],        [1, 2]),
    'ácido fosfórico':        ('OP(=O)(O)O',         ['H', 'P', 'O'],        [1, 2]),
}

# Bases
bases_data = {
    'hidróxido de sódio':     ('[Na+].[OH-]',        ['Na', 'O', 'H'],       [1]),
    'amônia':                 ('N',                   ['N', 'H'],             [1]),
    'hidróxido de potássio': ('[K+].[OH-]',         ['K', 'O', 'H'],        [1]),
}

# Sais
sais_data = {
    'cloreto de amônio':      ('[NH4+].[Cl-]',       ['N', 'H', 'Cl'],       [1]),
    'sulfato de sódio':       ('[Na+].[Na+].[O-]S(=O)(=O)[O-]', ['Na', 'S', 'O'], [1, 2]),
    'nitrato de potássio':    ('[K+].[O-]N(=O)=O',   ['K', 'N', 'O'],        [1, 2]),
}


lista_dicts_nomes = [
    'alcohols_data',
    'fenois_data',
    'eteres_data',
    'aldeidos_data',
    'cetonas_data',
    'acidos_carboxilicos_data',
    'esteres_data',
    'anidridos_acido_data',
    'aminas_data',
    'amidas_data',
    'nitrilas_data',
    'nitrocompostos_data',
    'tiol_data',
    'sulfeto_data',
    'acido_sulfonico_data',
    'haletos_data',
    'oxidos_data',
    'hidretos_data',
    'acidos_inorganicos_data',
    'bases_data',
    'sais_data',
    'alcanos_data'
]





#%%
def adicionar_nome_grupo(dados: dict, nome_grupo: str) -> dict:
    """
    Adiciona ao final da tupla de cada composto uma string com o nome do grupo.

    Parâmetros:
    - dados: dict - dicionário com dados dos compostos (nome -> tupla)
    - nome_grupo: str - nome do grupo (ex: 'tiol' se o dicionário for 'tiol_data')

    Retorna:
    - dict - novo dicionário com a string adicionada ao final da tupla
    """
    novo_dict = {}
    for composto, tupla in dados.items():
        novo_dict[composto] = tupla + (nome_grupo,)
    return novo_dict



#%%
def concat_group_data(*dicts):
    """
    Concatena vários dicionários em um único.

    Parâmetros:
    - dicts: múltiplos dicionários com mesmo formato (nome:str -> (smiles:str, elementos:list, ordens:list))

    Retorna:
    - dict: dicionário concatenado contendo todos os pares chave-valor dos dicionários fornecidos.
    """
    combined = {}
    for d in dicts:
        # Opcional: checar colisões e avisar
        overlap = set(combined.keys()) & set(d.keys())
        if overlap:
            print(f"Aviso: chaves duplicadas encontradas e sobrescritas: {overlap}")
        combined.update(d)
    return combined
#%%
def processar_dados_funcionais(lista_dicts_nomes):
    """
    Processa uma lista de nomes de variáveis de dicionários de grupos funcionais,
    adiciona o nome do grupo a cada entrada e concatena todos em um único dicionário.

    Retorna:
    - group_data: dicionário combinado com todos os grupos
    """
    dados_com_nome_grupo = {}

    for nome_var in lista_dicts_nomes:
        dados_originais = globals().get(nome_var)

        if not dados_originais:
            print(f"[AVISO] Variável '{nome_var}' não encontrada no escopo global.")
            continue

        nome_grupo = nome_var.removesuffix('_data')  # Requer Python 3.9+, senão use .replace('_data', '')
        dados_com_nome_grupo[nome_grupo] = adicionar_nome_grupo(dados_originais, nome_grupo)

    group_data = concat_group_data(*dados_com_nome_grupo.values())

    return group_data

#%%
def adicionar_peso_molecular(df, coluna_smiles,column_list ,nova_coluna='peso_molecular'):
    """
    Adiciona uma nova coluna com o peso molecular baseado nos SMILES fornecidos.

    Parâmetros:
    - df: pandas.DataFrame - DataFrame contendo a coluna com SMILES
    - coluna_smiles: str - nome da coluna que contém os SMILES
    - nova_coluna: str - nome da nova coluna a ser criada (padrão: 'peso_molecular')

    Retorna:
    - DataFrame com a nova coluna
    """
    
    def calcular_peso(smiles):
        try:
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                return None
            return Descriptors.MolWt(mol)
        except:
            return None

    df[nova_coluna] = df[coluna_smiles].apply(calcular_peso)
    
    column_list.append(nova_coluna)
    return df,column_list

#%%
def adicionar_logp(df, coluna_smiles, column_list, nova_coluna='logP'):
    """
    Adiciona uma nova coluna com o valor de LogP baseado nos SMILES fornecidos.

    Parâmetros:
    - df: pandas.DataFrame - DataFrame contendo a coluna com SMILES
    - coluna_smiles: str - nome da coluna que contém os SMILES
    - nova_coluna: str - nome da nova coluna a ser criada (padrão: 'logP')

    Retorna:
    - DataFrame com a nova coluna
    """

    def calcular_logp(smiles):
        try:
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                return None
            return Descriptors.MolLogP(mol)
        except:
            return None

    df[nova_coluna] = df[coluna_smiles].apply(calcular_logp)
    column_list.append(nova_coluna)
    return df,column_list

#%%
def adicionar_n_atomospesados(df, coluna_smiles, column_list, nova_coluna='n_atomospesados'):
    """
    Adiciona uma nova coluna com o número de átomos pesados da molécula (excluindo hidrogênios).

    Parâmetros:
    - df: pandas.DataFrame - DataFrame com a coluna de SMILES
    - coluna_smiles: str - nome da coluna com SMILES
    - nova_coluna: str - nome da nova coluna (padrão: 'n_atomospesados')

    Retorna:
    - DataFrame com a nova coluna
    """

    def calcular_n_pesados(smiles):
        try:
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                return None
            return Descriptors.HeavyAtomCount(mol)
        except:
            return None

    df[nova_coluna] = df[coluna_smiles].apply(calcular_n_pesados)
    column_list.append(nova_coluna)
    return df,column_list


#%%
def adicionar_hdoadores(df, coluna_smiles,column_list, nova_coluna='n_hdoadores'):
    def calcular_hbd(smiles):
        try:
            mol = Chem.MolFromSmiles(smiles)
            return Descriptors.NumHDonors(mol) if mol else None
        except:
            return None

    df[nova_coluna] = df[coluna_smiles].apply(calcular_hbd)
    column_list.append(nova_coluna)
    return df,column_list
#%%

def adicionar_haceptores(df, coluna_smiles, column_list, nova_coluna='n_haceptores'):
    def calcular_hba(smiles):
        try:
            mol = Chem.MolFromSmiles(smiles)
            return Descriptors.NumHAcceptors(mol) if mol else None
        except:
            return None

    df[nova_coluna] = df[coluna_smiles].apply(calcular_hba)
    column_list.append(nova_coluna)
    return df,column_list


def adicionar_tpsa(df, coluna_smiles, column_list,nova_coluna='tpsa'):
    def calcular_tpsa(smiles):
        try:
            mol = Chem.MolFromSmiles(smiles)
            return Descriptors.TPSA(mol) if mol else None
        except:
            return None

    df[nova_coluna] = df[coluna_smiles].apply(calcular_tpsa)
    column_list.append(nova_coluna)
    return df,column_list

#%%
def adicionar_rotatable_bonds(df, coluna_smiles,column_list, nova_coluna='n_rotatable_bonds'):
    def calcular_rotb(smiles):
        try:
            mol = Chem.MolFromSmiles(smiles)
            return Descriptors.NumRotatableBonds(mol) if mol else None
        except:
            return None

    df[nova_coluna] = df[coluna_smiles].apply(calcular_rotb)
    column_list.append(nova_coluna)
    return df,column_list



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
    df_train = load_input(relative_path=path_interim , filename=train_name)
    df_test = load_input(relative_path=path_interim , filename=test_name)
    #df_train = df_train.head(1)
        
    group_data = processar_dados_funcionais(lista_dicts_nomes)    
    df_train,column_list = generate_components_count_categoria(df_train,smile_name, group_data )
    df_train,column_list = generate_components_count(df_train,smile_name, group_data )
    df_train,column_list = adicionar_peso_molecular(df_train,smile_name,column_list)
    df_train,column_list = adicionar_logp(df_train,smile_name,column_list)
    df_train,column_list = adicionar_n_atomospesados(df_train,smile_name,column_list)
    
    df_train,column_list = adicionar_hdoadores(df_train,smile_name,column_list)
    df_train,column_list = adicionar_haceptores(df_train,smile_name,column_list)
    df_train,column_list = adicionar_tpsa(df_train,smile_name,column_list)
    df_train,column_list = adicionar_rotatable_bonds(df_train,smile_name,column_list)
    
    
    exportar_lista_pickle(column_list,pickle_path,pickle_name)
    
    save_csv(df_train,path_processed, train_name)
    save_csv(df_test,path_processed, test_name)
    return df_train

     

#%%
if __name__ == '__main__':


# Oculta todos os DeprecationWarnings
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    df = main()


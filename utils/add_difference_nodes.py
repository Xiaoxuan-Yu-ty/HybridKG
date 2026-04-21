
import re
from matplotlib import pyplot as plt
from matplotlib_venn import venn3, venn2
import matplotlib.patches as mpatches
import networkx as nx
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pickle
from collections import defaultdict
import seaborn as sns


def get_name_species(text:str, pattern:str = r'p\(UniProtKB:"(\w+)_(\w+)"\)$'):
    match = re.search(pattern, text)
    if match:
        #print(match.group())
        name = match.group(1)
        species = match.group(2)
        return name, species
    else:
        #print('No match found.')
        return "", ""

def get_hgnc_name(text, pattern:str=r'(HGNC:"(\w+)")\)'):
    match = re.search(pattern, text)
    if match:
        name = match.group(2)
        #print(name)
        return name
    else:
        return None

def get_proteins_in_healthykg(kg):
    protein_names = []
    for node, attrs in kg.nodes(data=True):
        label = attrs['type']
        namespace = attrs.get('namespace', None)
        #print(attrs)
        if namespace == 'UniProtKB' and label == 'Protein':
            
            pr, _ = get_name_species(node)
            if pr != "":
                protein_names.append(pr)
    print(f'Number of protein nodes: {len(protein_names)}')
    print(f"Number of unique proteins: {len(set(protein_names))}")
    return protein_names

def get_proteins_in_adkg(kg):
    ad_prs = []
    for node, attrs in kg.nodes(data=True):
        if node.startswith('p('):
            pr_name = get_hgnc_name(node)
            if pr_name is not None:
                ad_prs.append(pr_name)
    print(f'Number of protein nodes: {len(ad_prs)}')
    print(f"Number of unique proteins: {len(set(ad_prs))}")
    return ad_prs


def plot_venn2(set1, set2, label1, label2, title='Proteins Overlapping Comparison'):
    plt.figure(figsize=(10,6))
    venn = venn2((set(set1), set(set2)), set_labels=(label1, label2))
    #plt.legend(handles=[red_patch, blue_patch, green_patch], loc='lower right', title="Legend")
    plt.title(title)
    plt.show()

def load_graph(kg_path):
    with open(kg_path, 'rb') as f:
        kg = pickle.load(f)
    print(f"Graph has {kg.number_of_nodes()} nodes and {kg.number_of_edges()} edges.")
    return kg

def save_graph(G, filename):
    with open(filename, "wb") as f:
        pickle.dump(G, f)
    print(f"Saved graph to {filename}: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")

                
def add_isolated_proteins_to_kg(ad_kg_path, health_kg_path):
    ad_kg = load_graph(ad_kg_path)
    health_kg = load_graph(health_kg_path)

    ad_proteins = get_proteins_in_adkg(ad_kg)
    health_proteins = get_proteins_in_healthykg(health_kg)

    # get difference proteins
    pr_not_in_healthy = set(ad_proteins) - set(health_proteins)
    pr_not_in_ad = set(health_proteins) - set(ad_proteins)

    print(f"{len(pr_not_in_healthy)} proteins only exist in AD KG")
    print(f"{len(pr_not_in_ad)} proteins only exist in Healthy KG")
    print(f"{len(set(ad_proteins) & set(health_proteins))} proteins exist in both KGs")

    # add proteins from sets
    ad_kg.add_nodes_from((node, {'type':'Protein', 'namespace':'UniProtKB','source':"Healthy-KG"}) for node in pr_not_in_ad)
    health_kg.add_nodes_from((node, {'type': 'Protein', 'namespace':'HGNC','source':'AD-KG'}) for node in pr_not_in_healthy)

    # save graph
    save_graph(ad_kg, ad_kg_path)
    save_graph(health_kg, health_kg_path)

    return ad_kg, health_kg

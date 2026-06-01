# -*- coding: utf-8 -*-

"""Ensemble of methods for network generation."""
import argparse
from itertools import combinations
from os import listdir
import os
from os.path import isfile, join
import pickle
from typing import TextIO, Optional, Tuple, Union, Set, List
import logging

import networkx as nx
import pandas as pd
from tqdm import tqdm

logger = logging.getLogger(__name__)

VALUE_TO_COLNAME = {
    -1: 'negative_relation',
    1: 'positive_relation'
}

def do_graph_gen(
        data: pd.DataFrame,
        network_gen_method: str = 'interaction_network',
        gmt: Optional[str] = None,
        intersection_threshold: float = 0.1,
        kg_data: Optional[pd.DataFrame] = None,
        folder_path: Optional[str] = None,
        jaccard_threshold: float = 0.2,
        summary: bool = False,
) -> Tuple[nx.Graph, Optional[pd.DataFrame], Optional[Set[str]]]:
    """Generate patient-feature network given the data using a certain network generation method.

    :param data: Dataframe containing the patient-feature scores
    :param network_gen_method: Method to generate the patient-feature network
    :param gmt: Optional field for the path to the gmt file containing the pathway data
    :param intersection_threshold: Threshold to make edges in Pathway Overlap method
    :param kg_data: Optional field for the knowledge graph in edgelist format stored in a pandas dataframe
    :param folder_path: Optional field for the path to a folder containing multiple knowledge graphs
    :param jaccard_threshold: Threshold to make edges in Interaction Network Overlap method
    :param summary: Flag to indicate if the summary of the patient-feature network must be returned
    :return: Dataframe containing patient-feature network, and optionally the summary of the patient-feature network
    """
    information_graph = nx.DiGraph()

    if network_gen_method == 'interaction_network' and kg_data is not None:
        interaction_graph = nx.from_pandas_edgelist(
            df=kg_data,
            source=kg_data.columns[0],
            target=kg_data.columns[2],
            edge_attr=kg_data.columns[1]
        )

        if nx.number_connected_components(interaction_graph) > 1:
            logger.warning(f'The number of connected components in the graph is greater than 1. '
                           f'There are {nx.number_connected_components(interaction_graph)} connected components of size'
                           f', {[len(c) for c in sorted(nx.connected_components(interaction_graph), key=len, reverse=True)]}'
                           f' respectively.')

        information_graph = plot_interaction_network(kg_data)

   
    if summary:
        final_graph, summary_data, linked_genes = overlay_samples(data, information_graph, summary=True)
    else:
        final_graph, _, _ = overlay_samples(data, information_graph, summary=False)

    # graph_df: pd.DataFrame = nx.to_pandas_edgelist(final_graph)

    # graph_df['relation'] = graph_df['relation'].fillna('no_change', inplace=True)

    # graph_df = graph_df[['source', 'target', 'relation', 'label']]

    if summary:
        return final_graph, summary_data, linked_genes
    else:
        return final_graph, None, None


def plot_interaction_network(
        kg_data: pd.DataFrame
) -> nx.DiGraph:
    """Plot a knowledge graph based on the interaction data."""
    interaction_graph = nx.DiGraph()

    # Append the source to target mapping to the main data edgelist
    for idx in tqdm(kg_data.index, desc='Plotting interaction network: '):
        interaction_graph.add_edge(
            str(kg_data.iat[idx, 0]),
            str(kg_data.iat[idx, 2]),
            relation=str(kg_data.iat[idx, 1])
        )

    return interaction_graph


def overlay_samples(
        data: pd.DataFrame,
        information_graph: nx.DiGraph,
        summary: bool = False,
) -> Tuple[nx.Graph, Optional[pd.DataFrame], Optional[Set[str]]]:
    """Overlay the data onto the information graph by adding edges between patients and information nodes."""
    patient_label_mapping = {patient: label for patient, label in zip(data.index, data['label'])}
    value_mapping = {0: 'no_change', 1: 'up_reg', -1: 'down_reg'}

    overlay_graph = information_graph.copy()

    data_copy = data.drop(columns='label')
    values_data = data_copy.values

    summary_data = pd.DataFrame(0, index=data_copy.index, columns=["positive_relation", "negative_relation", 'linked_genes'])
    summary_data['linked_nodes'] = [[] for _ in range(len(summary_data))] # Initialize with empty lists
    summary_data['label'] = data['label'].to_list()
    
    linked_genes = set()
    edges_to_remove = []

    for index, value_list in enumerate(tqdm(values_data, desc='Adding patients to the network: ')):
        for column, value in enumerate(value_list):
            patient = data_copy.index[index]
            gene = data_copy.columns[column]

            # Avoid mangled duplicates from pandas
            if "." in gene:
                if gene.split(".")[0] in data_copy.columns:
                    gene = gene.split(".")[0]

            # Ignore features with score of 0
            if value == 0:
                continue

            # Skip if gene is not in the knowledge graph
            if gene+'_HUMAN' in information_graph.nodes:
                if overlay_graph.has_edge(patient, gene):
                    if overlay_graph.get_edge_data(patient, gene)['relation'] != value_mapping[value]:
                        if (patient, gene) not in edges_to_remove:
                            edges_to_remove.append((patient, gene))
                    continue
                linked_genes.add(gene)
                overlay_graph.add_edge(patient, gene, relation=value_mapping[value],
                                       label=patient_label_mapping[patient])
            #if summary:
                summary_data.at[patient, VALUE_TO_COLNAME[value]] += 1
                summary_data.at[patient, 'linked_nodes'].append(gene)
            
    # Remove patient-gene triples that have conflicting duplicates in the data
    for patient, gene in edges_to_remove:
        logger.warning(f"{patient}-{gene} triple is being discarded due to conflicting data")
        overlay_graph.remove_edge(patient, gene)

    if summary:
        non_conn_pats = summary_data[(summary_data['positive_relation'] == 0) & (summary_data['negative_relation'] == 0)]

        if len(non_conn_pats) > 0:
            logger.warning(f'{len(non_conn_pats)} samples is/are not connected to any genes.')

        return overlay_graph, summary_data, linked_genes

    return overlay_graph, None, None

def main(): 
    parser = argparse.ArgumentParser(description="Generate Hybrid Patient-Protein Networks.")

    # Stable Arguments
    parser.add_argument("--kg_path", type=str, default="../../datasets/base_kgs/ppi_hc.pkl", 
                        help="Path to Disease Knowledge Graph (.pkl).")
    parser.add_argument("--output_dir", type=str, default="../../CLEP_replication/networks/PPI_KGs_clep", 
                        help="Directory to save generated networks.")

    # Arguments need to change
    parser.add_argument("--dataset", type=str, default="adni", 
                        help="Name of the dataset (for naming files).")

    parser.add_argument("--scoring_path", type=str, default="../../data/ADNI/old_target/ecdf_1/sample_scoring_ecdf.csv", 
                        help="Path to sample scoring CSV (must contain 'label' column).")
    parser.add_argument("--scoring_type", type=str, default="ecdf", choices=['ecdf','std','all'],
                        help="The scoring method used (for naming files).")
    
    args = parser.parse_args()

    # 1. load kg 
    with open(args.kg_path, 'rb') as f:
        kg = pickle.load(f)
    kg_data: pd.DataFrame = nx.to_pandas_edgelist(kg)

    for threshold in [1]:#,1.5,2.5,5,10,20]:
        scoring_path = f"../../data/ADNI/old_target/ecdf_{threshold}/sample_scoring_ecdf.csv"
        # 2. load sample scoring df
        data = pd.read_csv(scoring_path, index_col=0)

        # 3.generate network

        graph, summary_df, linked_genes = do_graph_gen(data=data,
                    network_gen_method='interaction_network',
                    kg_data=kg_data,
                    summary=True)
        
        # 4. save
        # prepare save path
        os.makedirs(args.output_dir, exist_ok=True)
        save_network = os.path.join(args.output_dir, f"G_{args.dataset}_OldTarget_DiseaseKG_{args.scoring_type}_{threshold}.pkl")
        save_summary = os.path.join(args.output_dir, f"Summary_{args.dataset}_OldTarget_DiseaseKG_{args.scoring_type}_{threshold}.csv")
        
        with open(save_network, 'wb') as f:
            pickle.dump(graph, f)
        if summary_df is not None:
            summary_df.to_csv(save_summary)
    

if __name__ == "__main__":
    main()
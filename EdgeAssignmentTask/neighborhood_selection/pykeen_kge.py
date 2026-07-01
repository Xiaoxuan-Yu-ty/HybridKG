
"""Embed patients with the biomedical entities (genes and metabolites) using Knowledge graph embedding."""
import json
import os
import pickle
from typing import Optional, Tuple, Dict, Any
import argparse
import sys

import numpy as np
import numpy.typing as npt
import pandas as pd
import networkx as nx
from pykeen.hpo.hpo import hpo_pipeline_from_config
from pykeen.models.nbase import ERModel
from pykeen.typing import HeadRepresentation, RelationRepresentation, TailRepresentation
from pykeen.pipeline import pipeline_from_config, pipeline_from_path, PipelineResult
from pykeen.triples import TriplesFactory

def do_kge(
        edgelist: pd.DataFrame,
        out: str,
        model_config: Dict[str, Any],
        return_patients: bool = True,
        train_size: float = 0.8,
        validation_size: float = 0.1,
        complex_embedding: bool = False
) -> pd.DataFrame:
    """Carry out KGE on the given data.

    :param edgelist: Dataframe containing the patient-feature graph in edgelist format
    :param design: Dataframe containing the design table for the data
    :param out: Output folder for the results
    :param model_config: Configuration file for the KGE models, in JSON format.
    :param return_patients: Flag to indicate if the final data should contain only patients or even the features
    :param train_size: Size of the training data for KGE ranging from 0 - 1
    :param validation_size: Size of the validation data for KGE ranging from 0 - 1. It must be lower than training size
    :param complex_embedding: Flag to indicate if only the real part of the embedding should be returned.
    :return: Dataframe containing the embedding from the KGE
    """

    # Split the edgelist into training, validation and testing data
    train, validation, test = _weighted_splitter(
        edgelist=edgelist,
        train_size=train_size,
        validation_size=validation_size
    )

    train_path = os.path.join(out, 'train.edgelist')
    validation_path = os.path.join(out, 'validation.edgelist')
    test_path = os.path.join(out, 'test.edgelist')

    train.to_csv(train_path, sep='\t', index=False, header=False)
    validation.to_csv(validation_path, sep='\t', index=False, header=False)
    test.to_csv(test_path, sep='\t', index=False, header=False)
    
    train_triples_factory = TriplesFactory.from_path(train_path, create_inverse_triples=True)
    validation_triples_factory = TriplesFactory.from_path(validation_path, create_inverse_triples=True)
    test_triples_factory = TriplesFactory.from_path(test_path, create_inverse_triples=True)

    # HPO
    print("\n--------------Run KGE HPO------------------------------------")
    run_optimization(
        dataset=(train_triples_factory, validation_triples_factory, test_triples_factory),
        model_config=model_config,
        out_dir=out
    )
    # Retrain with best HP
    print("\n--------------Retrain KGE with Best HP------------------------------------")
    pipeline_results = run_pipeline(train_triples_factory, test_triples_factory, validation_triples_factory,
        out_dir=out
    )

    best_model, triple_factory = pipeline_results.model, pipeline_results.training

    # Get the embedding as a numpy array. Ignore the type as the model will be of type ERModel (Embedding model)
    embedding_values = _model_to_numpy(best_model, complex=complex_embedding)  # type: ignore

    # Create columns as component names
    embedding_columns = [f'Component_{i}' for i in range(1, embedding_values.shape[1] + 1)]

    # Get the nodes of the training triples as index
    node_list = list(triple_factory.entity_to_id.keys())
    embedding_index = sorted(node_list, key=lambda x: triple_factory.entity_to_id[x])

    embedding = pd.DataFrame(data=embedding_values, columns=embedding_columns, index=embedding_index)

    return embedding


def _weighted_splitter(
        edgelist: pd.DataFrame,
        train_size: float = 0.8,
        validation_size: float = 0.1
) -> Tuple[pd.DataFrame, ...]:
    """Split the given edgelist into training, validation and testing sets on the basis of the ratio of relations.

    :param edgelist: Edgelist in the form of (Source, Relation, Target)
    :param train_size: Size of the training data
    :param validation_size: Size of the training data
    :return: Tuple containing the train, validation & test splits
    """
    # Validation size is the size of the percentage of the remaining data (i.e. If required validation size is 10% of
    # the original data & training size is 80% then the new validation size is 50% of the data without the training
    # data. The similar calculation is done for training size, hence it is always 1
    validation_size = validation_size / (1 - train_size)
    test_size = 1

    # Get the unique relations in the network
    unique_relations = sorted(edgelist['relation'].unique())

    data = edgelist.drop_duplicates().copy()

    split = []
    # Split the data to get training, validation and test samples
    for frac_size in [train_size, validation_size, test_size]:
        frames = []
        # Random sampling of the data for every type of relation
        for relation in unique_relations:
            temp = data[data['relation'] == relation].sample(frac=frac_size)

            data = data[~data.index.isin(temp.index)]

            frames.append(temp)
        # Join all the different relations in one dataframe
        split.append(pd.concat(frames, ignore_index=True, sort=False))

    return tuple(split)


def _model_to_numpy(
        model: ERModel[HeadRepresentation, RelationRepresentation, TailRepresentation],
        complex: bool = False
) -> npt.NDArray[np.float64 | np.float32]:
    """Retrieve embedding from the models as a numpy array."""
    embedding_numpy: npt.NDArray[np.float64 | np.float32] = model.entity_representations[0](indices=None).detach().cpu().numpy()

    if complex:
        return embedding_numpy

    # Get the real part of the embedding for classification tasks
    return embedding_numpy.real


def run_optimization(dataset: Tuple[TriplesFactory, TriplesFactory, TriplesFactory], model_config: Dict[str, Any], out_dir: str) -> None:
    """Run HPO."""
    train_triples_factory, validation_triples_factory, test_triples_factory = dataset

    # Define HPO pipeline
    hpo_results = hpo_pipeline_from_config(
        dataset=None,
        training=train_triples_factory,
        testing=test_triples_factory,
        validation=validation_triples_factory,
        config=model_config
    )

    optimization_dir = os.path.join(out_dir, 'pykeen_results_optim')
    if not os.path.isdir(optimization_dir):
        os.makedirs(optimization_dir)

    hpo_results.save_to_directory(optimization_dir)


def run_pipeline(train_tf,
                 test_tf,
                 validation_tf,
                 out_dir: str,
                 best_config_dict: Optional[Dict[str, Any]] = None
                 ) -> PipelineResult:
    """Run Pipeline."""

    if best_config_dict is None:
        config_path = os.path.join(out_dir, 'pykeen_results_optim', 'best_pipeline', 'pipeline_config.json')
        with open(config_path, 'r') as f:
            best_config_dict = json.load(f)

    assert best_config_dict is not None

    # Remove ALL structural keys causing duplicates or path errors
    if "pipeline" in best_config_dict:
        inner_pipeline = best_config_dict["pipeline"]
        inner_pipeline.pop("dataset", None)
        inner_pipeline.pop("training", None)
        inner_pipeline.pop("testing", None)
        inner_pipeline.pop("validation", None)
        
        # REMOVE outdated model_kwargs
        if "model_kwargs" in inner_pipeline:
            inner_pipeline["model_kwargs"].pop("automatic_memory_optimization", None)

    # Execute training safely using your in-memory objects
    pipeline_results = pipeline_from_config(
        config=best_config_dict,
        training=train_tf,
        testing=test_tf,
        validation=validation_tf,
        #automatic_memory_optimization=True
    )

    best_pipeline_dir = os.path.join(out_dir, 'pykeen_results_final')
    if not os.path.isdir(best_pipeline_dir):
        os.makedirs(best_pipeline_dir)

    pipeline_results.save_to_directory(best_pipeline_dir, save_replicates=True)

    print(f"Retraining completed successfully! Results save to {best_pipeline_dir}")
    return pipeline_results

def load_graph(filepath)->nx.MultiDiGraph:
    with open(filepath, 'rb') as f:
        G = pickle.load(f)
    print(f"Load graph with {G.number_of_nodes()} nodes and {G.number_of_edges()} edges")
    return G

def save_graph(G, save_path):
    with open(save_path, 'wb') as f:
        pickle.dump(G, f)
    print(f"Save graph to {save_path}: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")

def parser():
    parser = argparse.ArgumentParser(description="Generate Hybrid Patient-Protein Networks.")

    # graph generator args
    parser.add_argument("--DiseaseKG", type=str, default='AD_KG', choices=['PPI_KG','Prime_KG','AD_KG'])
    parser.add_argument("--kg_disease", type=str, default="./datasets/base_kgs/ad_kg_with_reverse_edges.pkl", 
                        help="Path to Disease Knowledge Graph (.pkl).")
    parser.add_argument("--kg_healthy", type=str, default="./data/KG/healthy_aging_reversed_remove_noncausal.pkl", 
                        help="Path to Healthy Knowledge Graph (.pkl).")
    parser.add_argument("--output_dir", type=str, default="./EdgeAssignmentTask/neighborhood_selection/results/pykeen_kge", 
                        help="Directory to save generated networks.")

    # KGE arguments
    parser.add_argument("--kge_model", type=str, default="RotatE")
    
    args = parser.parse_args()
    return args

def main():
    args = parser()
    final_output = os.path.join(args.output_dir, args.DiseaseKG)
    os.makedirs(final_output, exist_ok=True)

    # 1. load KGs
    kg_disease = load_graph(args.kg_disease)
    kg_healthy = load_graph(args.kg_healthy)
    kg = nx.compose(kg_disease, kg_healthy)
    save_kg_path = os.path.join(final_output,f"G_merge_{args.DiseaseKG}_healthy.pkl")
    save_graph(kg, save_path=save_kg_path)

    # 2. convert networkx graph to edgelist
    graph_df = nx.to_pandas_edgelist(kg)
    graph_df = graph_df[~graph_df['relation'].str.contains('rev', na=False)]
    graph_df=graph_df[['source','relation','target']]
    save_edgelist_path = os.path.join(final_output,f"Edgelist_merge_{args.DiseaseKG}_healthy.csv")
    graph_df.to_csv(save_edgelist_path)
    
    # 3. KGE: HPO + Retrain 
    print("\n-------------Do KGE---------------------------------------------")
    hpo_config_path = f"./PyKeen/configs/{args.kge_model}_model_config.json"
    with open(hpo_config_path, 'r') as f:
        hpo_config_dict = json.load(f)

    assert hpo_config_dict is not None

    # Remove ALL structural keys causing duplicates or path errors
    if "pipeline" in hpo_config_dict:
        inner_pipeline = hpo_config_dict["pipeline"]
        inner_pipeline.pop("dataset", None)
        inner_pipeline.pop("training", None)
        inner_pipeline.pop("testing", None)
        inner_pipeline.pop("validation", None)
        
        # REMOVE outdated model_kwargs
        if "model_kwargs" in inner_pipeline:
            inner_pipeline["model_kwargs"].pop("automatic_memory_optimization", None)

    embeddings = do_kge(edgelist=graph_df,
                        out=final_output,
                        model_config=hpo_config_dict,
                        return_patients=True,
                        train_size=0.8, validation_size=0.1,
                        complex_embedding=False)
    
    embeddings.to_csv(os.path.join(final_output,'embedding.csv')) 

if __name__=="__main__":
    main()

"""Embed patients with the biomedical entities (genes and metabolites) using Knowledge graph embedding."""
import json
import os
import pickle
from typing import List, Optional, Tuple, Dict, Any
import argparse
import warnings

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
        train_size: float = 0.8,
        validation_size: float = 0.1,
        complex_embedding: bool = False,
        ensure_full_node_coverage: bool = True,
) -> pd.DataFrame:
    """Carry out KGE on the given data.

    :param edgelist: Dataframe containing the patient-feature graph in edgelist format
    :param out: Output folder for the results
    :param model_config: Configuration file for the KGE models, in JSON format.
    :param return_patients: Flag to indicate if the final data should contain only patients or even the features
    :param train_size: Size of the training data for KGE ranging from 0 - 1
    :param validation_size: Size of the validation data for KGE ranging from 0 - 1. It must be lower than training size
    :param complex_embedding: Flag to indicate if only the real part of the embedding should be returned.
    :param ensure_full_node_coverage: If True, guarantees every node has at least one edge placed in the
        training split so it receives a trained embedding.
    :return: Dataframe containing the embedding from the KGE, indexed by ALL nodes in `edgelist`.
    """

    # Build a GLOBAL entity/relation id mapping from the full edgelist up front.
    # This is the key fix: train/validation/test TriplesFactory objects will all
    # share this same id space, so the model allocates an embedding row for every
    # node in the original graph -- not just the ones that appear in train.edgelist.
    full_triples_factory = TriplesFactory.from_labeled_triples(
        edgelist[['source', 'relation', 'target']].astype(str).values
    )
    global_entity_to_id = full_triples_factory.entity_to_id
    global_relation_to_id = full_triples_factory.relation_to_id

    # Split the edgelist into training, validation and testing data
    train, validation, test = _weighted_splitter(
        edgelist=edgelist,
        train_size=train_size,
        validation_size=validation_size,
        ensure_full_node_coverage=ensure_full_node_coverage,
    )

    train_path = os.path.join(out, 'train.edgelist')
    validation_path = os.path.join(out, 'validation.edgelist')
    test_path = os.path.join(out, 'test.edgelist')

    train.to_csv(train_path, sep='\t', index=False, header=False)
    validation.to_csv(validation_path, sep='\t', index=False, header=False)
    test.to_csv(test_path, sep='\t', index=False, header=False)

    # Pass the GLOBAL id mapping so all three factories share the same entity/relation ids
    train_triples_factory = TriplesFactory.from_path(
        train_path,
        entity_to_id=global_entity_to_id,
        relation_to_id=global_relation_to_id,
        create_inverse_triples=True,
    )
    validation_triples_factory = TriplesFactory.from_path(
        validation_path,
        entity_to_id=global_entity_to_id,
        relation_to_id=global_relation_to_id,
        create_inverse_triples=True,
    )
    test_triples_factory = TriplesFactory.from_path(
        test_path,
        entity_to_id=global_entity_to_id,
        relation_to_id=global_relation_to_id,
        create_inverse_triples=True,
    )

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

    # Because train/validation/test all share global_entity_to_id, triple_factory.entity_to_id
    # now covers ALL nodes from the original edgelist -- this is the actual fix.
    node_list = list(triple_factory.entity_to_id.keys())
    embedding_index = sorted(node_list, key=lambda x: triple_factory.entity_to_id[x])

    embedding = pd.DataFrame(data=embedding_values, columns=embedding_columns, index=embedding_index)

    return embedding


def _weighted_splitter(
        edgelist: pd.DataFrame,
        train_size: float = 0.8,
        validation_size: float = 0.1,
        ensure_full_node_coverage: bool = True,
        random_state: Optional[int] = 42,
) -> Tuple[pd.DataFrame, ...]:
    """Split the given edgelist into training, validation and testing sets on the basis of the
    ratio of relations, while (by default) guaranteeing every node appears at least once in the
    training set so it actually receives a *trained* embedding.

    :param edgelist: Edgelist in the form of (source, relation, target)
    :param train_size: Size of the training data
    :param validation_size: Size of the validation data
    :param ensure_full_node_coverage: If True, forces one edge per node into train first
    :param random_state: Optional seed for reproducibility
    :return: Tuple containing the train, validation & test splits
    """
    test_size = 1 - train_size - validation_size
    if test_size < 0:
        raise ValueError("train_size + validation_size must not exceed 1")

    data = edgelist.drop_duplicates().reset_index(drop=True).copy()
    unique_relations = sorted(data['relation'].unique())

    forced_train_idx: set = set()

    if ensure_full_node_coverage:
        all_nodes = set(data['source']).union(data['target'])
        covered: set = set()
        # Shuffle so the edge "forced" into train for each node is randomized
        shuffled = data.sample(frac=1, random_state=random_state)

        for idx, row in shuffled.iterrows():
            if row['source'] not in covered or row['target'] not in covered:
                forced_train_idx.add(idx)
                covered.add(row['source'])
                covered.add(row['target'])
            if len(covered) == len(all_nodes):
                break

        missing = all_nodes - covered
        if missing:
            # Only possible if a node label appears in edgelist metadata but has literally no edges
            warnings.warn(f"{len(missing)} node(s) could not be covered by any edge: {missing}")

    # Remaining edges (not already forced into train) get split by relation as before
    remaining = data.drop(index=list(forced_train_idx))

    validation_frames = []
    test_frames = []

    for relation in unique_relations:
        subset = remaining[remaining['relation'] == relation]

        n_val = int(round(len(subset) * validation_size))
        val_sample = subset.sample(n=min(n_val, len(subset)), random_state=random_state)

        subset_after_val = subset.drop(index=val_sample.index)
        n_test = int(round(len(subset) * test_size))
        test_sample = subset_after_val.sample(n=min(n_test, len(subset_after_val)), random_state=random_state)

        validation_frames.append(val_sample)
        test_frames.append(test_sample)

    validation = (
        pd.concat(validation_frames, ignore_index=True, sort=False)
        if validation_frames else data.iloc[0:0].copy()
    )
    test = (
        pd.concat(test_frames, ignore_index=True, sort=False)
        if test_frames else data.iloc[0:0].copy()
    )

    excluded_idx = forced_train_idx | {
        idx for frame in (validation_frames + test_frames) for idx in frame.index
    }
    train = data.drop(index=list(excluded_idx)).reset_index(drop=True)

    return train, validation, test


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

def graph_to_edgelist(G) -> Tuple[pd.DataFrame, List[str]]:
    """Convert a MultiDiGraph to an edgelist dataframe, preserving isolated nodes separately
    since they cannot be represented as edge rows.
    """
    
    graph_df = nx.to_pandas_edgelist(G)
    graph_df = graph_df[~graph_df['relation'].str.contains('rev', na=False)]
    graph_df=graph_df[['source','relation','target']]

    all_nodes = list(G.nodes())  # includes isolates, unlike the edgelist
    return graph_df, all_nodes

def parser():
    parser = argparse.ArgumentParser(description="Generate Hybrid Patient-Protein Networks.")

    # graph generator args
    parser.add_argument("--DiseaseKG", type=str, nargs="+",default=['PPI_KG','Prime_KG','AD_KG'])
    parser.add_argument("--kg_disease", type=str, default="./datasets/base_kgs/ad_kg_with_reverse_edges.pkl", 
                        help="Path to Disease Knowledge Graph (.pkl).")
    parser.add_argument("--kg_healthy", type=str, default="./datasets/base_kgs/healthy_kg_with_reverse_edges.pkl", 
                        help="Path to Healthy Knowledge Graph (.pkl).")
    parser.add_argument("--output_dir", type=str, default="./EdgeAssignmentTask/neighborhood_selection/results/pykeen_kge_allnodes", 
                        help="Directory to save generated networks.")

    # KGE arguments
    parser.add_argument("--kge_model", type=str, default="RotatE")
    
    args = parser.parse_args()
    return args

def main():
    args = parser()
    
    for DiseaseKG in ['AD_KG', 'Prime_KG','PPI_KG']:
        if DiseaseKG == 'AD_KG':
            kg_disease_path = "./datasets/base_kgs/ad_kg_with_reverse_edges.pkl"
        elif DiseaseKG == 'Prime_KG':
            kg_disease_path = "./datasets/base_kgs/prime_kg_with_reverse_edges.pkl"
        elif DiseaseKG == 'PPI_KG':
            kg_disease_path = "./datasets/base_kgs/ppi_old_with_reverse_edges.pkl"
        else:
            print("Invalid DiseaseKG name")
            continue
        
        final_output = os.path.join(args.output_dir, DiseaseKG)
        os.makedirs(final_output, exist_ok=True)

        # 1. load KGs
        kg_disease = load_graph(kg_disease_path)
        kg_healthy = load_graph(args.kg_healthy)
        kg = nx.compose(kg_disease, kg_healthy)
        save_kg_path = os.path.join(final_output,f"G_merge_{DiseaseKG}_healthy.pkl")
        save_graph(kg, save_path=save_kg_path)

        # 2. convert networkx graph to edgelist
        graph_df, all_nodes = graph_to_edgelist(kg)
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
                            train_size=0.8, 
                            validation_size=0.1,
                            complex_embedding=False,
                            ensure_full_node_coverage=True)
        
        embeddings.to_csv(os.path.join(final_output,'embedding.csv')) 

if __name__=="__main__":
    main()
#!/usr/bin/env python3
"""
Build Disease-Sample-Healthy networks from Disease-KG, Healthy-KG, and Patient expression data for HeteroMultitaskGNN.
"""
import argparse
from os import listdir
import os
from os.path import isfile, join
import pickle
import sys
from typing import TextIO, Optional, Tuple, Union, Set, List
from sklearn.neighbors import NearestNeighbors
from sklearn.model_selection import train_test_split

import networkx as nx
import numpy as np
import pandas as pd
from tqdm import tqdm
import re
try:
    base_dir = os.path.dirname(os.path.abspath(__file__))
except NameError:
    base_dir = os.getcwd()
sys.path.append(os.path.dirname(base_dir))
from utils.graph_utils import load_graph, save_graph
from data_processing.sample_scoring import process_and_save

def main():
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp_path", "-e", default="", 
                        help="Expression data to generate network.")
    parser.add_argument("--dataset", default='adni',
                        help="Expression Dataset")
    parser.add_argument("--scoring_method", "-s", default='ecdf', 
                        choices=['ecdf', 'std','logFC'],
                        help="Method to use in sample scoring")
    parser.add_argument("--sample_scoring", action="store_true",
                        help="Generate radical search dataframes based on expression data.")
    
    parser.add_argument("--disease_kg", "-d", default="",
                        help="disease knowledge graph")
    parser.add_argument("--healthy_kg", "-h", default="",
                        help="healthy knowledge graph")

    args = parser.parse_args()

    # 1. if not sample_scoring_{method}, do sample scoring

    # 2. Generate Sample-KG network


if __name__ == "__main__":
    main()
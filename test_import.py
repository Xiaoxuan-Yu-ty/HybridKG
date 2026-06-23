import argparse
import json
import os
import sys

try:
    base_dir = os.path.dirname(os.path.abspath(__file__))
except NameError:
    base_dir = os.getcwd()
sys.path.append(os.path.dirname(base_dir))

from data_processing.pyg_graph_generator import generat_and_save_hybrid
from data_processing.sample_scoring import *

import gc
import pickle
import json
from typing import Any, Dict, List
import numpy as np
import psutil
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import HeteroData

import pandas as pd
import networkx as nx
import argparse
import os
import sys
import math
from tqdm import tqdm

import optuna
from sklearn.model_selection import StratifiedKFold

from data_processing.pyg_graph_generator import generat_and_save_hybrid
from data_processing.sample_scoring import *
from GateEmbeddingTask.train_utils import (
    compute_link_loss, 
    evaluate_link,
    build_data_dict,
    set_seed,
    convert_to_hetero_data,
    get_device
)
from GateEmbeddingTask.TwoStageMLT.TwoStageModel import get_model, TwoStageModel
from GateEmbeddingTask.TwoStageMLT.train import train_epoch, train, hpo_cross_validate, objective

print("Hello World!")

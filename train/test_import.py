#!/usr/bin/env python3
"""
Training script for fuzzy rule-enhanced FireGNN models.
"""

import argparse
import os
import json
import torch
import torch.nn.functional as F
import numpy as np
from tqdm import trange
import sys

# Add parent directory to path for imports
try:
    base_dir = os.path.dirname(os.path.abspath(__file__))
except NameError:
    base_dir = os.getcwd()
sys.path.append(os.path.dirname(base_dir))
from utils.graph_utils import (
    load_graph,
    prepare_pytorch_geometric_data,
    get_device,
    set_random_seeds
)

from fuzzy_models.fuzzy_models import get_fuzzy_model

print('hello world')
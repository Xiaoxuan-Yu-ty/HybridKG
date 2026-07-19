import torch
import torch.nn as nn
import torch.nn.functional as F
from GateEmbeddingTask.encoders import get_encoder, HRGATEncoder



class FuzzyRuleLayer(nn.Module):
    """
    FireGNN fuzzy rules exactly as defined in the paper.
    Implements Eq. (3): r_i(u) = sigmoid(alpha_i * (f_i(u) - theta_i))
    """

    def __init__(self, num_rules=6):
        super().__init__()
        self.num_rules = num_rules

        # Learnable thresholds θ_i
        self.theta = nn.Parameter(torch.zeros(num_rules))

        # Learnable sharpness α_i (initialized positive)
        self.alpha = nn.Parameter(torch.ones(num_rules))

    def forward(self, topo_features):
        """
        Args:
            topo_features: Tensor [N, 6]
                [degree, clustering coefficient, 2-hop label agreement]

        Returns:
            r: Tensor [N, 6] fuzzy rule activations
        """
        # Ensure correct dimensionality
        #assert topo_features.size(1) == self.num_rules

        # r_i(u) = sigmoid(alpha_i * (f_i(u) - theta_i))
        r = torch.sigmoid(self.alpha * (topo_features - self.theta))
        return r

class GaussianRuleLayer(nn.Module):
    """
    Trainable fuzzy rule layer using Gaussian membership functions.
    """
    
    def __init__(self, num_features, num_rules):
        super().__init__()
        self.num_rules = num_rules
        self.num_features = num_features
        
        # Learnable rule centers and widths
        self.centers = nn.Parameter(torch.randn(num_rules, num_features))
        self.log_sigmas = nn.Parameter(torch.zeros(num_rules, num_features))
        
        # Learnable rule weights
        self.rule_weights = nn.Parameter(torch.ones(num_rules))
        
    def forward(self, x):
        """
        Forward pass through fuzzy rule layer.
        
        Args:
            x: Input features [N, num_features]
            
        Returns:
            torch.Tensor: Fuzzy rule activations [N, num_rules]
        """
        # Expand dimensions for broadcasting
        x_e = x.unsqueeze(1)                 # [N, 1, num_features]
        c = self.centers.unsqueeze(0)        # [1, num_rules, num_features]
        s = torch.exp(self.log_sigmas).unsqueeze(0)  # [1, num_rules, num_features]
        
        # Compute Gaussian membership functions
        gauss = torch.exp(-((x_e - c)**2) / (2*s**2))
        
        # Product of membership functions (AND operation)
        rule_activations = gauss.prod(dim=2)  # [N, num_rules]
        
        # Apply rule weights
        rule_activations = rule_activations * torch.sigmoid(self.rule_weights)
        
        return rule_activations

class FuzzyAggregator(nn.Module):
    def __init__(self, data, 
                 hidden_channels, att_channels,
                 num_classes,
                 dropout_rate, 
                 num_rules,
                 negative_slope=0.2,
                 ):
        super().__init__()

        self.dropout_rate = dropout_rate
        self.data = data
        self.embeddings = nn.ModuleDict({
            ntype:torch.nn.Embedding(data[ntype].num_nodes, hidden_channels) for ntype in data.node_types
        })

        # Project both inputs to same hidden space
        self.proj_gene = nn.Linear(data['Patient'].x.size(-1), hidden_channels)
        self.proj_protein = nn.Linear(data['Protein'].x.size(-1), hidden_channels)
        self.input_norm = nn.LayerNorm(hidden_channels)
        self.output_norm = nn.LayerNorm(hidden_channels)

        self.aggregator = HRGATEncoder(
                data=data,
                hidden_channels=hidden_channels,
                out_channels=hidden_channels,
                att_channels=att_channels,
                num_layers=1,
                dropout_rate=dropout_rate,
                negative_slope=negative_slope,
            ) 
        
        self.gate = nn.Linear(2*hidden_channels,1)   # matches combined dim
        self.classifier = nn.Linear(hidden_channels,num_classes)

        # ------------Fuzzy Rule settings------------------------------------
        self.fuzzy_layer = FuzzyRuleLayer(num_rules=num_rules)
        #rule projection
        self.rule_proj = nn.Linear(num_rules, hidden_channels)
        self.rule_gate = nn.Linear(2*hidden_channels, hidden_channels)
    
    def forward(self, edge_index_dict, x_dict=None, topo_features=None):

        if x_dict is None:
            x_dict = {
                node_type:embedding.weight for node_type, embedding in self.embeddings.items()
            }

        patient_x, protein_embeddings = x_dict['Patient'], x_dict['Protein']

        # Force inputs to that same device
        device = next(self.parameters()).device
        patient_x = patient_x.to(device)
        protein_embeddings = protein_embeddings.to(device)
        
        # Ensure all edge indices are on the same device
        edge_index_dict = {k: v.to(device) for k, v in edge_index_dict.items()}

        # normalize input
        h_gene = self.input_norm(self.proj_gene(patient_x))
        h_protein = self.proj_protein(protein_embeddings)
        
        # encoder handles the nonlinearity consistency 
        # new_x_dict = {'Patient': h_gene, 'Protein': h_protein}
        h_out, att_dict = self.aggregator(x_dict, edge_index_dict)
        
        h_patient = self.output_norm(h_out['Patient'])
        combined = torch.cat([h_gene, h_patient], dim=-1)
        gate = torch.sigmoid(self.gate(combined))
        
        h_fused = (gate * h_gene + (1 - gate) * h_patient)

        # ------- concatenate with topological features ------------------
        embedding = h_fused
        r=None
        # add edge_type attention to topo_features
        layer_att = att_dict[-1] if isinstance(att_dict, list) else att_dict
        # semantic attention tensor: shape:[N, num_relations]
        beta=torch.tensor(layer_att['Patient']['attention'])
    
        if topo_features is not None:
            topo_features = torch.cat([topo_features, beta], dim=-1)
            r = self.fuzzy_layer(topo_features)
            e = self.rule_proj(r) # project topo features to hidden space
            g = torch.sigmoid(self.rule_gate(torch.cat([h_fused, e], dim=1)))
            embedding = g*h_fused + (1-g)*e

        return embedding, att_dict, r 
    
    def classify(self, embedding):
        return self.classifier(embedding)
       
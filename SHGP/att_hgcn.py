import torch
import torch.nn as nn
import torch.nn.functional as F

"""ie-HGCN model: 
1. Relation is defined as (src_NodeType, dst_NodeType);
2. Within relation message aggregation: GCN style;
3. Across relation message fusion: attention -> relation-specific attention

Logic:

for target node type:
	transform target features
	for source relation type:
		1. transform source features
		2. aggregate via adjacency
	compute relation attention
	weighted sum all relation messages

"""

class HeteGCNLayer(nn.Module):

	def __init__(self, net_schema, in_layer_shape, out_layer_shape, type_fusion, type_att_size):
		super(HeteGCNLayer, self).__init__()
		self.net_schema = net_schema
		self.in_layer_shape = in_layer_shape
		self.out_layer_shape = out_layer_shape

		self.hete_agg = nn.ModuleDict()
		for k in net_schema:
			self.hete_agg[k] = HeteAggregateLayer(k, net_schema[k], in_layer_shape, out_layer_shape[k], type_fusion, type_att_size)


	def forward(self, x_dict, adj_dict):
		attention_dict={}
		ret_x_dict = {}
		for k in self.hete_agg.keys():
			ret_x_dict[k],attention_dict[k] = self.hete_agg[k](x_dict, adj_dict[k])

		return ret_x_dict,attention_dict



class HeteAggregateLayer(nn.Module):
	
	def __init__(self, curr_k, nb_list, in_layer_shape, out_shape, type_fusion, type_att_size):
		super(HeteAggregateLayer, self).__init__()
		
		self.nb_list = nb_list # current node type's neighbors NodeType list: [num_NodeType,[num_nodes, in-shape]]
		self.curr_k = curr_k # current node type
		self.type_fusion = type_fusion
		self.W_rel = nn.ParameterDict() # projection of each neighbor NodeType to hidden space -> projection per relation (src_NodeType, dst_NodeType)
		for k in nb_list:
			try:
				self.W_rel[k] = nn.Parameter(torch.FloatTensor(in_layer_shape[k], out_shape))
			except KeyError as ke:
				self.W_rel[k] = nn.Parameter(torch.FloatTensor(in_layer_shape[self.curr_k], out_shape))
			finally:
				nn.init.xavier_uniform_(self.W_rel[k].data, gain=1.414)
		
		self.w_self = nn.Parameter(torch.FloatTensor(in_layer_shape[curr_k], out_shape)) # self transformation
		nn.init.xavier_uniform_(self.w_self.data, gain=1.414)

		self.bias = nn.Parameter(torch.FloatTensor(1, out_shape))
		nn.init.xavier_uniform_(self.bias.data, gain=1.414)

		if type_fusion == 'att':
			self.w_query = nn.Parameter(torch.FloatTensor(out_shape, type_att_size))
			nn.init.xavier_uniform_(self.w_query.data, gain=1.414)
			self.w_keys = nn.Parameter(torch.FloatTensor(out_shape, type_att_size))
			nn.init.xavier_uniform_(self.w_keys.data, gain=1.414)
			self.w_att = nn.Parameter(torch.FloatTensor(2*type_att_size, 1))
			nn.init.xavier_uniform_(self.w_att.data, gain=1.414)


	def forward(self, x_dict, adj_dict):
		attention_curr_k=0
		self_ft = torch.mm(x_dict[self.curr_k], self.w_self) # self feature projection: input feature -> hidden_channels
		
		nb_ft_list = [self_ft]
		nb_name = [self.curr_k+'_self']
		for k in self.nb_list:
			try:
				nb_ft = torch.mm(x_dict[k], self.W_rel[k]) # project neighbor nodes to hidden space
			except KeyError as ke:
				nb_ft = torch.mm(x_dict[self.curr_k], self.W_rel[k])
			finally:
				nb_ft = torch.spmm(adj_dict[k], nb_ft) # Inside (Intra) relation message aggregation: standard message aggregation and passing: 
															# update neighbor_embeddings by adjacency matrix of this neighbor node type
				nb_ft_list.append(nb_ft) # projected neighbor embeddings, include self embedding
				nb_name.append(k) # neighbor names, include self name

		if self.type_fusion == 'mean':
			agg_nb_ft = torch.cat([nb_ft.unsqueeze(1) for nb_ft in nb_ft_list], 1).mean(1)
			attention=[]
		elif self.type_fusion == 'att':
			att_query = torch.mm(self_ft, self.w_query).repeat(len(nb_ft_list), 1) # transform self embedding to Q
			att_keys = torch.mm(torch.cat(nb_ft_list, 0), self.w_keys) # concatenate all neighbors embeddings and transform to K
			att_input = torch.cat([att_keys, att_query], 1)
			att_input = F.dropout(att_input, 0.5, training=self.training)
			e = F.elu(torch.matmul(att_input, self.w_att)) # raw attention scores: 
			attention = F.softmax(e.view(len(nb_ft_list), -1).transpose(0,1), dim=1)   #4025*3 normalized by each NodeType's neighbors num_NodeType
			agg_nb_ft = torch.cat([nb_ft.unsqueeze(1) for nb_ft in nb_ft_list], 1).mul(attention.unsqueeze(-1)).sum(1)
			
		output = agg_nb_ft + self.bias

		return output,attention

class ATT_HGCN(nn.Module):

	def __init__(self, net_schema, layer_shape, label_keys, type_fusion='att', type_att_size=64):
		super(ATT_HGCN, self).__init__()
		self.hgc1 = HeteGCNLayer(net_schema, layer_shape[0], layer_shape[1], type_fusion, type_att_size)
		self.hgc2 = HeteGCNLayer(net_schema, layer_shape[1], layer_shape[2], type_fusion, type_att_size)

		self.embd2class = nn.ParameterDict()
		self.bias = nn.ParameterDict()
		self.label_keys = label_keys
		self.layer_shape=layer_shape
		for k in label_keys:
			self.embd2class[k] = nn.Parameter(torch.FloatTensor(layer_shape[-2][k], layer_shape[-1][k]))
			nn.init.xavier_uniform_(self.embd2class[k].data, gain=1.414)
			self.bias[k] = nn.Parameter(torch.FloatTensor(1, layer_shape[-1][k]))
			nn.init.xavier_uniform_(self.bias[k].data, gain=1.414)
	def ini_embd2class(self):
		for k in self.label_keys:
			nn.init.xavier_uniform_(self.embd2class[k].data, gain=1.414)
			nn.init.xavier_uniform_(self.bias[k].data, gain=1.414)

	def forward(self, ft_dict, adj_dict):
		attention_list=[]
		x_dict,attention_dict = self.hgc1(ft_dict, adj_dict)
		attention_list.append((attention_dict))
		x_dict= self.non_linear(x_dict)
		x_dict = self.dropout_ft(x_dict, 0.5)

		x_dict,attention_dict= self.hgc2(x_dict, adj_dict)
		attention_list.append((attention_dict))

		logits = {}
		embd = {}
		for k in self.label_keys:
			embd[k] = x_dict[k]
			logits[k] = torch.mm(x_dict[k], self.embd2class[k]) + self.bias[k]
		return logits, embd,attention_list


	def non_linear(self, x_dict):
		y_dict = {}
		for k in x_dict:
			y_dict[k] = F.elu(x_dict[k])
		return y_dict


	def dropout_ft(self, x_dict, dropout):
		y_dict = {}
		for k in x_dict:
			y_dict[k] = F.dropout(x_dict[k], dropout, training=self.training)
		return y_dict

from torch_geometric.utils.convert import to_networkx, from_networkx
import networkx as nx
import numpy as np
import torch
import scipy.stats as st
from utils import create_node_features, create_label_dict, get_parameters
from torch_geometric.utils.convert import from_networkx
np.random.seed(14)
cluster = 4
lambdas, n, num_nodes, mu1_edge, mu2_edge, mu, G = get_parameters(cluster)
print(lambdas)
G = create_node_features(G,n,mu)
label_dict = create_label_dict(n)
pyg_graph = from_networkx(G)
# hello
# Data(edge_index=[2, 324], block=[42], feature=[42, 128], num_nodes=42)

#hello, it's steve

import math
import numpy as np
import dgl
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchmetrics.functional import pairwise_cosine_similarity, pairwise_euclidean_distance, \
    pairwise_linear_similarity, pairwise_manhattan_distance

def cosine_similarity(x, y):
    similarity = pairwise_cosine_similarity(x, y)
    return torch.mean(similarity)

def euclidean_dist_similarity(x, y):
    similarity = pairwise_euclidean_distance(x, y)
    # the expression of 1 / (1 + dist) converts the distance into similarity value between 0 and 1
    return 1 / (1 + torch.mean(similarity))

def linear_similarity(x, y):
    similarity = pairwise_linear_similarity(x, y)
    # the expression of 1 / (1 + dist) converts the distance into similarity value between 0 and 1
    return 1 / (1 + torch.mean(similarity))

def manhattan_dist_similarity(x, y):
    similarity = pairwise_manhattan_distance(x, y)
    # the expression of 1 / (1 + dist) converts the distance into similarity value between 0 and 1
    return 1 / (1 + torch.mean(similarity))

class EGraphCONV(nn.Module):
    
    def __init__(self, in_size, e_size, h_size, num_edges=-1, similarity_threshold=0.5,
                 similarity_func=None):
        """
        Def: 
        Custom convolution layer for E-GraphSAGE edge embedding

        Args:
        in_size: Size of input feature (integer)
        e_size: Size of edge feature of graph (integer)
        h_size: Number of hidden units (integer)
        num_edges: Number of edges to be sampled. Default is -1, which means all edges will be sampled. (integer)
        similarity_threshold: Default is 0.5. A float value in between 0 and 1 that will be used to select the edges
                              that have similarity score lower than this value
        similarity_func: Default is None. A selection from four similarity functions can be made by using the
                         following strings such as "cosine", "euclidean", "linear" and "manhattan".
        """
        super(EGraphCONV, self).__init__()
        self.in_size = in_size
        self.e_size = e_size
        self.h_size = h_size
        self.num_edges = num_edges
        self.similarity_threshold = similarity_threshold
        all_functions = {"cosine":cosine_similarity,
                         "euclidean":euclidean_dist_similarity,
                         "linear":linear_similarity,
                         "manhattan":manhattan_dist_similarity}
        if similarity_func is not None:
            assert similarity_func in list(all_functions.keys()), "Invalid name of similarity func"
            self.similarity_func = all_functions[similarity_func]
        else:
            self.similarity_func = similarity_func
        weights = torch.Tensor(h_size, in_size + e_size)
        self.weights = nn.Parameter(weights)
        # initializing weights
        nn.init.kaiming_uniform_(self.weights, a=math.sqrt(0.5))
    
    def forward(self, graph, h_in):
        """
        Def:
        Computes the forward message passing.
        
        Args:
        graph: DGL graph with nodes and edges features
        h_in: Input feature embedding from previous layer of network
        
        Return:
        Embedding of input feature
        """
        # Initializing embeddings per layer of network
        h = []
        h.append(h_in)
        h.append(torch.tensor(np.zeros((h[0].shape[0], self.h_size), np.float32), dtype=torch.float32))
        for v in range(h[0].shape[0]):
            # Edges in the neighbourhood of node v
            Nv = dgl.sampling.sample_neighbors(graph, [v], -1)
            hk_Nv = torch.mean(Nv.edata['weight'], 0)

            if self.similarity_func is None:
                # The following condition will handle the nodes that have no neighbourhood
                if torch.sum(torch.isnan(hk_Nv) * torch.ones((hk_Nv.shape))) != 0:
                    hk_Nv = torch.zeros((hk_Nv.shape))
                
                concat_h = torch.cat((h[0][v], hk_Nv))
                h[1][v] = torch.mm(concat_h.reshape(-1, concat_h.shape[0]), self.weights.t())
            else:
                hk_Nv_reshaped = hk_Nv.reshape(1,-1)
            
                # The following code checks the similarity of each edge feature in the neighbourhood with 
                # the average of edge features. An average of all edge features having similarity less than 
                # similarity_threshold will be calculated
                
                similar_feat = torch.zeros(Nv.edata['weight'].shape[1])
                similar_count = 0
                for i in range(Nv.edata['weight'].shape[0]):
                    similarity = self.similarity_func(hk_Nv_reshaped, Nv.edata['weight'][i].reshape(1,-1))
                    if similarity < self.similarity_threshold:
                        similar_feat += Nv.edata['weight'][i]
                        similar_count += 1

                # The following condition will handle the nodes that have no neighbourhood
                if torch.sum(torch.isnan(hk_Nv) * torch.ones((hk_Nv.shape))) != 0:
                    hk_Nv = torch.zeros((hk_Nv.shape))
            
                if similar_count == 0:
                    concat_h = torch.cat((h[0][v], hk_Nv))
                else:
                    # average of all edge features having similarity less than 0.5
                    similar_feat = similar_feat / similar_count
                    concat_h = torch.cat((h[0][v], similar_feat))
            
                h[1][v] = torch.mm(concat_h.reshape(-1, concat_h.shape[0]), self.weights.t())
        
        return h[1]

class EGraphSAGE(nn.Module):
    
    def __init__(self, in_size, e_size, h_size, similarity_threshold=0.5,
                 similarity_func=None):
        """
        Def: 
        Model for E-GraphSAGE edge embedding

        Args:
        in_size: Size of input feature (integer)
        e_size: Size of edge feature of graph (integer)
        h_size: Number of hidden units for each layer of network (list of integers)
        similarity_threshold: Default is 0.5. A float value in between 0 and 1 that will be used to select the edges
                              that have similarity score lower than this value
        similarity_func: Default is None. A selection from four similarity functions can be made by using the
                         following strings such as "cosine", "euclidean", "linear" and "manhattan".
        """
        super(EGraphSAGE, self).__init__()
        self.list_of_layers = []
        dim = [in_size] + h_size
        for i in range(1, len(dim)):
            self.list_of_layers.append(EGraphCONV(in_size=dim[i-1], e_size=e_size, h_size=dim[i],
                                                  similarity_threshold=similarity_threshold, 
                                                  similarity_func=similarity_func))
        
    def forward(self, graph, in_feat):
        """
        Def:
        Computes the forward message passing.
        
        Args:
        graph: DGL graph with nodes and edges features
        in_feat: Input features for nodes
        
        Return:
        Embedding of edge features of the graph
        """
        h = in_feat
        for i in range(len(self.list_of_layers)):
            h = self.list_of_layers[i](graph, h)
            h = F.relu(h)
        
        U, V = graph.edges()
        z = []
        for u, v in zip(U, V):
            z.append(torch.cat((h[u], h[v])))
        
        return torch.stack(z)

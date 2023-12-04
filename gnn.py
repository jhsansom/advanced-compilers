from graph import InterferenceGraph
from sklearn.manifold import spectral_embedding
import numpy as np
import torch
from torch import nn
import warnings

class GraphNeuralNetwork(nn.Module):

    def __init__(self, embed_dim=32, num_layers=3, out_dim=4):
        super().__init__()
        self.embed_dim = embed_dim
        transformer_layer = nn.TransformerEncoderLayer(
            d_model=self.embed_dim,
            nhead=1,
            dim_feedforward=self.embed_dim,
            batch_first=True
        )
        self.encoder = nn.TransformerEncoder(transformer_layer, num_layers)
        self.output = nn.Linear(self.embed_dim, out_dim)

    def forward(self, graphs):
        embeddings, attns, padding_masks = self.create_embeddings(graphs)

        #out_embeddings = self.encoder(embeddings, src_key_padding_mask=padding_masks, mask=attns)
        out_embeddings = self.encoder(embeddings, src_key_padding_mask=padding_masks)

        embeds = self.output(out_embeddings)

        return nn.functional.normalize(embeds, dim=-1)

    def create_embeddings(self, graphs):

        max_graph_len = 0
        for graph in graphs:
            if len(graph.costList) > max_graph_len:
                max_graph_len = len(graph.costList)

        all_embeddings = []
        attns = torch.zeros((len(graphs), max_graph_len, max_graph_len))
        padding_masks = []

        for i, graph in enumerate(graphs):
            # Create spectral embeddings of nodes
            adjacency = np.array(graph.adjacencyMatrix)
            n_components = min(self.embed_dim, len(graph.adjacencyMatrix)-2)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                embeddings = spectral_embedding(adjacency, n_components=n_components)
            if embeddings.shape[1] < self.embed_dim:
                added_dim = self.embed_dim - embeddings.shape[1]
                added_zeros = np.zeros((embeddings.shape[0], added_dim))
                embeddings = np.concatenate((embeddings, added_zeros), axis=1)
            embeddings = torch.tensor(embeddings, dtype=torch.float)

            # Apply weights to embeddings
            weights = torch.tensor(graph.costList, dtype=torch.float).unsqueeze(1)
            embeddings = embeddings * weights

            # Append embeddings and lengths to a list
            all_embeddings.append(embeddings)
            padding_mask = torch.ones(embeddings.shape[0])
            #padding_mask = torch.tensor(graph.adjacencyMatrix, dtype=torch.bool)
            padding_masks.append(padding_mask)

            attn_mask = torch.tensor(graph.adjacencyMatrix, dtype=torch.bool)
            graph_size = len(attn_mask)
            attns[i, :graph_size, :graph_size] = attn_mask

        all_embeddings = torch.nn.utils.rnn.pad_sequence(all_embeddings, batch_first=True)
        padding_masks = torch.nn.utils.rnn.pad_sequence(padding_masks, batch_first=True)
        #attns = torch.nn.utils.rnn.pad_sequence(attns, batch_first=True)

        #padding_masks = torch.log(padding_masks)
        
        return all_embeddings, attns, padding_masks

'''
nodenames = ["a", "b", "c", "d", "e", "f", "g"]
nodecosts = [225, 200, 175, 150, 200, 50, 200]
adjacencyMatrix = [[0,1,1,1,1,1,1],[1,0,1,1,0,0,0],[1,1,0,1,0,1,1],[1,1,1,0,0,0,0], [1,0,0,1,0,1,0], [1,1,1,0,1,0,0],[1,0,1,0,0,0,0]]

irGraph = InterferenceGraph(nodenames, nodecosts, adjacencyMatrix)

GNN = GraphNeuralNetwork()
y = GNN([irGraph, irGraph])
'''

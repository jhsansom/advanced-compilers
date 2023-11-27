from graph import InterferenceGraph
from sklearn.manifold import spectral_embedding
import numpy as np
import torch
from torch import nn

class GraphNeuralNetwork(nn.Module):

    def __init__(self, embed_dim=32, num_layers=3, max_colors=10):
        super().__init__()
        self.embed_dim = embed_dim
        transformer_layer = nn.TransformerEncoderLayer(
            d_model=self.embed_dim,
            nhead=4,
            dim_feedforward=self.embed_dim,
            batch_first=True
        )
        self.encoder = nn.TransformerEncoder(transformer_layer, num_layers)
        self.output = nn.Linear(self.embed_dim, max_colors)

    def forward(self, graphs):
        embeddings, attns = self.create_embeddings(graphs)

        out_embeddings = self.encoder(embeddings, src_key_padding_mask=attns)

        return self.output(out_embeddings)

    def create_embeddings(self, graphs):
        all_embeddings = []
        attns = []

        for graph in graphs:
            # Create spectral embeddings of nodes
            adjacency = np.array(graph.adjacencyMatrix)
            n_components = min(self.embed_dim, len(graph.adjacencyMatrix)-2)
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
            attn_mask = torch.ones(embeddings.shape[0])
            attns.append(attn_mask)

        all_embeddings = torch.nn.utils.rnn.pad_sequence(all_embeddings, batch_first=True)
        attns = torch.nn.utils.rnn.pad_sequence(attns, batch_first=True)

        attns = torch.log(attns)
        
        return all_embeddings, attns

'''
nodenames = ["a", "b", "c", "d", "e", "f", "g"]
nodecosts = [225, 200, 175, 150, 200, 50, 200]
adjacencyMatrix = [[0,1,1,1,1,1,1],[1,0,1,1,0,0,0],[1,1,0,1,0,1,1],[1,1,1,0,0,0,0], [1,0,0,1,0,1,0], [1,1,1,0,1,0,0],[1,0,1,0,0,0,0]]

irGraph = InterferenceGraph(nodenames, nodecosts, adjacencyMatrix)

GNN = GraphNeuralNetwork()
y = GNN([irGraph, irGraph])
'''

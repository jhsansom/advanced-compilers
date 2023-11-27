from torch.utils.data import Dataset, DataLoader
import torch
from torch import nn
import networkx
from networkx.generators.random_graphs import fast_gnp_random_graph
from graph import InterferenceGraph
import numpy as np
from string import ascii_lowercase
import random
from gnn import GraphNeuralNetwork
from chaitin import findRegularChaitinColoring
from statistics import mean
import math
import wandb

track_via_wandb = True

class GraphDataset(Dataset):

    def __init__(self, num_graphs, p=0.25, max_nodes=100):
        self.num_graphs = num_graphs
        self.max_nodes = max_nodes
        self.p = p

    def __getitem__(self, idx):
        num_nodes = random.randint(3, self.max_nodes)
        weights = np.random.randint(0, 250, num_nodes)

        graph = fast_gnp_random_graph(num_nodes, self.p)

        adjacency = np.zeros((num_nodes, num_nodes), dtype=int)
        for key, value in graph._adj.items():
            for item in value.keys():
                adjacency[key, item] = 1

        labels = []
        for i in range(num_nodes):
            idx = i % 26
            idx_num = i // 26
            label = ascii_lowercase[idx] * (idx_num + 1)
            labels.append(label)

        irGraph = InterferenceGraph(labels, weights, adjacency)

        return irGraph

    def __len__(self):
        return self.num_graphs
    
def collate_fn(inputs):
    return inputs

if track_via_wandb:
    wandb.login()
    run = wandb.init(
        # Set the project where this run will be logged
        project="advanced-compilers"
        # Track hyperparameters and run metadata
        )
    
dataset = GraphDataset(1000)
dataloader = DataLoader(dataset, batch_size=16, shuffle=True, collate_fn=collate_fn)

GNN = GraphNeuralNetwork()
GNN.train()

loss_fn = nn.MSELoss()
optimizer = torch.optim.SGD(GNN.parameters(), lr=0.01, momentum=0.9)
softmax = nn.Softmax(dim=-1)

num_epochs = 100
for e in range(num_epochs):
    print(f'Epoch Num = {e+1}/{num_epochs}')
    for batch in dataloader:
        optimizer.zero_grad()

        out = GNN(batch)

        e = random.random()
        if e < 0.05:
            out_altered = softmax(out)
            (rows, cols, _) = out_altered.shape
            out_altered = torch.reshape(out_altered, (-1, out_altered.shape[-1]))
            colorings = torch.multinomial(out_altered, 1)
            colorings = torch.reshape(colorings, (rows, cols))
        else:
            colorings = torch.argmax(out, dim=2)
        q_values = out.gather(2, colorings.unsqueeze(-1)).squeeze()

        # Take mean
        padding_mask = torch.ones(q_values.shape)
        node_len_list = []
        for i in range(padding_mask.shape[0]):
            num_nodes = batch[i].adjacencyMatrix.shape[0]
            padding_mask[i,num_nodes:] = 0
            node_len_list.append(num_nodes)
        node_len_list = torch.tensor(node_len_list)
        q_values = q_values * padding_mask
        q_values = torch.sum(q_values, 1)
        q_values = q_values / node_len_list

        K = random.randint(3, 10)
        
        spill_costs = []
        spill_costs_chaitin = []
        for i, graph in enumerate(batch):
            coloring = colorings[i, :]
            spill_cost = graph.calc_spill_cost(coloring, K)
            spill_costs.append(spill_cost)

            coloring = findRegularChaitinColoring(graph, K)
            spill_cost_chaitin = 0
            for j, color in enumerate(coloring):
                if color is None:
                    spill_cost_chaitin += graph.costList[j]
            spill_costs_chaitin.append(spill_cost_chaitin)

        # Compute loss and backpropagate gradient
        spill_costs = torch.tensor(spill_costs, dtype=torch.float)
        loss = loss_fn(q_values, spill_costs)
        #print(loss)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(GNN.parameters(), 10)
        optimizer.step()

        # Calculate w.r.t. Chaitin
        avg_gnn = torch.mean(spill_costs)
        avg_chaitin = mean(spill_costs_chaitin)
        ratio = avg_gnn / avg_chaitin
        #print(f'RATIO = {ratio:.3f}')

        if track_via_wandb:
            wandb.log({"loss": loss, "ratio": ratio})

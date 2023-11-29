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
from collections import Counter
import wandb
from torch.utils.tensorboard import SummaryWriter
writer = SummaryWriter()
from sklearn.cluster import KMeans

track_via_wandb = True
track_via_tensorboard = False

class GraphDataset(Dataset):

    def __init__(self, num_graphs, p=0.25, max_nodes=100):
        self.num_graphs = num_graphs
        self.max_nodes = max_nodes
        self.p = p

    def __getitem__(self, idx):
        num_nodes = random.randint(3, self.max_nodes)
        #weights = np.random.randint(1, 1, num_nodes)
        weights = np.ones(num_nodes)

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
    
def compute_loss(gnn_outputs, irGraphs):
    #sims = torch.bmm(gnn_outputs, gnn_outputs.transpose(1, 2))
    sims = torch.cdist(gnn_outputs, gnn_outputs)
    with torch.no_grad():
        adjacencies = torch.zeros(sims.shape, dtype=torch.float)
        non_adjacencies = torch.zeros(sims.shape, dtype=torch.float)
        for i in range(len(irGraphs)):
            seq_len = len(irGraphs[i].costList)
            inverse_eye = torch.ones((seq_len, seq_len)) - torch.eye(seq_len)
            weights = torch.tensor(irGraphs[i].costList, dtype=torch.float).unsqueeze(1)
            adjacency = torch.tensor(irGraphs[i].adjacencyMatrix, dtype=torch.float)
            result = adjacency * weights * inverse_eye
            adjacencies[i,:seq_len,:seq_len] = result
            result = (1 - adjacency) * weights * inverse_eye
            non_adjacencies[i,:seq_len,:seq_len] = result

        neg_scale = torch.count_nonzero(adjacencies)
        pos_scale = torch.count_nonzero(non_adjacencies)

    masked_sims = sims * adjacencies
    masked_sims_non_adjacent = sims * non_adjacencies
    neg = torch.sum(masked_sims)/neg_scale
    pos = torch.sum(masked_sims_non_adjacent)/pos_scale
    batch_sum = -neg + pos

    return batch_sum, neg, pos
    
def collate_fn(inputs):
    return inputs

if track_via_wandb:
    wandb.login()
    run = wandb.init(
        # Set the project where this run will be logged
        project="advanced-compilers"
        # Track hyperparameters and run metadata
        )
if track_via_tensorboard:
    writer = SummaryWriter()
    
dataset = GraphDataset(1000)
dataloader = DataLoader(dataset, batch_size=16, shuffle=True, collate_fn=collate_fn)

GNN = GraphNeuralNetwork()
GNN.train()

loss_fn = nn.MSELoss()
optimizer = torch.optim.SGD(GNN.parameters(), lr=0.001, momentum=0.9)
softmax = nn.Softmax(dim=-1)

ii = 0

num_epochs = 100
for e in range(num_epochs):
    print(f'Epoch Num = {e+1}/{num_epochs}')
    for batch in dataloader:
        optimizer.zero_grad()

        out = GNN(batch)

        # Compute loss and take a gradient descent step
        loss, neg, pos = compute_loss(out, batch)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(GNN.parameters(), 10)
        optimizer.step()
        
        # Evaluate actual spill cost
        spill_costs = []
        spill_costs_chaitin = []
        for i, graph in enumerate(batch):
            ii += 1
            num_nodes = len(graph.costList)
            K = random.randint(3, num_nodes)
            graph_embeds = out[i,:num_nodes,:].squeeze().detach().numpy()

            #skm = SphericalKMeans(n_clusters=K).fit(graph_embeds)
            coloring = KMeans(n_clusters=K, n_init='auto').fit_predict(graph_embeds)
            our_color = coloring

            spill_cost = -graph.calc_spill_cost(coloring, K)
            spill_costs.append(spill_cost*K)

            coloring = findRegularChaitinColoring(graph, K)
            spill_cost_chaitin = 0
            for j, color in enumerate(coloring):
                if color is None:
                    spill_cost_chaitin -= graph.costList[j]
            spill_costs_chaitin.append(spill_cost_chaitin*K)

            if False: #(ii > 500) and (len(graph.costList) > 50):
                print(f'K = {K}')
                print(f'Coloring = {our_color}')
                print(f'Spill cost = {spill_cost}')
                print(f'Adjacency = {graph.adjacencyMatrix}')
                print(f'Output embeddings = {graph_embeds}')
                print(f'Chaitin coloring = {coloring}')
                print(f'Spill cost = {spill_cost_chaitin}')
                ii = 0


        # Calculate w.r.t. Chaitin
        avg_gnn = mean(spill_costs)
        avg_chaitin = mean(spill_costs_chaitin)
        ratio = avg_gnn / avg_chaitin
        #print(f'RATIO = {ratio:.3f}')

        if track_via_wandb:
            wandb.log({"loss": loss, "ratio": ratio, "neg": neg, "pos": pos})
        if track_via_tensorboard:
            writer.add_scalar("loss", loss, e)
            writer.add_scalar("ratio", ratio, e)

    

if track_via_tensorboard:
    writer.close()

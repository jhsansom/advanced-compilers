from gnn import GraphNeuralNetwork
from train_embed2 import RealGraphDataset, collate_fn
from chaitin import findRegularChaitinColoring
from torch.utils.data import DataLoader
from torch_kmeans import SoftKMeans
from statistics import mean
import numpy as np
import torch
import random
import matplotlib.pyplot as plt

# Load already-trained model
GNN = GraphNeuralNetwork(num_layers=2, embed_dim=128)
GNN.load_state_dict(torch.load('./model2.torch'))
GNN.eval()

# Create dataset of real graphs
real_dataset = RealGraphDataset()
dataloader = DataLoader(real_dataset, batch_size=1, shuffle=False, collate_fn=collate_fn)

graph_num = 0
summed = 0
spill_costs = []
spill_costs_chaitin = []
random_spills = []
graph_sizes = []
ratios = []
for k in range(5):
    for jj, batch in enumerate(dataloader):

        out = GNN(batch)
        l2_norm = torch.bmm(out, torch.transpose(out, 1, 2))
        summed += l2_norm.sum()

        for i, graph in enumerate(batch):
            num_nodes = len(graph.costList)
            if num_nodes == 8:
                graph_num += 1
                num_nodes = len(graph.costList)
                graph.costList = [1]*num_nodes # make costs 1
                graph_sizes.append(num_nodes)

                num_nodes = len(graph.adjacencyMatrix)
                K = random.randint(3, num_nodes-1)
                graph_embeds = out[i,:num_nodes,:].unsqueeze(0)

                softkmeans = SoftKMeans(n_clusters=K, verbose=False, n_init='auto')
                cluster_result = softkmeans(graph_embeds)

                soft_assignment = cluster_result.soft_assignment.squeeze()
                coloring = cluster_result.labels.squeeze().detach().numpy()

                spill_cost, spilled = graph.calc_spill_cost(coloring, K)
                spill_cost = -spill_cost
                spill_costs.append(spill_cost)

                coloring = findRegularChaitinColoring(graph, K)
                spill_cost_chaitin = 0
                for j, color in enumerate(coloring):
                    if color is None:
                        spill_cost_chaitin -= graph.costList[j]
                spill_costs_chaitin.append(spill_cost_chaitin)

                random_coloring = np.random.randint(0, K, len(coloring))
                random_spill_cost, spilled = graph.calc_spill_cost(random_coloring, K)
                random_spill_cost = -random_spill_cost
                random_spills.append(random_spill_cost)

                if spill_cost_chaitin == 0:
                    if spill_cost == 0:
                        ratio = 1
                    else:
                        ratio = 0#1_000_000
                else:
                    ratio = spill_cost / spill_cost_chaitin
                ratios.append(ratio)

                # Print out results
                print(f'Graph #{graph_num}:')
                print(f'Chaitin Cost = {-spill_cost_chaitin}')
                print(f'Random Cost = {-random_spill_cost}')
                print(f'Our Cost = {-spill_cost}')
                if spill_cost > spill_cost_chaitin:
                    print('======= GRAPH COLORING SUCCESSFUL ========')
                print()


avg_gnn = mean(spill_costs)
avg_chaitin = mean(spill_costs_chaitin)
avg_random = mean(random_spills)
ratio = avg_gnn / (avg_chaitin - 1e-7)

print('RESULTS ON REAL DATA:')
print(f'Ratio between GNN and Chaitin: {ratio:.3f}')
print(f'Avg GNN: {avg_gnn:.3f}')
print(f'Avg Chaitin: {avg_chaitin:.3f}')
print(f'Avg Random: {avg_random}')

#plt.scatter(graph_sizes, ratios)
#plt.show()
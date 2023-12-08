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
#from sklearn.cluster import KMeans
from torch_kmeans import SoftKMeans
from os import listdir
from os.path import isfile, join
from graph import read_from_csv

track_via_wandb = False
track_via_tensorboard = True
run_on_real_graphs = True

# Synthetic graph generation function
def generate_random_graph(n_nodes):
    """
    Generates a random graph with each node connected to at least one other node.

    :param n_nodes: Number of nodes in the graph
    :return: A networkx graph
    """
    if n_nodes < 2:
        raise ValueError("Number of nodes must be at least 2.")

    G = networkx.Graph()

    # Add nodes
    G.add_nodes_from(range(n_nodes))

    # Ensure each node is connected to at least one other node
    for node in range(n_nodes):
        # Choose a node to connect that is not the current node
        other_node = random.choice([n for n in range(n_nodes) if n != node])
        G.add_edge(node, other_node)

    # Optionally, add more edges randomly
    additional_edges = random.randint(0, n_nodes * (n_nodes - 1) // 16)
    for _ in range(additional_edges):
        node1, node2 = random.sample(range(n_nodes), 2)
        G.add_edge(node1, node2)

    return G


# Composed of graphs from LLVM
class RealGraphDataset(Dataset):

    def __init__(self):
        mypath = './graphs/'
        onlyfiles = [f for f in listdir(mypath) if isfile(join(mypath, f))]
        graphs = []
        for filename in onlyfiles:
            full_filename = join(mypath, filename)
            graph = read_from_csv(full_filename)
            if len(graph.adjacencyMatrix) > 3:
                graphs.append(graph)

        self.graphs = graphs

    def __getitem__(self, idx):
        return self.graphs[idx]

    def __len__(self):
        return len(self.graphs)

# Composed of synthetically generated graphs
class GraphDataset(Dataset):

    def __init__(self, num_graphs, p=0.25, max_nodes=80):
        self.num_graphs = num_graphs
        self.max_nodes = max_nodes
        self.p = p

    def __getitem__(self, idx):
        num_nodes = random.randint(4, self.max_nodes)
        #num_nodes = random.randint(9, 10)
        #num_nodes = 10
        weights = np.random.randint(1, 50, num_nodes)
        #weights = np.ones(num_nodes)

        #graph = fast_gnp_random_graph(num_nodes, self.p)
        graph = generate_random_graph(num_nodes)

        #adjacency = np.zeros((num_nodes, num_nodes), dtype=int)
        adjacency = np.eye(num_nodes, dtype=int)
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

if __name__ == '__main__':

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
    dataloader = DataLoader(dataset, batch_size=64, shuffle=True, collate_fn=collate_fn)

    real_dataset = RealGraphDataset()
    (train, test) = torch.utils.data.random_split(real_dataset, [0.05, 0.95])
    train_dataloader = DataLoader(train, batch_size=64, shuffle=True, collate_fn=collate_fn)
    test_dataloader = DataLoader(test, batch_size=64, shuffle=True, collate_fn=collate_fn)

    GNN = GraphNeuralNetwork(num_layers=2, embed_dim=128)
    GNN.train()

    optimizer = torch.optim.SGD(GNN.parameters(), lr=0.001, momentum=0.9)
    softmax = nn.Softmax(dim=-1)

    ii = 0

    num_epochs = 50
    for e in range(num_epochs):
        print(f'\nEpoch Num = {e+1}/{num_epochs}')
        for jj, batch in enumerate(dataloader):
            optimizer.zero_grad()

            # Get embeddings for all nodes across all graphs in the batch
            out = GNN(batch)

            loss = 0
            
            spill_costs = []
            spill_costs_chaitin = []
            for i, graph in enumerate(batch):
                ii += 1
                num_nodes = len(graph.costList)
                K = random.randint(3, num_nodes-1)
                graph_embeds = out[i,:num_nodes,:].unsqueeze(0)#.squeeze().detach().numpy()

                #skm = SphericalKMeans(n_clusters=K).fit(graph_embeds)
                #coloring = SoftKMeans(n_clusters=K, n_init='auto').fit_predict(graph_embeds)
                softkmeans = SoftKMeans(n_clusters=K, verbose=False, n_init='auto', init_method='rnd')
                cluster_result = softkmeans(graph_embeds)
                soft_assignment = cluster_result.soft_assignment.squeeze()
                coloring = cluster_result.labels.squeeze().detach().numpy()
                

                spill_cost, spilled = graph.calc_spill_cost(coloring, K)
                spill_cost = -spill_cost
                #loss += spill_cost
                spill_costs.append(spill_cost*K)


                # Construct ground truth labels
                #default_color = torch.argmax(soft_assignment, dim=-1)
                num_spilled = len(spilled)
                num_not_spilled = num_nodes - num_spilled
                if not torch.any(torch.isnan(soft_assignment)):
                    for j in range(num_nodes):
                        if j in spilled:
                            # Reduce probability of spilled graph coloring
                            loss += torch.log(soft_assignment[j, coloring[j]]) / num_spilled
                        else:
                            # Increase probability of successful coloring
                            loss -= torch.log(soft_assignment[j, coloring[j]]) / num_not_spilled
                else:
                    print('skipped this one')

                # Get the Chaitin coloring for performance comparison
                coloring = findRegularChaitinColoring(graph, K)
                spill_cost_chaitin = 0
                for j, color in enumerate(coloring):
                    if color is None:
                        spill_cost_chaitin -= graph.costList[j]
                spill_costs_chaitin.append(spill_cost_chaitin*K)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(GNN.parameters(), 1)
            optimizer.step()

            # Calculate w.r.t. Chaitin
            avg_gnn = mean(spill_costs)
            avg_chaitin = mean(spill_costs_chaitin)
            ratio = avg_gnn / (avg_chaitin - 1e-7)
            #print(f'RATIO = {ratio:.3f}')

            if track_via_wandb:
                wandb.log({"loss": loss, "ratio": ratio})
            if track_via_tensorboard:
                writer.add_scalar("loss", loss, jj + e*dataloader.__len__())
                writer.add_scalar("ratio", ratio, jj + e*dataloader.__len__())


        if run_on_real_graphs:

            spill_costs = []
            spill_costs_chaitin = []

            summed = 0
            for batch in test_dataloader:

                out = GNN(batch)
                l2_norm = torch.bmm(out, torch.transpose(out, 1, 2))
                summed += l2_norm.sum()

                for i, graph in enumerate(batch):
                    num_nodes = len(graph.adjacencyMatrix)
                    K = random.randint(3, num_nodes-1)
                    graph_embeds = out[i,:num_nodes,:].unsqueeze(0)#.squeeze().detach().numpy()

                    #skm = SphericalKMeans(n_clusters=K).fit(graph_embeds)
                    #coloring = SoftKMeans(n_clusters=K, n_init='auto').fit_predict(graph_embeds)
                    softkmeans = SoftKMeans(n_clusters=K, verbose=False, n_init='auto')
                    cluster_result = softkmeans(graph_embeds)
                    soft_assignment = cluster_result.soft_assignment.squeeze()
                    coloring = cluster_result.labels.squeeze().detach().numpy()

                    spill_cost, spilled = graph.calc_spill_cost(coloring, K)
                    spill_cost = -spill_cost
                    #loss += spill_cost
                    spill_costs.append(spill_cost*K)

                    coloring = findRegularChaitinColoring(graph, K)
                    spill_cost_chaitin = 0
                    for j, color in enumerate(coloring):
                        if color is None:
                            spill_cost_chaitin -= graph.costList[j]
                    spill_costs_chaitin.append(spill_cost_chaitin*K)

            avg_gnn = mean(spill_costs)
            avg_chaitin = mean(spill_costs_chaitin)
            ratio = avg_gnn / (avg_chaitin - 1e-7)

            print('RESULTS ON REAL DATA:')
            print(f'Ratio between GNN and Chaitin: {ratio:.3f}')
            print(f'Avg GNN: {avg_gnn:.3f}')
            print(f'Avg Chaitin: {avg_chaitin:.3f}')

            if track_via_tensorboard:
                writer.add_scalar("real_avg_gnn", avg_gnn, e)
                writer.add_scalar("real_avg_chaitin", avg_chaitin, e)
                writer.add_scalar("real_ratio", ratio, e)
                writer.add_scalar("summed", summed, e)

    if track_via_tensorboard:
        writer.close()

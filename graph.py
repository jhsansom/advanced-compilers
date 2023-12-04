class Node:
    def __init__(self, label, cost):
        self.label = label      # to give the node a label like "a" or "b"
        self.indexNumber = -1
        self.cost = cost        # Note - this cost should be Sum(ExecutionFreq * SpillCost)
        self.neighbors = []     # index numbers of the neighbors
        self.priority = 0       # Used in the coloring


class InterferenceGraph:

    def __init__(self, labelList, costList, adjacencyMatrix):
        self.adjacencyMatrix = adjacencyMatrix
        self.costList = costList
        self.nodeList = []
        # Just make sure that the input dimensions are correct
        assert (len(labelList) == len(costList)) and (len(costList) == len(adjacencyMatrix)) and (len(adjacencyMatrix) == len(adjacencyMatrix[0]))

        # Populate node objects
        for i in range(len(labelList)):
            newNode = Node(labelList[i], costList[i])
            newNode.indexNumber = i
            self.nodeList.append(newNode)

        for i in range(len(self.nodeList)):
            neighbors = adjacencyMatrix[i]
            for j in range(len(neighbors)):
                if neighbors[j]:
                    self.nodeList[i].neighbors.append(j)

    '''
        Inputs:
            - coloring: list of colorings [0, 2, 1, 3, ...], where 0 is most prioritized color
    '''
    def calc_spill_cost(self, coloring, K):
        spilled = []
        cost = 0
        num_nodes = self.adjacencyMatrix.shape[0]

        for i in range(num_nodes):
            if coloring[i] >= K:
                spilled.append(i)
                cost += self.costList[i]

        for i in range(num_nodes):
            for j in range(num_nodes):
                to_check = (self.adjacencyMatrix[i,j] == 1)
                to_check = to_check and (coloring[i] == coloring[j])
                to_check = to_check and (i not in spilled)
                to_check = to_check and (j not in spilled)
                if to_check:
                    if self.costList[i] > self.costList[j]:
                        spilled.append(i)
                        cost += self.costList[i]
                    else:
                        spilled.append(j)
                        cost += self.costList[j]

        return cost, spilled
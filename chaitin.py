"""
Simple python implementation of Chaitin-Briggs coloring
"""

from graph import InterferenceGraph
import copy

# Given K colors C1 .... CK, assigns them to nodes attempting to reduce spill costs
# Spilled nodes have the color None
# Regular Chaitin algorithm just computes the costs at the start - this code supports that by default
# If recalculate priority is set to True, then the priority is recalculated each time nodes are removed from the graph
def findRegularChaitinColoring(graph, K, recalculatePriority = False):
    colorNames = ["C"+str(i+1) for i in range(K)]
    nodeStack = []
    spillStack = []
    nodeCopy = copy.deepcopy(graph.nodeList)
    assigned = [None for i in range(len(graph.nodeList))]

    # Regular Chaitin just computes the priority by using Cost(x)/N at the start
    for node in nodeCopy:
        # If there aren't any neighbors, then it doesn't matter. The <K condition will always apply to these nodes
        if len(node.neighbors): 
            node.priority = node.cost/len(node.neighbors)
        # Create a backup of the neighbors
        node.originalNeighbors = copy.deepcopy(node.neighbors)
    

    # While the copy of the interference graph is not empty
    while len(nodeCopy):
        # First remove all nodes in the graph that have <K neighbors
        # get the indices of the nodes and add to the stack
        removeIndices = []
        for node in nodeCopy:
            if len(node.neighbors) < K:
                removeIndices.append(node.indexNumber)
                nodeStack.append(node)
                spillStack.append(0)

        # hacky way to delete - make a copy and then rename
        reducedNodes = []
        for node in nodeCopy:
            if node.indexNumber not in removeIndices:
                for index in removeIndices:
                    if index in node.neighbors:
                        node.neighbors.remove(index)
                reducedNodes.append(node)

        del nodeCopy
        nodeCopy = copy.deepcopy(reducedNodes)

        if recalculatePriority:
            for node in nodeCopy:
                if len(node.neighbors): # this if condition shouldn't actually be needed anymore
                    node.priority = node.cost/len(node.neighbors)

        # Now the graph is reduced to the intereference graph where all nodes have >= K edges
        if len(nodeCopy):
            # Find the node with the lowest priority
            lp = nodeCopy[0].priority
            lowestPriorityIndex = nodeCopy[0].indexNumber
            lpNode = nodeCopy[0]
            for node in nodeCopy:
                if node.priority < lp:
                    lpNode = node
                    lp = node.priority
                    lowestPriorityIndex = node.indexNumber
            # Throw this into the stack, but mark it as spilled
            nodeStack.append(lpNode)
            spillStack.append(1)
            nodeCopy.remove(lpNode)
            for node in nodeCopy:
                if lowestPriorityIndex in node.neighbors:
                    node.neighbors.remove(lowestPriorityIndex)

    # Node stack is now done
    # Spilled nodes are identified
    # Now color by popping off the stack and using the first available color

    nodeStack = nodeStack[::-1]
    spillStack = spillStack[::-1]

    
    for i in range(len(nodeStack)):
        if not spillStack[i]:
            currentNode = nodeStack[i]
            availableColors = copy.deepcopy(colorNames)
            for j in currentNode.originalNeighbors:
                if (assigned[j] is not None) and (assigned[j] in availableColors):
                    availableColors.remove(assigned[j])
            assigned[currentNode.indexNumber] = availableColors[0]

    return assigned



if __name__ == '__main__':
    # Example 1 from lecture

    nodenames = ["a", "b", "c", "d", "e", "f", "g"]
    nodecosts = [225, 200, 175, 150, 200, 50, 200]
    adjacencyMatrix = [[0,1,1,1,1,1,1],[1,0,1,1,0,0,0],[1,1,0,1,0,1,1],[1,1,1,0,0,0], [1,0,0,1,0,1,0], [1,1,1,0,1,0,0],[1,0,1,0,0,0,0]]

    irGraph = InterferenceGraph(nodenames, nodecosts, adjacencyMatrix)

    K = 3
    print("Example 1:\n")
    print("Regular Chaitin Coloring for K=",K)
    colorList = findRegularChaitinColoring(irGraph, 3)
    for i in range(len(nodenames)):
        print(nodenames[i], "colored", colorList[i])
    print("--------------------")
    print("Modified Chaitin Coloring for K=",K)
    colorList = findRegularChaitinColoring(irGraph, 3, recalculatePriority = True)
    for i in range(len(nodenames)):
        print(nodenames[i], "colored", colorList[i])
    print("--------------------\n\n") 
    print()


    # Example 2 from lecture (HW problem)
    print("Example 2:\n")
    nodenames = ["r1", "r2", "r3", "r4", "r5"]
    nodecosts = [11, 18, 20, 6, 11]
    adjacencyMatrix = [[0,1,1,1,1], [1,0,1,1,1], [1,1,0,1,1], [1,1,1,0,0], [1,1,1,0,0]]
    irGraph = InterferenceGraph(nodenames, nodecosts, adjacencyMatrix)
    K = 3
    print("Regular Chaitin Coloring for K=",K)
    colorList = findRegularChaitinColoring(irGraph, 3)
    for i in range(len(nodenames)):
        print(nodenames[i], "colored", colorList[i])
    print("--------------------")
    print("Modified Chaitin Coloring for K=",K)
    colorList = findRegularChaitinColoring(irGraph, 3, recalculatePriority = True)
    for i in range(len(nodenames)):
        print(nodenames[i], "colored", colorList[i])
    print("--------------------")

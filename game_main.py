import networkx as nx
import matplotlib.pyplot as plt
from simulations import *


def initalize(nodes, adj, num_nodes):
    random.seed()
    for i in range (0, num_nodes):
        status = random.randint(0, 1)     # 0 = cooperate, 1 = defect
        nodes[i].status = status
    for i in range (0, num_nodes):
        for j in range (0, i):
            edge_type = random.randint(0, 1)     # 0 = no edge_type, 1 = edge_type
            adj[i][j] = edge_type
            adj[j][i] = edge_type


def visualization(nodes, adj):
    G = nx.convert_matrix.from_numpy_matrix(adj)
    nodes_color = []

    color = ['#03b500', '#b52a00']         # green, red (orangish)
    for i in range (len(nodes)):
        if nodes[i].status == 0:
            nodes_color.append(color[0])    # green = cooperate
        else:
            nodes_color.append(color[1])    # red = defect

    nx.draw(G, with_labels=True, node_color = nodes_color)
    plt.show()


if __name__ == '__main__':
        numNodes = 100
        numIterations = 100
        nodes = {nodeID: Node() for nodeID in range(numNodes)}
        adj = np.zeros((numNodes, numNodes))
        initalize(nodes, adj, numNodes)

        saveRate = 20
        simulation_simplest = Simulation(numIterations, saveRate, strategy= 0) # choose from 0, 1, 2
        visualization(nodes, adj)

        nodes, adj = simulation_simplest(nodes, adj)
        visualization(nodes, adj)
        


import math
import random
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

def initalize(nodes, adj, num_nodes):
        random.seed()
        for i in range (0, num_nodes):
                status = random.randint(0, 1)     # 0 = cooperate, 1 = defect
                nodes[i] = status
        for i in range (0, num_nodes):
                for j in range (0, i):
                        edge_type = random.randint(0, 1)     # 0 = no edge_type, 1 = edge_type
                        adj[i][j] = edge_type
                        adj[j][i] = edge_type

                

def simulation(nodes, a, num_iterations):
        for _ in range (num_iterations):
                for i in range (0, num_nodes):
                        for j in range (0, i):
                                if adj[i][j]:
                                        if nodes[i] == 1 or nodes[j] == 1:
                                                adj[i][j] = 0
                                                adj[j][i] = 0
                                        else:
                                                adj[i][j] = 1
                                                adj[j][i] = 1

def visualization(nodes, adj):
        G = nx.convert_matrix.from_numpy_matrix(adj)
        nodes_color = []

        color = ['#03b500', '#b52a00']         # green, red (orangish)
        for i in range (len(nodes)):
                if nodes[i] == 0:
                        nodes_color.append(color[0])    # green = cooperate
                else:
                        nodes_color.append(color[1])    # red = defect
        
        nx.draw(G, with_labels=True, node_color = nodes_color)
        plt.show()

if __name__ == '__main__':
        #  defining variables
        num_nodes = 100
        num_iterations = 100
        nodes = {}
        adj = np.zeros((num_nodes, num_nodes))

        # begin simulation
        initalize(nodes, adj, num_nodes)
        # print(adj)
        # print()
        visualization(nodes, adj)
        simulation(nodes, adj, num_iterations)
        # print(adj)
        visualization(nodes, adj)
        


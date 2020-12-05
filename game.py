import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from visualization import *
from copy import *


class Node:
    def __init__(self):
        self.status = 0 # 0 = cooperate, 1 = defect
        self.wealth = 0
        self.lastPayoff = []

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

def simulation(nodes, a, num_iterations):
        random.seed()
        for _ in range (num_iterations):
                for i in range (0, num_nodes):
                        for j in range (0, i):
                                if adj[i][j]:
                                        if nodes[i].status == 1 or nodes[j].status == 1:
                                                if (random.uniform(0, 1) <= 0.1):      # 10 percent chance
                                                        adj[i][j] = 0
                                                        adj[j][i] = 0
                                        else:
                                                adj[i][j] = 1
                                                adj[j][i] = 1
        for a in nodes.values():
                if (random.uniform(0, 1) <= 0.1):       # 20% chance a node just wants to change over
                        a.status = 0 if a.status==1 else 1

if __name__ == '__main__':
        #  defining variables
        num_nodes = 10
        numIterations = 10
        nodes = {x:Node() for x in range (num_nodes)}
        adj = np.zeros((num_nodes, num_nodes))

        # begin simulation
        initalize(nodes, adj, num_nodes)
        
        nodes_list, adj_list = [], []
        nodes_list.append(deepcopy(nodes))
        adj_list.append(np.copy(adj))
        
        for i in range(0, numIterations):
                simulation(nodes, adj, 1)
                nodes_list.append(deepcopy(nodes))
                adj_list.append(np.copy(adj))
                
                
        for state in nodes_list:
                for i in state.values():
                        print(i.status, end=", ") 
                print()


        visualize_list(nodes_list, adj_list, numIterations, "ER+strat2")        # 4th parameter (model name) is for bookkeeping purposes

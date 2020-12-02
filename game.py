import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from visualization import *

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

if __name__ == '__main__':
        #  defining variables
        num_nodes = 100
        num_iterations = 10
        nodes = {x:Node() for x in range (num_nodes)}
        adj = np.zeros((num_nodes, num_nodes))

        # begin simulation
        initalize(nodes, adj, num_nodes)
        
        for i in range(0, 60):
                print("Iteration#: " + str(i))
                simulation(nodes, adj, 1)
                visualization(nodes, adj, i)
        print("Generation Complete")
        
        generate_gif()         # GIF is stored in 'animation' folder
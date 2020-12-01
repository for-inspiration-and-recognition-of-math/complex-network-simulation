import math
import random
import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import os
import imageio
import time
from tqdm import tqdm

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
        random.seed()
        for _ in range (num_iterations):
                for i in range (0, num_nodes):
                        for j in range (0, i):
                                if adj[i][j]:
                                        if nodes[i] == 1 or nodes[j] == 1:
                                                if (random.uniform(0, 1) <= 0.1):      # 10 percent chance
                                                        adj[i][j] = 0
                                                        adj[j][i] = 0
                                        else:
                                                adj[i][j] = 1
                                                adj[j][i] = 1

'''
func visualization
@param 
        nodes: dictionary of nodes
        adj: adjacency matrix (numpy 2D array)
        index (optional): current simulation iteration
        color_edges (true by default): colors edges depending on the nodes attached
@return
        none
@usage
        by passing in a dictionary of nodes along with its adjacency matrix
        a visualization of the data will be generated
        if an optional index is passed, a PNG named: "[index].png" will be generated 
        in "graph" subdirectory of current working directory
        E.g. index of 1 will generate "1.png" in CWD/graph
        if an optional color_edges is passed, then edges will be colored with this rule:
                if both stubs connect to cooperating nodes, edge = dark green
                if both stubs connect to defecting nodes, edge = red
                if one stub connects to cooperating node and the other defecting node, edge = yellow
'''
def visualization(nodes, adj, index=-1, color_edges=True):
        G = nx.convert_matrix.from_numpy_matrix(adj)
        # green, red (orangish), dark green, yellow, red (leaning magenta)
        color = ['#03b500', '#b52a00', "#005907", "#debd00", "#db0231", ]         
        nodes_color = [color[1] if (nodes[i]) else color[0] for i in range (len(nodes))]

        optimized_pos = nx.spring_layout(G, iterations=20)
        plt.figure(figsize = (10, 10))
        edge_width = 0.1
        edge_alpha = 0.7
        nx.draw(G, optimized_pos, with_labels=False, node_color = nodes_color, node_size = 15, width=edge_width)

        # Add relationship-sensitive coloring to edges
        if color_edges:
                good_good_edges = []
                bad_bad_edges = []
                mixed_edges = []

                for i in range (len(adj)):
                        for j in range (len(adj)):
                                if adj[i][j] == 1:
                                        if nodes[i] == 0 and nodes[j] == 0: good_good_edges.append((i,j))
                                        elif nodes[i] == 1 and nodes[j] == 1: bad_bad_edges.append((i,j))
                                        else: mixed_edges.append((i,j))

                nx.draw_networkx_edges(G,optimized_pos,
                        edgelist=good_good_edges,
                        width=edge_width,alpha=edge_alpha,edge_color=color[2])
                nx.draw_networkx_edges(G,optimized_pos,
                        edgelist=mixed_edges,
                        width=edge_width,alpha=edge_alpha,edge_color=color[3])
                nx.draw_networkx_edges(G,optimized_pos,
                        edgelist=bad_bad_edges,
                        width=edge_width,alpha=edge_alpha,edge_color=color[4])
        
        # creating file directories and outputting graphs as PNGs
        if index != -1:
                dir_path = os.path.dirname(os.path.realpath(__file__))          # NOT os.getcwd() <——> this incantation is faulty
                path_graph = dir_path + "/graph/"
                path_csv = dir_path + "/csv/"
                mode = 0o755
                # output graph png
                try:  
                        os.makedirs(path_graph, mode)
                        os.makedirs(path_csv, mode)
                except OSError as error:  
                        #print("Directory Already Exists")   
                        pass
                finally:
                        plt.savefig(path_graph + repr(index) + ".png", format="PNG")
                        dataframe = pd.DataFrame(adj)
                        dataframe.to_csv(path_csv + repr(index) + ".csv")
                        
        #plt.show()
        plt.close()

'''
func generate_gif
@param path: directory path to image folder
@return none
@output compiles all images by index into "animated.gif", and outputs gif into @path
Note: only call when all graphs have been generated using func visualization(...)
'''
def generate_gif(path):
        my_path = path
        # basically the python version of regular expression search of list segments
        # you can also use globbing and bash commands, but apparently wildcard symbols can be unstable across different shell versions
        only_PNGs = [os.path.join(my_path, f) for f in os.listdir(my_path) if os.path.isfile(os.path.join(my_path, f)) and f.endswith(".png")]
        
        # encapsulated helper function to parse unconventional file names
        def digit_grub(f):
                digits = list(filter(str.isdigit, f))
                list_to_string = ''.join(map(str, digits))
                return int(list_to_string)
        
        only_PNGs.sort(key=digit_grub, reverse=False) 

        sizecounter = 0
        for filepath in only_PNGs:
                sizecounter += os.stat(filepath).st_size

        with tqdm(total=sizecounter, unit='B', unit_scale=True, unit_divisor=1024) as pbar:
                with imageio.get_writer(os.path.join(my_path, "animated.gif"), mode='I', duration=0.2) as writer:
                        for pic in only_PNGs:
                                image = imageio.imread(pic)
                                writer.append_data(image)
                                pbar.set_postfix(file=pic, refresh=True)
                                pbar.update(os.stat(pic).st_size)
        

if __name__ == '__main__':
        #  defining variables
        num_nodes = 100
        num_iterations = 10
        nodes = {}
        adj = np.zeros((num_nodes, num_nodes))

        # begin simulation
        initalize(nodes, adj, num_nodes)
        
        for i in range(0, 60):
                print("Iteration#: " + str(i))
                simulation(nodes, adj, 1)
                visualization(nodes, adj, i)
        print("Generation Complete")
        print("Compiling GIF...")

        generate_gif("./graph")         # GIF is stored in graph folder
        


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


class Node:
    def __init__(self, status=0, wealth=0):
        self.status = status
        self.history = []

    def update(self, status, wealth):
        self.status = status

# helper function: create directory if not already exists
def creat_dir(folder_name):
        dir_path = os.path.dirname(os.path.realpath(__file__))          # NOT os.getcwd() <——> this incantation is faulty
        path = "{directory}/{subdirectory}/".format(directory = dir_path, subdirectory=folder_name)
        mode = 0o755
        # output graph png
        try:  
                os.makedirs(path, mode)
        except OSError as error:  
                pass
        return path

'''
func generate_png_csv
@param 
        nodes: dictionary of nodes
        adj: adjacency matrix (numpy 2D array)
        index (optional): current simulation iteration (needed to generate png, else just oopens for 1 time viewing)
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
        color = ['#03b500', '#b52a00', "#005907", "#f5f122", "#db0231"]         
        nodes_color = [color[1] if (nodes[i].status) else color[0] for i in range (len(nodes))]

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
                                        if nodes[i].status == 0 and nodes[j].status == 0: good_good_edges.append((i,j))
                                        elif nodes[i].status == 1 and nodes[j].status == 1: bad_bad_edges.append((i,j))
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
        path_graph, path_csv = creat_dir("graph"), creat_dir("csv")
        if index != -1:
                # output graph png
                plt.savefig(path_graph + repr(index) + ".png", format="PNG")
                dataframe = pd.DataFrame(adj)
                dataframe.to_csv(path_csv + repr(index) + ".csv")
        else:               
                plt.show()
        plt.close()

'''
func generate_gif
@param 
        input_path: directory path to image folder
        index (optional): current gif iteration (recommended if more than 1 gif is generated)
@return none
@output compiles all images by index into "animated.gif", and outputs gif into /animation
Note: only call when all graphs have been generated using func visualization(...)
'''
def generate_gif(input_path="./graph/", index=1):
        print("Compiling GIF...")
        my_path = input_path
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
        
        path_animation = creat_dir("animation")
        with tqdm(total=sizecounter, unit='B', unit_scale=True, unit_divisor=1024) as pbar:
                with imageio.get_writer(os.path.join(path_animation, "animated-" + repr(index) + ".gif"), mode='I', duration=0.2) as writer:
                        for pic in only_PNGs:
                                image = imageio.imread(pic)
                                writer.append_data(image)
                                pbar.set_postfix(file=pic, refresh=True)
                                pbar.update(os.stat(pic).st_size)
        print("GIF Generated: stored in 'animation' subdirectory")
        
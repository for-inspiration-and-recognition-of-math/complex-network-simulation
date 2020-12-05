import math
import random
import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import collections
import os
import imageio
import time
from tqdm import tqdm

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
        abs_path: absolute path to the subdirectory in which all graphs will be stored (defaults to "graph")
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
def visualization(nodes, adj, path_network, path_node_histogram, path_edge_histogram, index=-1, color_edges=True):
        G = nx.convert_matrix.from_numpy_matrix(adj)
        # node colors [0,1] : green, red (orangish), 
        # edge colors [2,3,4] : dark green, yellow, red (leaning magenta)
        color = ['#03b500', '#b52a00', "#005907", "#f5f122", "#db0231"]         
        nodes_color = [color[1] if (nodes[i].status == 1) else color[0] for i in range (len(nodes))]

        optimized_pos = nx.spring_layout(G, iterations=20)
        plt.figure(figsize = (10, 10))
        plt.title("Iteration={0}".format(index))
        
        
        # ---------------------------------------------------------
        # generating network visualization
        # ---------------------------------------------------------
        
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
                                if adj[i][j] > 0:
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
        
        # saving network graphs and histograms as PNGs in separate folders
        if index != -1: plt.savefig(path_network + "net-" + repr(index) + ".png", format="PNG")      # output graph png 
        else: plt.show()
        plt.close()
        
        
        # ---------------------------------------------------------
        # plotting degree distribution histogram
        # ---------------------------------------------------------
        
        # 1. edge histogram

        heights = [len(good_good_edges), len(mixed_edges), len(bad_bad_edges)]
        edge_types = ("good-good", "mixed", "bad-bad")
        
        bar_spacing = np.arange(len(edge_types))
        plt.bar(bar_spacing, heights, width=0.80, color=[color[2], color[3], color[4]])    # generate the histogram
        
        # setting attributes of bar graph
        plt.title("Edge Type Distribution (iter={0})".format(index))
        plt.ylabel("Number of Edges")
        plt.xlabel("Edge Type")
        
        plt.xticks(bar_spacing, edge_types)
        
        if index != -1: plt.savefig(path_edge_histogram + "edge-" + repr(index) + ".png", format="PNG")      # output graph png 
        else: plt.show()
        plt.close()
        
        
        # 2. node histogram
        good_nodes_count, bad_nodes_count = 0, 0
        for col in nodes_color:
                if col == color[0]: good_nodes_count += 1
                elif col == color[1]: bad_nodes_count += 1
        
        heights = [good_nodes_count, bad_nodes_count]
        edge_types = ("Cooperator", "Defector")
        
        bar_spacing = np.arange(len(edge_types))
        plt.bar(bar_spacing, heights, width=0.80, color=[color[0], color[1]])    # generate the histogram
        
        # setting attributes of bar graph
        plt.title("Node Type Distribution (iter={0})".format(index))
        plt.ylabel("Number of Nodes")
        plt.xlabel("Node Type")
        
        plt.xticks(bar_spacing, edge_types)
        
        if index != -1: plt.savefig(path_node_histogram + "/" +"node-" + repr(index) + ".png", format="PNG")      # output graph png 
        else: plt.show()
        plt.close()
        
        
        #enable method chaining
        return G        

'''
# INTERFACE: 
# if input is a list of node lists and list of adj matricies, 
# outputs graph given folder_name and generates gif
'''
def visualize_list(nodesDict_list, adjMatrix_list, iterations, model_name):
        print("Generating graphs...")
        # create directories and generate correct absolute path name
        path_network, path_node_histogram, path_edge_histogram, path_animation = creat_dir(model_name + " (network)"), creat_dir(model_name + " (node-histogram)"), creat_dir(model_name + " (edge-histogram)"), creat_dir("animation")
        
        # cleans directory containing the network and histogram visualizations
        for model_path in [path_network, path_node_histogram, path_edge_histogram]:
                for root, dirs, files in os.walk(model_path, onerror=lambda err: print("OSwalk error: " + repr(err))):
                        for file in files:
                                os.remove(os.path.join(root, file))
        
        # create graphs
        for i in tqdm(range(0, iterations + 1)):
                visualization(nodesDict_list[i], adjMatrix_list[i], path_network, path_node_histogram, path_edge_histogram, i)
        
        # compile PNGs into gif (for both network and histogram)
        generate_gif(model_name + " (network)", path_network, path_animation)
        generate_gif(model_name + " (edge-histogram)", path_edge_histogram, path_animation)
        generate_gif(model_name + " (node-histogram)", path_node_histogram, path_animation)


'''
func generate_gif
@param 
        input_path: directory path to image folder
        index (optional): current gif iteration (recommended if more than 1 gif is generated)
@return none
@output compiles all images by index into "animated.gif", and outputs gif into /animation
Note: only call when all graphs have been generated using func visualization(...)
'''
def generate_gif(model_name, input_path, output_path):
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
        
        with tqdm(total=sizecounter, unit='B', unit_scale=True, unit_divisor=1024) as pbar:
                with imageio.get_writer(os.path.join(output_path, model_name + "-animated.gif"), mode='I', duration=0.2) as writer:
                        for pic in only_PNGs:
                                image = imageio.imread(pic)
                                writer.append_data(image)
                                pbar.set_postfix(file=os.path.basename(pic), refresh=True)             # does NOT work on a POSIX system to get the base name from a Windows styled path
                                pbar.update(os.stat(pic).st_size)
        print("GIF Generated: stored in {0}".format(output_path))
        
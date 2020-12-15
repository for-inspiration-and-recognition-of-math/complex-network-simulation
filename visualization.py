'''
 ___________________
/\                  \  
\_|                 |  
  |  Read Section 3 |  
  |      for API    |  
  |                 |
  |  _______________|_ 
  \_/_______________/  
'''
import math
import random
import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import collections
import os
import sys
import imageio
import time
import concurrent.futures
from multiprocessing import Process, Pipe
import threading
import pickle
from statistics import mean 
from tqdm import tqdm


''' 
------------> ------------> ------------> ------------> ------------>
        Section 1: Global Parameters + Helper Functions 
------------> ------------> ------------> ------------> ------------>
'''

# global variables
color_rg = ['#03b500', '#b52a00', "#077d11", "#ffbb3d", "#db0231"] # (green-red color scheme)
# node colors [0,1] : green, red (orangish), 
# edge colors [2,3,4] : dark green, orange(leaning yellow), red (leaning magenta)
color = ['#0d009c', '#9e1500', "#68d7ed", "#f0c059", "#f27461"]         # alternate (blue-red) color scheme
# node colors [0,1] : royal blue (almost indigo), crimson, 
# edge colors [2,3,4] : teal (desaturated), orange(leaning yellow, less saturated), salmon

shape = ['8', '^']      # 8 = good, ^ = bad

good_good_edges_list, mixed_edges_list, bad_bad_edges_list = {}, {}, {}
# no these are not misnomers, they will eventually be sorted into lists

optimized_pos, position_set, done_flag = False, False, False            #setting position variables

# visualization options
network_histogram_flag = 1              
gif_flag = 1
line_scatter_plot_flag = 1
plot_3d_flag = 1

# helper function: create directory if not already exists
def creat_dir(folder_name, catagory="visualization"):
        dir_path = os.path.dirname(os.path.realpath(__file__))          # NOT os.getcwd() <——> this incantation is faulty
        path = "{directory}/{catagory}/{subdirectory}/".format(directory = dir_path, catagory=catagory, subdirectory=folder_name)
        mode = 0o755
        # output graph png
        try:  
                os.makedirs(path, mode)
        except OSError as error:  
                pass
        return path

def commandline_decorator(interface):
        def inner(*args, **kwargs):     # must have inner function to take and transfer the proper arguments
                print("\n----------------------------------")
                print("      Begin Visualization...")
                print("----------------------------------")
                
                interface(*args, **kwargs)
                
                dir_path = os.path.dirname(os.path.realpath(__file__))
                print("\n----------------------------------")
                print("     Visualization Complete!")
                print("----------------------------------\n")
                print(f'View Visualizations in: \n{dir_path}/visualization\n\n')
        return inner
  
class Dots:     # for user entertainment
        def __init__(self, num_dots=5):
                self.num_dots = num_dots
                self.done_flag = 0
        def __call__(self, status):
                if status == 0: self.start()
                elif status == 1: self.stop()
                else: print("Error: Invalid Dot Animation State", flush=True)
        
        def start(self):
                def begin_loading(num_dots):
                        self.done_flag = 0   # begins the animation
                        while True:
                                for i in range(num_dots+1):
                                        if self.done_flag:
                                                sys.stdout.write('\r' + "." * num_dots + "Done\n")       # draw all dots + "Done"
                                                sys.stdout.flush()
                                                return
                                        x = i % (num_dots+1)
                                        sys.stdout.write('\r' + "." * x )
                                        sys.stdout.write(" " * (num_dots - x))
                                        sys.stdout.flush()
                                        # time.sleep(0.1)
                t1 = threading.Thread(target=begin_loading, args=[self.num_dots])
                self.t1 = t1
                t1.start()
                
        def stop(self):
                self.done_flag = 1
                self.t1.join()
                


''' <------------ Section 1 END ------------> '''


''' 
------------> ------------> ------------> ------------> ------------> 
        Section 2: Visualization Generation Functions
------------> ------------> ------------> ------------> ------------>
'''

# '''
# func generate_png_csv (the alternate interface for 1 time visualizations)
# @param 
#         nodes: dictionary of nodes
#         adj: adjacency matrix (numpy 2D array)
#         abs_path: absolute path to the subdirectory in which all graphs will be stored (defaults to "graph")
#         index (optional): current simulation iteration (needed to generate png, else just oopens for 1 time viewing)
#         color_edges (true by default): colors edges depending on the nodes attached
# @return
#         none
# @usage
#         by passing in a dictionary of nodes along with its adjacency matrix
#         a visualization of the data will be generated
#         if an optional index is passed, a PNG named: "[index].png" will be generated 
#         in "graph" subdirectory of current working directory
#         E.g. index of 1 will generate "1.png" in CWD/graph
#         if an optional color_edges is passed, then edges will be colored with this rule:
#                 if both stubs connect to cooperating nodes, edge = dark green
#                 if both stubs connect to defecting nodes, edge = red
#                 if one stub connects to cooperating node and the other defecting node, edge = orange (leaning yellow)
# '''

def visualization(nodes, adj, optimized_pos, path_network, path_node_histogram, path_edge_histogram, index=-1, pos_lock = True, color_edges=True):
        global color, shape

        G = nx.convert_matrix.from_numpy_matrix(adj)
        if not pos_lock: optimized_pos = nx.spring_layout(G)    # generate new position for nodes if node positions are not locked

        good_nodes = [ i for i in range (len(nodes)) if (nodes[i].status == 0) ]
        bad_nodes = [ i for i in range (len(nodes)) if (nodes[i].status == 1) ]


        plt.figure(figsize = (10, 10))
        plt.title("Iteration={0}".format(index))
        
        
        # ---------------------------------------------------------
        # generating network visualization
        # ---------------------------------------------------------
        
        edge_width = 0.1
        edge_alpha = 0.6
        nx.draw(G, optimized_pos, with_labels=False, node_size = 0, width=edge_width)
        
        # custom nodes
        node_size = 35
        nx.draw_networkx_nodes(G, optimized_pos, nodelist = good_nodes, node_color=color[0], node_shape=shape[0], node_size = node_size)
        nx.draw_networkx_nodes(G, optimized_pos, nodelist = bad_nodes, node_color=color[1], node_shape=shape[1], node_size = node_size)

        # Add relationship-sensitive coloring to custom edges
        if color_edges:
                good_good_edges = []
                mixed_edges = []
                bad_bad_edges = []
                
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

        top = len(good_good_edges) + len(bad_bad_edges) + len(mixed_edges)
        if top <= 0: return index, good_good_edges, mixed_edges, bad_bad_edges            
                # since no more interactions (because no more edges), simulation is finished (just return empty lists)
        
        plt.ylim([0, top])
        plt.grid(True, axis='y')
        
        if index != -1: plt.savefig(path_edge_histogram + "edge-" + repr(index) + ".png", format="PNG")      # output graph png 
        else: plt.show()
        plt.close()
        
        
        # 2. node histogram
        heights = [len(good_nodes), len(bad_nodes)]
        edge_types = ("Cooperator", "Defector")
        
        bar_spacing = np.arange(len(edge_types))
        plt.bar(bar_spacing, heights, width=0.80, color=[color[0], color[1]])    # generate the histogram
        
        # setting attributes of bar graph
        plt.title("Node Type Distribution (iter={0})".format(index))
        plt.ylabel("Number of Nodes")
        plt.xlabel("Node Type")
        
        plt.xticks(bar_spacing, edge_types)
        plt.ylim([0, len(nodes)])
        plt.grid(True, axis='y')
        
        if index != -1: plt.savefig(path_node_histogram + "/" +"node-" + repr(index) + ".png", format="PNG")      # output graph png 
        else: plt.show()
        plt.close()
        
        
        # enable parallel concurrency
        return index, good_good_edges, mixed_edges, bad_bad_edges


# graph the status of all nodes over time (proportion normalized by total)
def plot_status_over_time(nodesDict_list, model_name, path_evolution, good_good_edges_list, mixed_edges_list, bad_bad_edges_list):
        print('\nPlotting node status evolution', flush=True)
        dots = Dots()
        dots.start()
        
        #--------------# 
        # 1. graph node type evolution 
        #--------------#
        
        num_defect = []
        num_nodes = len(nodesDict_list[0])
        for nodeDict in nodesDict_list:
                def_count = 0
                for nodes in nodeDict.values():
                        def_count += nodes.status         # assuming 1 represents defect, no multiedges, and nodes have binary status (0/1)
                num_defect.append(def_count/num_nodes)
        
        num_coop = [1 - def_count for def_count in num_defect]      # populate cooperating array with the 'complement' of num_defect
        
        # convert to np array
        iteration_count = [ i for i in range (0, len(nodesDict_list)) ]
        x = np.array(iteration_count)
        y1 = np.array(num_coop)    
        y2 = np.array(num_defect)    
        
        # plot with line of best fit
        global shape
        plt.figure(figsize=(25, 19), dpi=300, facecolor='w', edgecolor='k')
        plt.plot(x, y1, f'-{shape[0]}', color=color[0], label="total cooperators")
        plt.plot(x, y2, f'-{shape[1]}', color=color[1], label="total defectors")

        # plotting reference lines (.25, .5, .75)
        plt.plot([-1, len(iteration_count)], [0.5, 0.5], 'k-', lw=1,dashes=[2, 2])
        plt.plot([-1, len(iteration_count)], [0.25, 0.25], 'k-', lw=1,dashes=[2, 2])
        plt.plot([-1, len(iteration_count)], [0.75, 0.75], 'k-', lw=1,dashes=[2, 2])
        plt.ylim(0, 1)
        
        plt.xlabel('Iteration #')
        plt.ylabel('Proportion')
        plt.title("Proportion of Cooperators to Defectors over Iterations")
        # plt.text(0, 1,'matplotlib', horizontalalignment='center',
        #      verticalalignment='center'))
        plt.legend()
        plt.savefig(path_evolution + model_name + "-Evolution-Nodes.png", format="PNG")
        plt.close()
        dots.stop()
        
        
        #--------------# 
        # 2. graph edge type evolution 
        #--------------#
        
        print('Plotting edge type evolution', flush=True)
        dots.start()
        
        # get edge counts
        total = [ len(good_good_edges_list[i]) + len(mixed_edges_list[i]) + len(bad_bad_edges_list[i]) 
                 if len(good_good_edges_list[i]) + len(mixed_edges_list[i]) + len(bad_bad_edges_list[i]) > 0
                 else -1
                 for i in range (len(iteration_count)) ]
        
        good_edges_count = [ len(good_good_edges_list[i])/total[i] for i in range(len(iteration_count)) ]
        mixed_edges_count = [ len(mixed_edges_list[i])/total[i] for i in range(len(iteration_count)) ]
        bad_edge_count = [ len(bad_bad_edges_list[i])/total[i] for i in range(len(iteration_count)) ]

        y1 = np.array(good_edges_count)    
        y2 = np.array(mixed_edges_count)    
        y3 = np.array(bad_edge_count)    
        
        # plot with line of best fit
        plt.figure(figsize=(25, 19), dpi=300, facecolor='w', edgecolor='k')
        plt.plot(x, y1, '-o', color=color[2], label="total good-good edges")
        plt.plot(x, y2, '-o', color=color[3], label="total mixed edges")
        plt.plot(x, y3, '-o', color=color[4], label="total bad-bad edges")

        # plotting reference lines (.25, .5, .75)
        plt.plot([-1, len(iteration_count)], [0.5, 0.5], 'k-', lw=1,dashes=[2, 2])
        plt.plot([-1, len(iteration_count)], [0.25, 0.25], 'k-', lw=1,dashes=[2, 2])
        plt.plot([-1, len(iteration_count)], [0.75, 0.75], 'k-', lw=1,dashes=[2, 2])
        plt.ylim(0, 1)
        
        plt.xlabel('Iteration #')
        plt.ylabel('Proportion')
        plt.title("Proportion of Edge Types")

        plt.legend()
        plt.savefig(path_evolution + model_name + "--Evolution-Edges.png", format="PNG")
        plt.close()
        dots.stop()
        
        
        #--------------# 
        # 3. wealth vs avg state scatter plot 
        #--------------#
        
        print('Plotting Gains vs State', flush=True)
        dots.start()
        
        highest_wealth = 0
        num_nodes = len(nodesDict_list[0])
        num_iterations = len(nodesDict_list)
        last_index = num_iterations - 1       # -1 to exclude initial state
                        
        node_status_history = [ [] for i in range(num_nodes)]
        node_wealth_history = [ [] for i in range(num_nodes)]   # history of accumulation of wealth, not history of gain per iteration
        
        for nodeDict in nodesDict_list:
                for i in range(num_nodes):
                        node_status_history[i].append(nodeDict[i].status)       # gets status history
                        node_wealth_history[i].append(nodeDict[i].wealth)       # gets wealth history
                        highest_wealth = nodeDict[i].wealth if nodeDict[i].wealth > highest_wealth else highest_wealth  # gets max wealth 
        
        node_avg_status = [ mean(node_status) for node_status in node_status_history ]
        node_avg_wealth = [ node_wealth[last_index]/num_iterations for node_wealth in node_wealth_history ]

        highest_avg_wealth = max(node_avg_wealth)
        lowest_avg_wealth = min(node_avg_wealth)

        x = node_avg_status
        y1 = node_avg_wealth
        
        lower_bound = lowest_avg_wealth - abs(lowest_avg_wealth*0.05)
        upper_bound = highest_avg_wealth + abs(highest_avg_wealth*0.05)
        
        # plot with line of best fit
        plt.figure(figsize=(25, 19), dpi=300, facecolor='w', edgecolor='k', )
        plt.plot(x, y1, 'o', markersize=10, label="nodes")

        # plotting reference lines (using grid))
        plt.grid(True, axis='both', linestyle='-', linewidth=1)
        plt.ylim(lower_bound, upper_bound)
        
        plt.xlabel('Average Status (0-1)', fontsize=15)
        plt.ylabel('Average Gain per Iteration', fontsize=15)
        plt.title("Gain per iteration VS Average State for All Nodes (fitted scale)", fontsize=20)

        plt.savefig(path_evolution + model_name + "--Wealth-to-Status(1).png", format="PNG")
        
        ### generate another with fixed x-scale (better for comparison) ###
        plt.xlim(-0.05, 1.05)
        plt.plot([0.5, 0.5], [lower_bound, upper_bound], 'k-', lw=1,dashes=[2, 2])      # 50% reference line
        plt.title("Gain per iteration VS Average State for All Nodes (static scale)", fontsize=20)
        plt.savefig(path_evolution + model_name + "--Wealth-to-Status(2).png", format="PNG")
        
        plt.close()
        dots.stop()
        print()


def centrality_plot_3d(nodesDict_list, adjMatrix_list, model_name, path_evolution, measures_list):
        #--------------#
        # 4. graphing properties in analysis.py (centralities, mean geodesic distance, clustering coefficient) #
        #       Extension of above function
        #--------------#
        
        labels = ['degree', 'eigenvector', 'katz', 'closeness', 'betweeness', 'avg geodesic']
        results = [{}]
                
        num_iteration = len(nodesDict_list)
        num_nodes = len(nodesDict_list[0])
        
        # non-discriminant list storage
        node_status_history =  [ [] for i in range(num_nodes) ]
        node_degree_history =  [ [] for i in range(num_nodes) ]
        node_eigenvector_history =  [ [] for i in range(num_nodes) ]
        node_pagerank_history =  [ [] for i in range(num_nodes) ]
        node_local_clustering_history =  [ [] for i in range(num_nodes) ]
        
        # specific list storage
        good_node_degree_avg = []
        good_node_eigenvector_avg = []
        good_node_pagerank_avg = []
        good_node_local_clustering_avg = []
        
        bad_node_degree_avg = []
        bad_node_eigenvector_avg = []
        bad_node_pagerank_avg = []
        bad_node_local_clustering_avg = []
        
        path = creat_dir("dataframe", catagory="test")
        
        for i in range(num_iteration):
                measures_dataframe = measures_list[i]
                matrix = measures_dataframe.to_numpy()
        
                for node in matrix:
                        # building non-discriminant lists
                        node_status_history[i].append(node[2])
                        node_degree_history[i].append(node[3])
                        node_eigenvector_history[i].append(node[4])
                        node_pagerank_history[i].append(node[5])
                        node_local_clustering_history[i].append(node[6])
                        
                        # building specific lists
                        if node[2] == 0:
                                good_node_degree_avg
                        
                np.savetxt(f"{path}/some_output.txt", matrix, fmt='%s')
        
        # calculate specific lists for plotting
        for i in range (num_iteration):
                good_degree_sum, good_eigenvector_sum, good_pagerank_sum, good_local_clustering_sum = 0, 0, 0, 0
                bad_degree_sum, bad_eigenvector_sum, bad_pagerank_sum, bad_local_clustering_sum = 0, 0, 0, 0
                num_bad_nodes = sum(node_status_history[i])
                num_good_nodes = num_nodes - num_bad_nodes
                
                for j in range (num_nodes):
                        if node_status_history[i][j] == 0:
                                good_degree_sum += node_degree_history[i][j]
                                good_eigenvector_sum += node_eigenvector_history[i][j]
                                good_pagerank_sum += node_pagerank_history[i][j]
                                good_local_clustering_sum += node_local_clustering_history[i][j]
                        elif node_status_history[i][j] == 1:    # this explicit if statement is for logical clarity
                                bad_degree_sum += node_degree_history[i][j]
                                bad_eigenvector_sum += node_eigenvector_history[i][j]
                                bad_pagerank_sum += node_pagerank_history[i][j]
                                bad_local_clustering_sum += node_local_clustering_history[i][j]
                                
                good_node_degree_avg.append(good_degree_sum / num_good_nodes)
                good_node_eigenvector_avg.append(good_eigenvector_sum / num_good_nodes)
                good_node_pagerank_avg.append(good_pagerank_sum / num_good_nodes)
                good_node_local_clustering_avg.append(good_local_clustering_sum / num_good_nodes)
                
                bad_node_degree_avg.append(bad_degree_sum / num_bad_nodes)
                bad_node_eigenvector_avg.append(bad_eigenvector_sum / num_bad_nodes)
                bad_node_pagerank_avg.append(bad_pagerank_sum / num_bad_nodes)
                bad_node_local_clustering_avg.append(bad_local_clustering_sum / num_bad_nodes)
        

        x = [ x for x in range(num_iteration)]
        g1, g2, g3, g4 = good_node_degree_avg, good_node_eigenvector_avg, good_node_pagerank_avg, good_node_local_clustering_avg
        b1, b2, b3, b4 = bad_node_degree_avg, bad_node_eigenvector_avg, bad_node_pagerank_avg, bad_node_local_clustering_avg
        
        plt.figure(figsize=(20, 15), dpi=80, facecolor='w', edgecolor='k', linewidth=1)
        
        plt.plot(x, g1, '-o', color=color[0], markersize=10, label="good degree")
        plt.plot(x, g2, '--^', color=color[0], markersize=10, label="good eigenvector")
        plt.plot(x, g3, '-.x', color=color[0], markersize=10, label="good pagerank")
        plt.plot(x, g4, ':v', color=color[0], markersize=10, label="good local clustering")
        plt.plot(x, b1, '-o', color=color[1], markersize=10, label="bad degree")
        plt.plot(x, b2, '--^', color=color[1], markersize=10, label="bad eigenvector")
        plt.plot(x, b3, '-.x', color=color[1], markersize=10, label="bad pagerank")
        plt.plot(x, b4, ':v', color=color[1], markersize=10, label="bad local clustering")
        
        plt.xlabel('Iteration #', fontsize=15)
        plt.ylabel('Measurements', fontsize=15)
        plt.title("Analysis calculations over iterations", fontsize=20)

        plt.legend()
        plt.savefig(path_evolution + model_name + "--Analysis.png", format="PNG")
        plt.close()
        
        


# '''
# func generate_gif
# @param 
#         input_path: directory path to image folder
#         index (optional): current gif iteration (recommended if more than 1 gif is generated)
# @return none
# @output compiles all images by index into "animated.gif", and outputs gif into /animation
# Note: only call when all graphs have been generated using func visualization(...)
# '''
def generate_gif(model_name, input_path, output_path):
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


''' <------------ Section 2 END ------------> '''

        

''' 
------------> ------------> ------------> ------------> ------------>
                Section 3: User API
------------> ------------> ------------> ------------> ------------>
'''
# (i.e the only function you need read) 

'''INTERFACE (i.e. the only function you need to read): 
        if input is a list of node lists and list of adj matricies, 
        outputs graph given folder_name and generates gif
        5th parameter (model name) is for bookkeeping purposes
        6th parameter (defaulted to True) means position is LOCKED for future iteration 
        7th parameter (continuation=False) set to true if continuing a previous simulation using pickled data
        choose False to recalculate the position of Nodes every iteration (which significantly slows down the process)'''

@commandline_decorator    
def visualizer_spawner(nodesDict_list, adjMatrix_list, measures_list, iterations, model_name, pos_lock=True, continuation=False):
        global network_histogram_flag, gif_flag, line_scatter_plot_flag, plot_3d_flag
        
        # create directories and generate correct absolute path name
        path_network, path_node_histogram, path_edge_histogram, path_animation, path_evolution = creat_dir(model_name + " (network)"), creat_dir(model_name + " (node-histogram)"), creat_dir(model_name + " (edge-histogram)"), creat_dir("animation"), creat_dir("evolution")
        
        # cleans directory containing the network and histogram visualizations
        for model_path in [path_network, path_node_histogram, path_edge_histogram]:
                for root, dirs, files in os.walk(model_path, onerror=lambda err: print("OSwalk error: " + repr(err))):
                        for file in files:
                                os.remove(os.path.join(root, file))
                                
        # generate all network graphs + histogram
        if network_histogram_flag:

                print("\nSpawning demons...")                
                # ---------------------------------------------------------
                # generating graphs using multiple subprocesses 
                #       (use previous releases for non-concurrent version)
                # ---------------------------------------------------------
                
                # 1. Instantiate node positions of graphs before iterations for optimized position
                G = nx.convert_matrix.from_numpy_matrix(adjMatrix_list[0])
                global optimized_pos, position_set
                if pos_lock:
                        if not position_set:
                                optimized_pos = nx.spring_layout(G, threshold=1e-5, iterations=100)     # increased node distribution accuracy
                                position_set = True
                else: optimized_pos = nx.spring_layout(G)  
                # optimized_pos = nx.shell_layout(G) / nx.spiral_layout(G) / nx.spectral_layout(G)
                
                global good_good_edges_list, mixed_edges_list, bad_bad_edges_list
                runs = []

                def update_bar(pbar, total):
                        cur = 0
                        while len(runs) < total:
                                x_sync = len(runs)
                                pbar.update(x_sync - cur)
                                cur=x_sync
                        pbar.update(len(runs) - cur)

                
                with concurrent.futures.ProcessPoolExecutor() as executor:
                        pbar = tqdm(total = iterations+1, unit='graphs')
                        
                        t1 = threading.Thread(target=update_bar, args=[pbar, iterations+1])
                        t1.start()
                        
                        for i in range(0, iterations + 1):
                                f = executor.submit(visualization, nodesDict_list[i], adjMatrix_list[i], optimized_pos, 
                                                path_network, path_node_histogram, path_edge_histogram, i, pos_lock)
                                f.add_done_callback(lambda x: print(f'{x.result()[0]} ', end='', flush=True))
                                runs.append(f)

                        t1.join()
                        pbar.close()
                        print("all demons queued, waiting to complete...\n\nRunning graph generations... \n>", end = ' ', flush=True)
                        
                        
                        for run in concurrent.futures.as_completed(runs):
                                index = run.result()[0]
                                good_good_edges_list[index] = run.result()[1]
                                mixed_edges_list[index] = run.result()[2]
                                bad_bad_edges_list[index] = run.result()[3]
                        print("\n<--- all demons returned safely to tartarus --->")
                
                ### convert from dictionary into lists sorted by dictionary key ###

                good_good_edges_list = sorted(good_good_edges_list.items())     # coverts into sorted tuples
                good_good_edges_list = [ x[1] for x in good_good_edges_list ]    # converts into lists of lists (removes the index => tuple[0])
                
                mixed_edges_list = sorted(mixed_edges_list.items())     
                mixed_edges_list = [ x[1] for x in mixed_edges_list ] 
                
                bad_bad_edges_list = sorted(bad_bad_edges_list.items())     
                bad_bad_edges_list = [ x[1] for x in bad_bad_edges_list ] 
        
        
        # compile PNGs into gif (for both network and histogram)
        if gif_flag:
                print("\nCompiling GIF...")
                with concurrent.futures.ThreadPoolExecutor() as executor:
                        executor.submit(generate_gif, model_name + " (network)", path_network, path_animation)
                        executor.submit(generate_gif, model_name + " (edge-histogram)", path_edge_histogram, path_animation)
                        executor.submit(generate_gif, model_name + " (node-histogram)", path_node_histogram, path_animation)
        
        # generating graph over time for node status + edge type
        if line_scatter_plot_flag:
                plot_status_over_time(nodesDict_list, model_name, path_evolution, 
                                good_good_edges_list, mixed_edges_list, bad_bad_edges_list)
        
        # generating graph of all measures (3D)
        if plot_3d_flag:
                centrality_plot_3d(nodesDict_list, adjMatrix_list, model_name, path_evolution, measures_list)
                
        


''' <------------ Section 3 END ------------> '''



''' 
------------> ------------> ------------> ------------> ------------>
                Section 4: Meta API (testing)
------------> ------------> ------------> ------------> ------------>
'''
# the function to enable modular testing
# set the flags to activate specific visualizations to be generated

def visualize(*args, **kwargs):
        global network_histogram_flag, gif_flag, line_scatter_plot_flag, plot_3d_flag
        network_histogram_flag = 0
        gif_flag = 0
        line_scatter_plot_flag = 0
        plot_3d_flag = 1
        
        visualizer_spawner(*args, **kwargs)
        

''' <------------ Section 4 END ------------> '''
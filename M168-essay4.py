# -*- coding: utf-8 -*-

import collections
import networkx as nx
import matplotlib.pyplot as plt
import pandas as pd # won't need right away


def twitter():
        '''
        Reading in network data from an edge list
        '''

        file = open("circle_78813.txt", "r")  # saves the file
        lines = file.read().splitlines()    # Reads in the file line by line, saves each line as an element in a list

        # L.split('\t') splits line L every time it sees \t and stores each piece in a list. 
        # It returns a list with two elements, the source node and target node.
        sources = [int(L.split('\x20')[0]) for L in lines]
        targets = [int(L.split('\x20')[1]) for L in lines]

        # Make the network
        G = nx.Graph()
        E = len(sources)    # Number of edges
        print("num nodes= " + repr(E))
        for i in range(E):
                G.add_edge(sources[i], targets[i])

        # Draw the network. (for the kicks)
        # nx.draw(G, with_labels = False)
        # plt.show()

        # Better visualization. (Can make a bigger difference with larger networks.)
        pos = nx.spring_layout(G)   # Uses Fruchterman-Reingold algorithm to find "optimal" node positions
        plt.figure(figsize = (30, 30))  # Make figure bigger so we can actually see all the edges! (See what happens otherwise by calling nx.draw(G) before this line.)
        nx.draw(G, pos, node_size = 15, node_color = 'g', node_shape = '^', edge_color = '#00000022', with_labels = False)
        plt.show()


        file2 = open('twitter_combined.txt', 'r')
        lines = file2.read().splitlines()
        print ("total nodes combined = " + str(len(lines)))


        # plotting degree distribution histogram
        degree_sequence = sorted([d for n, d in G.degree()], reverse=True)  # degree sequence
        degreeCount = collections.Counter(degree_sequence)
        deg, cnt = zip(*degreeCount.items())

        fig, ax = plt.subplots()
        plt.bar(deg, cnt, width=0.80, color="#0084ffff")

        plt.title("Degree Distribution")
        plt.ylabel("Number of Nodes")
        plt.xlabel("Degree")
        ax.set_xticks([d + 0.4 for d in deg])
        ax.set_xticklabels(deg)




def student():
        '''
        Reading in network data from an adjacency matrix stored as csv file
        '''

        f = open("multigraph_hashAnonymized.csv")
        n_cols = len(f.readline().split(','))           # just to check how many columns there are
        df = pd.read_csv("multigraph_hashAnonymized.csv", skiprows = 1, usecols = range(0, n_cols), header = None)    # skips the top row in the file and only uses columns 1 though ncols (using 0-indicing). Tells pandas not to expect a header
        
        n_edges = len(df)
        print(df) # notice the column names are 1, ...., ncols-1
        
        sources = [df[0][i] for i in range (0, n_edges)]         # ahhhh, why does dataframes have to be column major!!!
        targets = [df[1][i] for i in range (0, n_edges)]         
        edge_type = [df[2][i] for i in range (0, n_edges)]         

        # add to graph
        G = nx.Graph()
        print("num nodes= " + repr(n_edges))
        for i in range(n_edges):
                G.add_edge(sources[i], targets[i])

        # Visualize the network
        pos = nx.spring_layout(G)
        plt.figure(figsize = (25, 25))  # Make figure bigger so we can actually see all the edges! (See what happens otherwise by calling nx.draw(G) before this line.)

        # How should we visualize the edge weights?
        # Access edge data and assign colors according to edge weight. (This is just one option for visualizing edge weights.)
        color = ['#CB3600AA', '#37AD52AA', '#4265B0AA'] # reddish-orange, green, blue, 
        i = 0
        for u, v in G.edges():  # Iterate through all edges (u, v).
                if edge_type[i] == "Computer":
                        G[u][v]['color'] = color[0]     # orange for computer
                elif edge_type[i] == "Time":
                        G[u][v]['color'] = color[1]     # green for time
                else:
                        G[u][v]['color'] = color[2]     # blue for partnership
                i+=1

        my_edge_colors = [G[u][v]['color'] for u, v in G.edges()]

        nx.draw(G, pos, node_size = 15, node_color = '#00000088', edge_color = my_edge_colors)
        plt.show()


        # plotting degree distribution histogram
        degree_sequence = sorted([d for n, d in G.degree()], reverse=True)  # degree sequence
        degreeCount = collections.Counter(degree_sequence)
        deg, cnt = zip(*degreeCount.items())

        fig, ax = plt.subplots()
        plt.bar(deg, cnt, width=0.80, color="#0084ffff")

        plt.title("Degree Distribution")
        plt.ylabel("Number of Nodes")
        plt.xlabel("Degree")
        ax.set_xticks([d + 0.4 for d in deg])
        ax.set_xticklabels(deg)

        # draw graph in inset
        plt.axes([0.4, 0.4, 0.5, 0.5])
        Gcc = G.subgraph(sorted(nx.connected_components(G), key=len, reverse=True)[0])
        pos = nx.spring_layout(G)
        plt.axis("off")
        nx.draw_networkx_nodes(G, pos, node_size=5, node_shape='v', node_color='#CE1FF4ff')
        nx.draw_networkx_edges(G, pos, alpha=0.4)
        plt.show()

if __name__ == '__main__':
        #twitter()
        student()





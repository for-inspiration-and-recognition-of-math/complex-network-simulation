import csv
import os
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import re
import random
from simulations import Node


def twitter_random_subgraph_clean( cooperator_proportion= 0.5, num_subgraphs =100, seed =100):
	directory = os.path.dirname(__file__) + '/resources' + '/twitter'
	os.chdir(directory)
	subgraph_names = os.listdir(directory)
	subgraph_set = set()
	for name in subgraph_names:
		subgraph_id = re.findall("^\d+", name)
		subgraph_set.add(subgraph_id[0])

	subgraph_set = list(subgraph_set)
	subgraph_set = [ name + '.edges' for name in subgraph_set]
	random.seed(seed)
	subgraph_random_set = random.sample(subgraph_set, num_subgraphs)
	combined = []
	for subgraph in subgraph_random_set:
		network = open(subgraph, 'r').read().splitlines()
		combined.extend(network)

	
	sources = [int(L.split()[0]) for L in combined]
	targets = [int(L.split()[1]) for L in combined]
	G = nx.Graph()
	E = len(sources)    # Number of edges
	for i in range(E):
		G.add_edge(sources[i], targets[i])

	node_dict = generate_node_dict(G.number_of_nodes(), cooperator_proportion, 100 )

	return node_dict, nx.to_numpy_array(G)


def facebook_clean(  cooperator_proportion = 0.5 ): #reads in an edgelist and returns the corresponding numpy adjacnecy matrix
	directory = os.path.dirname(__file__) + '/resources'
	os.chdir(directory)
	network1 = open('facebook_combined.txt', 'r')
	network1 = network1.read().splitlines()
	sources = [int(L.split()[0]) for L in network1]
	targets = [int(L.split()[1]) for L in network1]
	G = nx.Graph()
	E = len(sources)    # Number of edges
	for i in range(E):
		G.add_edge(sources[i], targets[i])
	node_dict = generate_node_dict(G.number_of_nodes(), cooperator_proportion, 100 )
	return node_dict, nx.to_numpy_array(G)

def karate_clean(  cooperator_proportion = 0.5 ): #reads in an edgelist and returns the corresponding numpy adjacnecy matrix
	directory = os.path.dirname(__file__) + '/resources'
	os.chdir(directory)
	network1 = open('soc-karate.txt', 'r')
	network1 = network1.read().splitlines()
	sources = [int(L.split()[0]) for L in network1]
	targets = [int(L.split()[1]) for L in network1]
	G = nx.Graph()
	E = len(sources)    # Number of edges
	for i in range(E):
		G.add_edge(sources[i], targets[i])
	node_dict = generate_node_dict(G.number_of_nodes(), cooperator_proportion, 100 )
	return node_dict, nx.to_numpy_array(G)





def generate_random_network(type, num_nodes , cooperator_proportion=0.5,  p =0.5, m =3, k =3, seed =100):
	#Enter type of graph and corresponding parameters and returns randomly generated graph's numpy adjacency matrix
	node_dict = generate_node_dict(num_nodes, cooperator_proportion, seed )

	if type == 'ER':
		graph = nx.erdos_renyi_graph(num_nodes, p, seed= seed)
		return node_dict, nx.to_numpy_array(graph)
	elif type == 'BA':
		graph = nx.barabasi_albert_graph(num_nodes, m, seed = seed)
		return node_dict, nx.to_numpy_array(graph)
	elif type == 'WS':
		graph = nx.watts_strogatz_graph(num_nodes, k, p, seed= seed)
		return node_dict, nx.to_numpy_array(graph)
	else:
		return "Enter a valid graph type : ER, BA, or WS"


def generate_node_dict(num_nodes,  cooperator_proportion = 0.5, seed =100 ):
	random.seed( seed)
	nodes = {}
	for i in range (0, num_nodes):
		unif = random.uniform(0, 1)
		if unif <= cooperator_proportion:
			nodes[i] = Node(0)
		else:
			nodes[i] = Node(1)
	return nodes

	


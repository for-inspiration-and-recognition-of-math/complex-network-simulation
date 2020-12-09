import networkx as nx 
import numpy as np
from cleaning import *
import pandas as pd

def centrality(adj , node_dict, type):
	G = nx.convert_matrix.from_numpy_matrix(adj)
	value_dict = {}
	return_dict = {}
	if type == 'degree':
		value_dict =  nx.degree_centrality(G)
	if type == 'eigenvector':
		value_dict =  nx.eigenvector_centrality(G)
	if type =='katz':
		value_dict = nx.katz_centrality(G)
	if type == 'closeness':
		value_dict =  nx.closeness_centrality(G)
	if type == 'betweenness':
		value_dict = nx.betweenness_centrality(G)
	for (index1, node), (index2, value) in zip(nodes.items(), value_dict.items()):
		return_dict[node] = value
	return return_dict


def all_measures( node_dict, adj, alpha = 0.9):
	G = nx.convert_matrix.from_numpy_matrix(adj)
	dict_list = []

	degree = nx.degree_centrality(G)
	df = pd.DataFrame.from_dict(degree, orient = 'index')
	df.columns = ['Degree']
	df['eigenvector'] = pd.Series(nx.eigenvector_centrality(G))
	df['katz'] = pd.Series(nx.katz_centrality(G))
	df['closeness'] = pd.Series(nx.closeness_centrality(G))
	df['betweenness'] = pd.Series(nx.betweenness_centrality(G))
	df['pagerank'] = pd.Series(nx.pagerank(G, alpha))
	df['local_clustering_coefficients'] = pd.Series(nx.clustering(G))

	return( df)


def geodesic(adj):
	G = nx.convert_matrix.from_numpy_matrix(adj)
	return nx.average_shortest_path_length(G)

def average_clustering_coefficient(adj):
	G = nx.convert_matrix.from_numpy_matrix(adj)
	return nx.average_clustering(G)


if __name__ == '__main__':
	model = 'BA'
	num_nodes = 15
	nodes, adj = generate_network(model, num_nodes, p=0.9)
	print(all_measures(nodes, adj))


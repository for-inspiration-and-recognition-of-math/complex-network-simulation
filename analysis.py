import networkx as nx 
import numpy as np
from cleaning import *
import pandas as pd
from networkx.algorithms.community import greedy_modularity_communities
import os 


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


def all_measures( node_dict, adj, iteration, alpha = 0.9):
	G = nx.convert_matrix.from_numpy_matrix(adj)

	df = pd.DataFrame.from_dict(node_dict, orient = 'index')
	df.columns = ['Node']
	df['iteration'] = iteration
	df['status'] = pd.Series([val.status for val in node_dict.values()])
	df['degree'] = pd.Series(nx.degree_centrality(G))
	df['eigenvector'] = pd.Series(nx.eigenvector_centrality(G))
	df['katz'] = pd.Series(nx.katz_centrality(G))
	df['closeness'] = pd.Series(nx.closeness_centrality(G))
	df['betweenness'] = pd.Series(nx.betweenness_centrality(G))
	df['pagerank'] = pd.Series(nx.pagerank(G, alpha))
	df['local_clustering_coefficients'] = pd.Series(nx.clustering(G))

	return( df)

def all_measures_master(node_dict_list, adj_list ):
	master_df = pd.DataFrame()
	it = 0
	for node_dict, adj in zip(node_dict_list, adj_list):
		values = all_measures(node_dict, adj, it, 0.9)
		master_df = pd.concat([master_df, values])
		it = it +1
	if not os.path.exists('analysis'):
		os.makedirs('analysis')
	directory = os.path.dirname(__file__) + '/analysis' + '/centrality_values.csv'
	master_df.to_csv(directory)
	return master_df

def community_detection(node_dict_list, adj_list ):
	master_df = pd.DataFrame()
	it = 0
	for node_dict, adj in zip(node_dict_list, adj_list):
		G = nx.from_numpy_matrix(adj)
		c = list(greedy_modularity_communities(G))
		print('iteration:' + str(it))
		for val in c:
			print(sorted(val))
			print([node_dict.get(item,item).status for item in sorted(val)])
			print()
		it = it +1
	return 


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


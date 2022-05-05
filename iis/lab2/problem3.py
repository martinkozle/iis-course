import time
from glob import glob

import networkx as nx
from sklearn.cluster import SpectralClustering
from sklearn.metrics import (adjusted_mutual_info_score,
                             normalized_mutual_info_score)


def get_circles(num_nodes):
    y_true = [0] * num_nodes
    circle_index = 1
    for file_name in glob('data/facebook/*.circles'):
        with open(file_name, 'r') as f:
            for line in f:
                _, *nodes = line.strip().split()
                for node in nodes:
                    y_true[int(node)] = circle_index
        circle_index += 1
    return y_true


def main():
    graph: nx.Graph = nx.read_edgelist('data/facebook_combined.txt')
    print(graph)
    # graph.remove_edges_from(test_positive_edges)
    start_time = time.perf_counter()
    adjacency_matrix = nx.to_numpy_array(graph)
    clustering = SpectralClustering(
        affinity='precomputed').fit(adjacency_matrix)
    y_true = get_circles(graph.number_of_nodes())
    adjusted_score = adjusted_mutual_info_score(y_true, clustering.labels_)
    normalized_score = normalized_mutual_info_score(y_true, clustering.labels_)
    end_time = time.perf_counter()
    print(f'{adjusted_score=}, {normalized_score=}')
    print(f'Finished in: {end_time - start_time}s')


if __name__ == '__main__':
    main()

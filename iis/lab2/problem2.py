import networkx as nx
import numpy as np
import seaborn as sns
import tqdm
from matplotlib import pyplot as plt
from sklearn.metrics import (average_precision_score, confusion_matrix,
                             roc_auc_score)

from ..utils.edge_sampling import (generate_negative_samples,
                                   generate_positive_samples)
from ..utils.random_walk import random_walk


def main():
    graph: nx.Graph = nx.read_edgelist('data/facebook_combined.txt')
    print(graph)
    edge_count = graph.number_of_edges()
    edge_count_20_percent = int(edge_count * 0.2)
    test_positive_edges = generate_positive_samples(
        graph, edge_count_20_percent)
    test_negative_edges = generate_negative_samples(
        graph, edge_count_20_percent)
    test_samples = test_positive_edges + test_negative_edges
    y_true = np.array([1] * len(test_positive_edges) +
                      [0] * len(test_negative_edges))
    # graph.remove_edges_from(test_positive_edges)
    density = nx.density(graph)
    print(f'{density=}')
    adjacency_matrix = nx.to_numpy_array(graph)
    similarities = []
    nodes_to_index = {node: i for i, node in enumerate(graph.nodes)}
    cache = {}
    for (i, j) in tqdm.tqdm(test_samples):
        ind_i = nodes_to_index[i]
        ind_j = nodes_to_index[j]
        if (ind_i, ind_j) not in cache:
            values = random_walk(adjacency_matrix, ind_i)
            cache.update({(ind_i, k): value for k, value in enumerate(values)})
        similarities.append(cache[ind_i, ind_j])
    # threshold given random samples from the graph as prior
    threshold = np.percentile(similarities, 100 - density * 100)
    # but we use 0.5 since the test prior is half positive half negative
    threshold_50 = np.percentile(similarities, 50)
    y_pred = [s >= threshold_50 for s in similarities]
    roc_auc = roc_auc_score(y_true, y_pred)
    average_precision = average_precision_score(y_true, y_pred)
    print(f'{roc_auc=}, {average_precision=}, {threshold=}')
    conf_matrix = confusion_matrix(y_true, y_pred)
    ax = sns.heatmap(conf_matrix, annot=True, fmt='d')
    ax.set_title('random_walk')
    ax.set_ylabel('True label')
    ax.set_xlabel('Predicted label')
    plt.savefig('plots/lab2/random_walk.png')
    plt.close()


if __name__ == '__main__':
    main()

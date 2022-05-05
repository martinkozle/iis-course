import networkx as nx
import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt
from sklearn.metrics import (average_precision_score, confusion_matrix,
                             roc_auc_score)

from ..utils.edge_sampling import (generate_negative_samples,
                                   generate_positive_samples)
from ..utils.similarity_measures import similarity


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
    for method in ('adamic_adar', 'jaccard', 'preferential_attachment'):
        similarities = []
        for (i, j) in test_samples:
            similarities.append(similarity(graph, i, j, method))
        # threshold given random samples from the graph as prior
        threshold = np.percentile(similarities, 100 - density * 100)
        # but we use 0.5 since the test prior is half positive half negative
        threshold_50 = np.percentile(similarities, 50)
        y_pred = [s >= threshold_50 for s in similarities]
        roc_auc = roc_auc_score(y_true, y_pred)
        average_precision = average_precision_score(y_true, y_pred)
        print(f'{method}: {roc_auc=}, {average_precision=}, {threshold=}')
        conf_matrix = confusion_matrix(y_true, y_pred)
        ax = sns.heatmap(conf_matrix, annot=True, fmt='d')
        ax.set_title(f'{method}')
        ax.set_ylabel('True label')
        ax.set_xlabel('Predicted label')
        plt.savefig(f'plots/lab2/{method}.png')
        plt.close()


if __name__ == '__main__':
    main()

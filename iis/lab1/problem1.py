from collections import Counter

import matplotlib.pyplot as plt
import networkx as nx
import seaborn as sns


def plot_histogram(data, file_path, **kwargs):
    _, ax = plt.subplots(figsize=(14, 8))
    sns.histplot(
        data,
        ax=ax,
        **kwargs
    )
    plt.savefig(file_path)


def main():
    graph: nx.Graph = nx.read_edgelist('data/cora/cora.cites')
    print(graph)
    print('Degree counts:', Counter(degree for _, degree in graph.degree))
    plot_histogram(
        [degree for _, degree in graph.degree],
        'plots/lab1/cora_degree_hist.png',
        bins=50,
        binrange=(0, 50)
    )
    components = list(nx.connected_components(graph))
    print('Number of connected components:', len(components))
    print(
        'Component sizes:',
        sorted(Counter(len(c) for c in components).items(), reverse=True)
    )
    print('Largest component size:', max(map(len, components)))
    largest_subgraph = graph.subgraph(max(components, key=len))
    print('Diameter of largest component subgraph:',
          nx.diameter(largest_subgraph))
    print('Average clustering coefficient:', nx.average_clustering(graph))

    random_graph = nx.gnp_random_graph(
        graph.number_of_nodes(),
        graph.number_of_edges() / graph.number_of_nodes()
    )
    small_world_graph = nx.watts_strogatz_graph(
        graph.number_of_nodes(),
        2,
        0.1
    )
    print('Random graph degree counts:',
          Counter(degree for _, degree in random_graph.degree))
    print('Random graph degree counts:',
          Counter(degree for _, degree in small_world_graph.degree))

    random_graph_components = list(nx.connected_components(random_graph))
    small_world_graph_components = list(
        nx.connected_components(small_world_graph))

    print(
        'Random graph component sizes:',
        sorted(Counter(len(c) for c in random_graph_components).items(),
               reverse=True)
    )
    print(
        'Small world graph component sizes:',
        sorted(Counter(len(c) for c in small_world_graph_components).items(),
               reverse=True)
    )

    print('Random graph diameter:', nx.diameter(random_graph))
    print(
        'Small world graph diameter:',
        nx.diameter(
            small_world_graph.subgraph(
                max(small_world_graph_components, key=len)
            )
        )
    )

    print('Random graph average clustering coefficient:',
          nx.average_clustering(random_graph))
    print('Small world graph average clustering coefficient:',
          nx.average_clustering(small_world_graph))


if __name__ == '__main__':
    main()

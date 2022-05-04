import os

import networkx as nx
import pandas as pd
from gem.embedding.sdne import SDNE
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OrdinalEncoder

from ..utils.evaluation import calculate_metrics
from ..utils.graph_embeddings import read_embeddings, save_embeddings


def read_graph():
    # Load the graph from edgelist
    edgelist = pd.read_table('data/cora/cora.cites',
                             header=None, names=['source', 'target'])
    edgelist['label'] = 'cites'
    graph = nx.from_pandas_edgelist(edgelist, edge_attr='label')
    nx.set_node_attributes(graph, 'paper', 'label')

    # Load the features and subject for the nodes
    feature_names = ['w_{}'.format(ii) for ii in range(1433)]
    column_names = feature_names + ['subject']
    node_data = pd.read_table('data/cora/cora.content',
                              header=None, names=column_names)

    return graph, node_data, feature_names


def get_embeddings(graph, embedding_path):
    if os.path.exists(embedding_path):
        return read_embeddings(embedding_path)
    sdne = SDNE(d=50, beta=5, alpha=1, nu1=1e-6, nu2=1e-6, K=3,
                n_units=[100, 50], n_iter=50, xeta=0.01, n_batch=500,
                modelfile=['data/cora/sdne_enc_model.json',
                           'data/cora/sdne_dec_model.json'],
                weightfile=['data/cora/sdne_enc_weights.hdf5',
                            'data/cora/sdne_dec_weights.hdf5'])
    embeddings, _ = sdne.learn_embedding(graph=graph, edge_f=None,
                                         is_weighted=False, no_python=True)
    save_embeddings(embedding_path, embeddings, graph.nodes())
    return embeddings


def main():
    graph, node_data, feature_names = read_graph()
    embeddings = get_embeddings(graph, 'data/cora/cora_sdne.emb')
    df_embeddings = pd.DataFrame(embeddings, index=graph.nodes())
    df_subject = pd.DataFrame(node_data['subject'], columns=['subject'],
                              index=graph.nodes())

    encoder = OrdinalEncoder()
    df_subject['subject_encoding'] = encoder.fit_transform(df_subject)

    x_train, x_test, y_train, y_test = train_test_split(
        df_embeddings, df_subject['subject_encoding'], test_size=0.1,
        stratify=df_subject['subject_encoding']
    )

    classifier = RandomForestClassifier(n_estimators=100, random_state=42)

    classifier.fit(x_train, y_train)
    y_pred = classifier.predict(x_test)
    calculate_metrics(y_test, y_pred)


if __name__ == '__main__':
    main()

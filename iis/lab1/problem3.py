import os

import networkx as nx
import numpy as np
import pandas as pd
from gem.embedding.sdne import SDNE
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer
from stellargraph.datasets import Cora
from stellargraph.layer import GraphSAGE
from stellargraph.mapper.sampled_node_generators import GraphSAGENodeGenerator
from tensorflow.keras.layers import Dense
from tensorflow.keras.losses import categorical_crossentropy
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam

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
    dataset = Cora()
    graph, node_subjects = dataset.load()

    train_subjects, test_subjects = train_test_split(
        node_subjects, test_size=0.2, stratify=node_subjects
    )

    val_subjects, test_subjects = train_test_split(
        test_subjects, test_size=0.5, stratify=test_subjects
    )

    label_binarizer = LabelBinarizer()
    train_targets = label_binarizer.fit_transform(train_subjects)
    val_targets = label_binarizer.transform(val_subjects)
    test_targets = label_binarizer.transform(test_subjects)

    generator = GraphSAGENodeGenerator(graph, 50, [10, 5])
    train_gen = generator.flow(train_subjects.index, train_targets,
                               shuffle=True)
    val_gen = generator.flow(val_subjects.index, val_targets, shuffle=True)
    test_gen = generator.flow(test_subjects.index, test_targets, shuffle=True)

    graphsage_model = GraphSAGE(
        layer_sizes=[32, 32],
        generator=generator,
        bias=True,
        dropout=0.5,
    )

    x_inp, x_out = graphsage_model.in_out_tensors()

    hidden_layer_1 = Dense(
        units=32, activation='relu',
    )(x_out)

    predictions = Dense(
        units=train_targets.shape[1], activation='softmax'
    )(hidden_layer_1)

    model = Model(inputs=x_inp, outputs=predictions)

    model.compile(optimizer=Adam(learning_rate=0.01),
                  loss=categorical_crossentropy,
                  metrics=['accuracy'])

    model.fit(train_gen, epochs=20, validation_data=val_gen, shuffle=False,
              verbose=2)

    model.evaluate(test_gen)

    pred_targets = model.predict(test_gen)

    calculate_metrics(np.argmax(test_targets, axis=1),
                      np.argmax(pred_targets, axis=1))

    # Conclusion: GraphSAGE gave worse results


if __name__ == '__main__':
    main()

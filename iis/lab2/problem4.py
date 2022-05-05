import time
from glob import glob

import networkx as nx
import pandas as pd
import stellargraph as sg
from sklearn.cluster import KMeans
from sklearn.metrics import (adjusted_mutual_info_score,
                             normalized_mutual_info_score)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer
from stellargraph.layer import GCN
from stellargraph.mapper.full_batch_generators import FullBatchNodeGenerator
from tensorflow.keras import Model, layers, losses, optimizers
from tensorflow.keras.callbacks import EarlyStopping


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
    y_true = get_circles(graph.number_of_nodes())
    node_features = pd.DataFrame(y_true, index=graph.nodes())
    train_subjects, test_subjects = train_test_split(
        node_features, train_size=140, test_size=None, stratify=node_features
    )
    val_subjects, test_subjects = train_test_split(
        test_subjects, train_size=500, test_size=None, stratify=test_subjects
    )
    target_encoding = LabelBinarizer()
    train_targets = target_encoding.fit_transform(train_subjects)
    val_targets = target_encoding.transform(val_subjects)
    test_targets = target_encoding.transform(test_subjects)
    sg_graph = sg.StellarGraph.from_networkx(
        graph, node_features=node_features)
    generator = FullBatchNodeGenerator(sg_graph, method='gcn')
    train_gen = generator.flow(train_subjects.index, train_targets)
    gcn = GCN(
        layer_sizes=[16, 16], activations=['relu', 'relu'],
        generator=generator, dropout=0.5
    )
    x_inp, x_out = gcn.in_out_tensors()
    predictions = layers.Dense(
        units=train_targets.shape[1], activation='softmax')(x_out)
    model = Model(inputs=x_inp, outputs=predictions)
    model.compile(
        optimizer=optimizers.Adam(learning_rate=0.01),
        loss=losses.categorical_crossentropy,
        metrics=['acc'],
    )
    val_gen = generator.flow(val_subjects.index, val_targets)
    es_callback = EarlyStopping(monitor='val_acc', patience=50,
                                restore_best_weights=True)
    history = model.fit(
        train_gen,
        epochs=200,
        validation_data=val_gen,
        verbose=2,
        shuffle=False,
        callbacks=[es_callback],
    )
    sg.utils.plot_history(history)
    test_gen = generator.flow(test_subjects.index, test_targets)
    test_metrics = model.evaluate(test_gen)
    print('\nTest Set Metrics:')
    for name, val in zip(model.metrics_names, test_metrics):
        print('\t{}: {:0.4f}'.format(name, val))
    all_nodes = node_features.index
    all_gen = generator.flow(all_nodes)
    embedding_model = Model(
        inputs=model.input, outputs=model.layers[-2].output)
    all_embeddings = embedding_model.predict(all_gen)[0]
    kmeans_model = KMeans(n_clusters=len(target_encoding.classes_))
    km_predictions = kmeans_model.fit_predict(all_embeddings)
    adjusted_score = adjusted_mutual_info_score(
        node_features.values.flatten(), km_predictions)
    normalized_score = normalized_mutual_info_score(
        node_features.values.flatten(), km_predictions)
    end_time = time.perf_counter()
    print(f'{adjusted_score=}, {normalized_score=}')
    print(f'Finished in: {end_time - start_time}s')


if __name__ == '__main__':
    main()

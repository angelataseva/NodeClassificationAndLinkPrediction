import json
import random
from functools import partial

import networkx as nx
import stellargraph as sg
import pandas as pd
import numpy as np
from scipy.sparse import spdiags
from stellargraph.data import EdgeSplitter


def read_graph(nodes, edges, features, social_network):
    g = nx.Graph()
    features_file = open(features)
    features_dict = json.load(features_file)
    length = int(sum([len(value) for value in features_dict.values()]) / len(features_dict))

    with open(nodes, 'r', encoding='utf-8') as file:
        next(file)
        for i, line in enumerate(file.readlines()):
            line = line.rstrip("\r\n")
            line_list = line.split(',')
            id = line_list[0]
            if social_network == 'facebook':
                name = ''.join(line_list[2:-1])
                type = line_list[-1]
            else:
                name = line_list[1]
                type = line_list[2]
            feature = []
            if features_dict[id] is not None:
                feature = features_dict[id]
                if len(feature) < length:
                    feature += [0] * (length - len(feature))
                else:
                    feature = feature[:length]
            g.add_node(id, type=type, name=name, feature=feature)

    with open(edges, 'r', encoding='utf-8') as file:
        next(file)
        for i, line in enumerate(file.readlines()):
            line = line.rstrip("\r\n")
            node1 = line.split(',')[0]
            node2 = line.split(',')[1]
            if node1 in g.nodes() and node2 in g.nodes():
                g.add_edge(node1, node2)

    return g


def create_samples(g):
    G = sg.StellarGraph.from_networkx(g, node_features="feature")
    edge_splitter_test = EdgeSplitter(G)
    G_test, edge_ids_test, edge_labels_test = edge_splitter_test.train_test_split(
        p=0.1, method="global", keep_connected=True)

    id_1, id_2 = [], []
    for edge in edge_ids_test:
        id_1.append(edge[0])
        id_2.append(edge[1])
    link = list(edge_labels_test)

    num = int(nx.number_of_edges(g) * 0.1)
    positive_sample = edge_ids_test[:num]
    # negative_sample = edge_ids_test[num:]

    for edge in positive_sample:
        g.remove_edge(edge[0], edge[1])

    df = pd.DataFrame({'id_1': id_1, 'id_2': id_2, 'link': link})
    # df.to_csv('nodes.csv', encoding='utf-8')

    return g, df


def create_test_data(list):
    y_test = []
    for u, v, p in list:
        y_test.append(p)
    return y_test


def create_index_of_nodes(nodes):
    dict = {}
    for i, node in enumerate(nodes):
        dict.setdefault(node, i)
    return dict


def rwr(x, T, R=0.2, max_iters=100):
    '''
    This function will perform the random walk with restart algorithm on a given vector x and the associated
    transition matrix of the network

    args:
        x (Array) : Initial vector
        T (Matrix) : Input matrix of transition probabilities
        R (Float) : Restart probabilities
        max_iters (Integer) : The maximum number of iterations

    returns:
        This function will return the result vector x
    '''

    old_x = x
    err = 1.

    for i in range(max_iters):
        x = (1 - R) * (T.dot(old_x)) + (R * x)
        err = np.linalg.norm(x - old_x, 1)
        if err <= 1e-6:
            break
        old_x = x
    return x


def run_rwr(g, R, max_iters):
    '''
    This function will run the `rwr` on a network

    args:
        g (Network) : This is a networkx network you want to run rwr on
        R (Float) : The restart probability
        max_iters (Integer) : The maximum number of iterations

    returns:
        This fuunction will return a numpy array of affinities where each element in the array will represent
        the similarity between two nodes
    '''

    A = nx.adjacency_matrix(g, weight='weight')
    m, n = A.shape

    d = A.sum(axis=1)
    d = np.asarray(d).flatten()
    d = np.maximum(d, np.ones(n))

    invd = spdiags(1.0 / d, 0, m, n)
    T = invd.dot(A)

    rwr_fn = partial(rwr, T=T, R=R, max_iters=max_iters)

    aff = [rwr_fn(x) for x in np.identity(m)]
    aff = np.array(aff)
    return aff
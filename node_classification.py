import numpy as np
from gensim.models import Word2Vec
from node2vec import Node2Vec
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from tqdm import tqdm

from helper import read_graph, get_randomwalk


def convert_labels(labels, type):
    numbered_labels = []
    if type == 'Facebook':
        for label in labels:
            if label == 'company':
                numbered_labels.append(0)
            elif label == 'tvshow':
                numbered_labels.append(1)
            elif label == 'government':
                numbered_labels.append(2)
            else:
                numbered_labels.append(3)
    else:
        numbered_labels = [int(label) for label in labels]
    return numbered_labels


def print_results(x_train, x_test, y_train, y_test):
    random_forest_classifier = RandomForestClassifier(n_estimators=1000, random_state=0)
    random_forest_classifier.fit(x_train, y_train)
    y_predicted = random_forest_classifier.predict(x_test)
    print(f"Random Forest Classifier accuracy: ", accuracy_score(y_test, y_predicted))
    print(f"Random Forest Classifier precision: ", precision_score(y_test, y_predicted, average='macro'))
    print(f"Random Forest Classifier recall score: ", recall_score(y_test, y_predicted, average='macro'))
    print(f"Random Forest Classifier F1 score: ", f1_score(y_test, y_predicted, average='macro'))

    clf = SVC()
    clf.fit(x_train, y_train)
    y_predicted = clf.predict(x_test)
    print(f"SVM accuracy: ", accuracy_score(y_test, y_predicted))
    print(f"SVM precision: ", precision_score(y_test, y_predicted, average='macro'))
    print(f"SVM recall score: ", recall_score(y_test, y_predicted, average='macro'))
    print(f"SVM F1 score: ", f1_score(y_test, y_predicted, average='macro'))

    clf_knn = KNeighborsClassifier(n_neighbors=3)
    clf_knn.fit(x_train, y_train)
    y_predicted = clf_knn.predict(x_test)
    print(f"KNN accuracy: ", accuracy_score(y_test, y_predicted))
    print(f"KNN  precision: ", precision_score(y_test, y_predicted, average='macro'))
    print(f"KNN  recall score: ", recall_score(y_test, y_predicted, average='macro'))
    print(f"KNN  F1 score: ", f1_score(y_test, y_predicted, average='macro'))

    lr = LogisticRegression(penalty='l2', max_iter=500, C=1, random_state=42)
    lr.fit(x_train, y_train)
    y_predicted = lr.predict(x_test)
    print(f"Logistic Regression accuracy: ", accuracy_score(y_test, y_predicted))
    print(f"Logistic Regression  precision: ", precision_score(y_test, y_predicted, average='macro'))
    print(f"Logistic Regression  recall score: ", recall_score(y_test, y_predicted, average='macro'))
    print(f"Logistic Regression  F1 score: ", f1_score(y_test, y_predicted, average='macro'))


def classification_with_pretrained_vectors(graph, type):
    data, labels = [], []
    for node in graph.nodes:
        data.append(list(graph.nodes[node]['feature']))
        labels.append(graph.nodes[node]['type'])

    labels = convert_labels(labels, type)
    x_train, x_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, random_state=36, shuffle=True)

    print(f"Classification results for {type} dataset with pretrained vectors:")
    print_results(x_train, x_test, y_train, y_test)


def classification_with_node2vec(graph, type):
    n2v_obj = Node2Vec(graph, dimensions=10, walk_length=5, num_walks=15, p=1, q=1, workers=1)
    n2v_model = n2v_obj.fit()
    data = [list(n2v_model.wv.get_vector(n)) for n in graph.nodes]
    data = np.array(data)

    labels = [graph.nodes[node]['type'] for node in graph.nodes]
    labels = convert_labels(labels, type)

    x_train, x_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, random_state=36, shuffle=True)

    print(f"Classification results for {type} dataset with Nodе2Vec vectors:")
    print_results(x_train, x_test, y_train, y_test)


def classification_with_deepwalk(graph, type):
    all_nodes = list(graph.nodes())

    random_walks = []
    for n in tqdm(all_nodes):
        for i in range(5):
            random_walks.append(get_randomwalk(graph, n, 15))

    model = Word2Vec(window=4, sg=1, hs=0, negative=10, alpha=0.03, min_alpha=0.0007, seed=14)
    model.build_vocab(random_walks, progress_per=2)
    model.train(random_walks, total_examples=model.corpus_count, epochs=20, report_delay=1)

    vectors, labels = [], []
    for i, node in enumerate(model.wv.index2word):
        vectors.append(model.wv.vectors[i])
        labels.append(graph.nodes[node]['type'])
    labels = convert_labels(labels, type)

    x_train, x_test, y_train, y_test = train_test_split(vectors, labels, test_size=0.2, random_state=36, shuffle=True)

    print(f"Classification results for {type} dataset with DeepWalk vectors:")
    print_results(x_train, x_test, y_train, y_test)


if __name__ == '__main__':
    nodes_file = "data/facebook_large/musae_facebook_target.csv"
    edges_file = "data/facebook_large/musae_facebook_edges.csv"
    features_file = 'data/facebook_large/musae_facebook_features.json'
    facebook = read_graph(nodes_file, edges_file, features_file, 'facebook')
    classification_with_pretrained_vectors(facebook, 'Facebook')
    classification_with_node2vec(facebook, 'Facebook')
    classification_with_deepwalk(facebook, 'Facebook')

    nodes_file = "data/git_web_ml/musae_git_target.csv"
    edges_file = "data/git_web_ml/musae_git_edges.csv"
    features_file = 'data/git_web_ml/musae_git_features.json'
    git = read_graph(nodes_file, edges_file, features_file, 'git')
    classification_with_pretrained_vectors(git, 'GitHub')
    classification_with_node2vec(git, 'GitHub')
    classification_with_deepwalk(git, 'GitHub')

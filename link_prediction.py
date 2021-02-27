import networkx as nx
from sklearn.metrics import average_precision_score, roc_auc_score

from helper import read_graph, create_samples, create_test_data, run_rwr


def link_prediction_with_metrics(subgraph, tuples, df):
    jaccard_coefficient_list = list(nx.jaccard_coefficient(subgraph, tuples))
    y_test = create_test_data(jaccard_coefficient_list)
    print(f"ROC AUC Score with Jaccard Coefficient: {roc_auc_score(df['link'], y_test)}\n"
          f"Average Precision with Jaccard Coefficient: {average_precision_score(df['link'], y_test)}")

    adamic_adar_list = list(nx.adamic_adar_index(subgraph, tuples))
    y_test = create_test_data(adamic_adar_list)
    print(f"ROC AUC Score with Adamic Adar Index: {roc_auc_score(df['link'], y_test)}\n"
          f"Average Precision with Adamic Adar Index: {average_precision_score(df['link'], y_test)}")

    preferential_attachment_list = list(nx.preferential_attachment(subgraph, tuples))
    y_test = create_test_data(preferential_attachment_list)
    print(f"ROC AUC Score with Preferential Attachment: {roc_auc_score(df['link'], y_test)}\n"
          f"Average Precision with Preferential Attachment: {average_precision_score(df['link'], y_test)}")

    resource_allocation_list = list(nx.resource_allocation_index(subgraph, tuples))
    y_test = create_test_data(resource_allocation_list)
    print(f"ROC AUC Score with Resource Allocation Index: {roc_auc_score(df['link'], y_test)}\n"
          f"Average Precision with Resource Allocation Index: {average_precision_score(df['link'], y_test)}")


def link_prediction_with_random_walk(subgraph, node_pairs, df):
    aff = run_rwr(subgraph, 0.2, 1000)
    y_test = []
    for node1, node2 in node_pairs:
        y_test.append(aff[int(node1)][int(node2)])

    print(f"ROC AUC Score with Random Walk: {roc_auc_score(df['link'], y_test)}\n"
          f"Average Precision with Random Walk: {average_precision_score(df['link'], y_test)}")


if __name__ == '__main__':
    nodes_file = "data/facebook_large/musae_facebook_target.csv"
    edges_file = "data/facebook_large/musae_facebook_edges.csv"
    features_file = 'data/facebook_large/musae_facebook_features.json'
    facebook_graph = read_graph(nodes_file, edges_file, features_file, 'facebook')

    subgraph, df = create_samples(facebook_graph)
    node_pairs = [tuple(x) for x in df[['id_1', 'id_2']].to_numpy()]

    link_prediction_with_metrics(subgraph, node_pairs, df)
    link_prediction_with_random_walk(subgraph, node_pairs, df)

    nodes_file = "data/git_web_ml/musae_git_target.csv"
    edges_file = "data/git_web_ml/musae_git_edges.csv"
    features_file = 'data/git_web_ml/musae_git_features.json'
    git_graph = read_graph(nodes_file, edges_file, features_file, 'git')

    subgraph, df = create_samples(git_graph)
    node_pairs = [tuple(x) for x in df[['id_1', 'id_2']].to_numpy()]

    link_prediction_with_metrics(subgraph, node_pairs, df)
    link_prediction_with_random_walk(subgraph, node_pairs, df)

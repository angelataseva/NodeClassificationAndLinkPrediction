from random import sample

import matplotlib.pyplot as plt
import networkx as nx

from helper import read_graph


def draw(facebook, git):
    nodes = sorted(list(nx.connected_components(nx.subgraph(facebook, sample(facebook.nodes, 500)))),
                   key=lambda x: len(x), reverse=True)[0]
    subgraph_facebook = nx.subgraph(facebook, nodes).copy()

    nodes = sorted(list(nx.connected_components(nx.subgraph(git, sample(git.nodes, 500)))),
                   key=lambda x: len(x), reverse=True)[0]
    subgraph_git = nx.subgraph(git, nodes).copy()

    pos = nx.spring_layout(subgraph_facebook)
    nx.draw(subgraph_facebook, pos, node_color='#A0CBE2', edge_color='#00bb5e', width=1, with_labels=True)
    plt.savefig("plots/facebook_subgraph.png")

    pos = nx.spring_layout(subgraph_git)
    nx.draw(subgraph_git, pos, node_color='#A0CBE2', edge_color='#00bb5e', width=1, with_labels=True)
    plt.savefig("plots/git_subgraph.png")


def draw_class_distribution(facebook, git):
    facebook_groups = ['company', 'politician', 'tvshow', 'government']
    facebook_counts = [0, 0, 0, 0]
    for node in facebook.nodes():
        if facebook.nodes[node]['type'] == 'company':
            facebook_counts[0] += 1
        elif facebook.nodes[node]['type'] == 'politician':
            facebook_counts[1] += 1
        elif facebook.nodes[node]['type'] == 'tvshow':
            facebook_counts[2] += 1
        else:
            facebook_counts[3] += 1

    github_groups = ['web', 'ml']
    github_counts = [0, 0]
    for node in git.nodes():
        if git.nodes[node]['type'] == '0':
            github_counts[0] += 1
        else:
            github_counts[1] += 1

    plt.figure(figsize=(15, 5))

    plt.subplot(121)
    plt.bar(facebook_groups, facebook_counts, align='center', alpha=0.3, color=['blue', 'red', 'yellow', 'green'])
    plt.xlabel('Classes')
    plt.ylabel('Nodes')
    plt.title('Facebook Nodes Distribution')

    plt.subplot(122)
    plt.bar(github_groups, github_counts, align='center', alpha=0.3, color=['blue', 'red'])
    plt.xlabel('Classes')
    plt.ylabel('Nodes')
    plt.title('GitHub Nodes Distribution')

    plt.savefig('plots/nodes_distribution.png')


def get_nodes_with_biggest_degree(facebook, git):
    facebook_degree = nx.degree(facebook)
    git_degree = nx.degree(git)
    nodes_facebook = sorted(facebook_degree, key=lambda x: x[1], reverse=True)[:10]
    nodes_git = sorted(git_degree, key=lambda x: x[1], reverse=True)[:10]
    pages, developers = '', ''
    for i in range(9):
        id = nodes_facebook[i][0]
        pages += facebook.nodes[id]['name'] + ', '
        id = nodes_git[i][0]
        developers += git.nodes[id]['name'] + ', '
    id = nodes_facebook[9][0]
    pages += facebook.nodes[id]['name']
    id = nodes_git[9][0]
    developers += git.nodes[id]['name']
    return pages, developers


def get_nodes_with_biggest_eigenvector(facebook, git):
    eigenvector_facebook = nx.eigenvector_centrality(facebook)
    eigenvector_git = nx.eigenvector_centrality(git)
    nodes_facebook = sorted(eigenvector_facebook.items(), key=lambda x: x[1], reverse=True)[:10]
    nodes_git = sorted(eigenvector_git.items(), key=lambda x: x[1], reverse=True)[:10]
    pages, developers = '', ''
    for i in range(9):
        id = nodes_facebook[i][0]
        pages += facebook.nodes[id]['name'] + ', '
        id = nodes_git[i][0]
        developers += git.nodes[id]['name'] + ', '
    id = nodes_facebook[9][0]
    pages += facebook.nodes[id]['name']
    id = nodes_git[9][0]
    developers += git.nodes[id]['name']
    return pages, developers


def get_nodes_with_biggest_betweenness(facebook, git):
    betweenness_facebook = nx.betweenness_centrality(facebook)
    betweenness_git = nx.betweenness_centrality(git)
    nodes_facebook = sorted(betweenness_facebook.items(), key=lambda x: x[1], reverse=True)[:10]
    nodes_git = sorted(betweenness_git.items(), key=lambda x: x[1], reverse=True)[:10]
    pages, developers = '', ''
    for i in range(9):
        id = nodes_facebook[i][0]
        pages += facebook.nodes[id]['name'] + ', '
        id = nodes_git[i][0]
        developers += git.nodes[id]['name'] + ', '
    id = nodes_facebook[9][0]
    pages += facebook.nodes[id]['name']
    id = nodes_git[9][0]
    developers += git.nodes[id]['name']
    return pages, developers


if __name__ == '__main__':
    nodes_file = "data/facebook_large/musae_facebook_target.csv"
    edges_file = "data/facebook_large/musae_facebook_edges.csv"
    features_file = 'data/facebook_large/musae_facebook_features.json'
    facebook = read_graph(nodes_file, edges_file, features_file, 'facebook')

    nodes_file = "data/git_web_ml/musae_git_target.csv"
    edges_file = "data/git_web_ml/musae_git_edges.csv"
    features_file = 'data/git_web_ml/musae_git_features.json'
    git = read_graph(nodes_file, edges_file, features_file, 'git')

    draw(facebook, git)
    draw_class_distribution(facebook, git)

    print(f"Facebook Graph info:\n{nx.info(facebook)}")
    print(f"Git Graph info\n{nx.info(git)}")

    print(f"No. of unique Facebook pages: {facebook.number_of_nodes()}")
    print(f"No. of unique GitHub users: {git.number_of_nodes()}")

    print(f"No. of communities in Facebook network: {nx.number_connected_components(facebook)}")
    print(f"No. of communities in GitHub network: {nx.number_connected_components(git)}")

    facebook_density = nx.density(facebook)
    git_density = nx.density(git)
    print("Facebook network density:", facebook_density)
    print("GitHub network density:", git_density)

    facebook_diameter = nx.diameter(facebook)
    git_diameter = nx.diameter(git)
    print(f"Diameter of Facebook Network is: {facebook_diameter}")
    print(f"Diameter of GitHub Network is: {git_diameter}")

    facebook_triadic_closure = nx.transitivity(facebook)
    git_triadic_closure = nx.transitivity(git)
    print(f"Triadic closure of Facebook Network: {facebook_triadic_closure}")
    print(f"Triadic closure of GitHub Network: {git_triadic_closure}")

    pages, developers = get_nodes_with_biggest_degree(facebook, git)
    print(f"Pages with biggest degree in Facebook are: {pages}")
    print(f"Developers with biggest degree on GitHub are: {developers}")

    pages, developers = get_nodes_with_biggest_eigenvector(facebook, git)
    print(f"Pages with biggest eigenvector centrality on Facebook are: {pages}")
    print(f"Developers with biggest eigenvector centrality on GitHub: {developers}")

    pages, developers = get_nodes_with_biggest_betweenness(facebook, git)
    print(f"Pages with biggest betweenness centrality on Facebook are: {pages}")
    print(f"Developers with biggest betweenness centrality on GitHub: {developers}")

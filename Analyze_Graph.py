import networkx as nx
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from community import community_louvain
from collections import Counter
from tqdm.notebook import tqdm
import os

def load_graph(file_path):
    """Load the directed graph from a GraphML file."""
    return nx.read_graphml(file_path, node_type=str)

def basic_graph_statistics(G):
    """Calculate and print basic graph statistics for a directed graph."""
    print("Basic Graph Statistics:")
    print(f"Number of nodes: {G.number_of_nodes()}")
    print(f"Number of edges: {G.number_of_edges()}")
    print(f"Graph density: {nx.density(G):.4f}")
    print(f"Is strongly connected: {nx.is_strongly_connected(G)}")
    print(f"Number of strongly connected components: {nx.number_strongly_connected_components(G)}")

def community_detection(G):
    """Perform community detection using the Louvain method."""
    partition = community_louvain.best_partition(G.to_undirected())

    print("\nCommunity Detection:")
    print(f"Number of communities: {len(set(partition.values()))}")

    community_sizes = Counter(partition.values())
    print(f"Largest community size: {max(community_sizes.values())}")
    print(f"Smallest community size: {min(community_sizes.values())}")

    return partition

def centrality_measures(G, output_folder):
    """Calculate various centrality measures for a directed graph, weighted by node frequency."""
    print("\nCentrality Measures:")

    # Assign node frequency (this would be defined based on your data)
    # Here, a placeholder value is provided; replace it with the actual frequency data.
    print("Assigning node frequency values...")
    for node in G.nodes():
        G.nodes[node]['frequency'] = np.random.randint(1, 10)  # Placeholder: Assign a random frequency to each node

    print("Calculating in-degree and out-degree centrality (weighted by node frequency)...")
    in_degree_cent = nx.in_degree_centrality(G)
    out_degree_cent = nx.out_degree_centrality(G)

    # Weight in-degree and out-degree centrality by node frequency
    for node in G.nodes():
        frequency = G.nodes[node]['frequency']
        in_degree_cent[node] *= frequency
        out_degree_cent[node] *= frequency

    print("Calculating betweenness centrality (weighted by node frequency)...")
    betweenness_cent = nx.betweenness_centrality(G)

    # Weight betweenness centrality by node frequency
    for node in G.nodes():
        frequency = G.nodes[node]['frequency']
        betweenness_cent[node] *= frequency

    print("Calculating eigenvector centrality (weighted by node frequency)...")
    eigenvector_cent = nx.eigenvector_centrality(G, max_iter=1000)

    # Weight eigenvector centrality by node frequency
    for node in G.nodes():
        frequency = G.nodes[node]['frequency']
        eigenvector_cent[node] *= frequency

    # Create DataFrame to store the centrality measures
    centrality_df = pd.DataFrame({
        'Node': list(G.nodes()),
        'Frequency': [G.nodes[n]['frequency'] for n in G.nodes()],
        'In-Degree Centrality': [in_degree_cent[n] for n in G.nodes()],
        'Out-Degree Centrality': [out_degree_cent[n] for n in G.nodes()],
        'Betweenness Centrality': [betweenness_cent[n] for n in G.nodes()],
        'Eigenvector Centrality': [eigenvector_cent[n] for n in G.nodes()]
    })

    # Sort by in-degree centrality for easier interpretation of top nodes
    centrality_df = centrality_df.sort_values('In-Degree Centrality', ascending=False)

    # Print top 10 nodes by different centrality metrics
    print("\nTop 10 nodes by In-Degree Centrality:")
    print(centrality_df[['Node', 'In-Degree Centrality']].head(10).to_string(index=False))

    print("\nTop 10 nodes by Out-Degree Centrality:")
    print(centrality_df[['Node', 'Out-Degree Centrality']].head(10).to_string(index=False))

    centrality_df.to_csv(os.path.join(output_folder, 'Centrality_Measures_Weighted.csv'), index=False)
    print("\nFull centrality measures saved to 'Centrality_Measures_Weighted.csv'")

def analyze_graph(input_graph_path, output_folder):
    """Main function to analyze the directed graph."""
    os.makedirs(output_folder, exist_ok=True)

    print("Loading graph...")
    G = load_graph(input_graph_path)

    basic_graph_statistics(G)
    partition = community_detection(G)
    centrality_measures(G, output_folder)

if __name__ == "__main__":
    input_graph_path = ""
    output_folder = ""

    analyze_graph(input_graph_path, output_folder)

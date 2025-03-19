import os
import networkx as nx
import numpy as np
from tqdm.notebook import tqdm
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from google.colab import drive

# Mount Google Drive
drive.mount('/content/drive')

# Load the pre-trained BERT model for text embedding
model = SentenceTransformer('all-MiniLM-L6-v2')

def load_graph(file_path):
    """Load the graph from a GraphML file."""
    return nx.read_graphml(file_path, node_type=str)

def generate_node_embeddings(G):
    """Generate embeddings for each node in the graph."""
    embeddings = {}
    for node in tqdm(G.nodes(), desc="Generating node embeddings"):
        embeddings[node] = model.encode(node)
    return embeddings

def should_exclude_node(node):
    """Check if the node should be excluded based on specific prefixes, numbers, month names, or written numbers."""
    exclude_prefixes = ('non-', 'un-', 'not ')
    months = [
        'january', 'february', 'march', 'april', 'may', 'june', 'july',
        'august', 'september', 'october', 'november', 'december',
        'jan', 'feb', 'mar', 'apr', 'jun', 'jul', 'aug', 'sep', 'oct', 'nov', 'dec'
    ]
    written_numbers = ['two', 'three', 'four', 'five', 'six', 'seven', 'eight', 'nine', 'ten']

    node_lower = node.lower()
    return (
        any(node_lower.startswith(prefix) for prefix in exclude_prefixes) or
        any(char.isdigit() for char in node) or
        any(month in node_lower for month in months) or
        any(num in node_lower.split() for num in written_numbers)
    )

def merge_nodes(G, node1, node2):
    """Merge two nodes in the directed graph."""
    # Combine incoming edges
    for predecessor in G.predecessors(node2):
        if G.has_edge(predecessor, node1):
            # If edge already exists, update attributes
            for attr, value in G[predecessor][node2].items():
                if attr in G[predecessor][node1]:
                    if isinstance(value, (int, float)):
                        G[predecessor][node1][attr] += value
                    elif isinstance(value, str):
                        G[predecessor][node1][attr] = f"{G[predecessor][node1][attr]}; {value}"
                else:
                    G[predecessor][node1][attr] = value
        else:
            # If edge doesn't exist, add it
            G.add_edge(predecessor, node1, **G[predecessor][node2])

    # Combine outgoing edges
    for successor in G.successors(node2):
        if G.has_edge(node1, successor):
            # If edge already exists, update attributes
            for attr, value in G[node2][successor].items():
                if attr in G[node1][successor]:
                    if isinstance(value, (int, float)):
                        G[node1][successor][attr] += value
                    elif isinstance(value, str):
                        G[node1][successor][attr] = f"{G[node1][successor][attr]}; {value}"
                else:
                    G[node1][successor][attr] = value
        else:
            # If edge doesn't exist, add it
            G.add_edge(node1, successor, **G[node2][successor])

    # Remove the merged node
    G.remove_node(node2)

def simplify_graph(G, embeddings, similarity_threshold=0.95):
    """Simplify the directed graph by merging similar nodes."""
    nodes = list(G.nodes())
    embeddings_matrix = np.array([embeddings[node] for node in nodes])

    similarity_matrix = cosine_similarity(embeddings_matrix)

    merged_nodes = set()
    for i in range(len(nodes)):
        if nodes[i] in merged_nodes or should_exclude_node(nodes[i]):
            continue
        for j in range(i+1, len(nodes)):
            if nodes[j] in merged_nodes or should_exclude_node(nodes[j]):
                continue
            if similarity_matrix[i][j] > similarity_threshold:
                print(f"Merging: {nodes[i]} <-> {nodes[j]}: {similarity_matrix[i][j]:.4f}")
                merge_nodes(G, nodes[i], nodes[j])
                merged_nodes.add(nodes[j])

    return G

def main():
    # File paths
    input_graph_path = '/content/drive/MyDrive/Output/Knowledge_Graph.graphml'
    output_graph_path = '/content/drive/MyDrive/Output/Processed_Knowledge_Graph.graphml'

    print("Loading graph...")
    G = load_graph(input_graph_path)

    print("Generating node embeddings...")
    node_embeddings = generate_node_embeddings(G)

    print("Simplifying graph...")
    G_simplified = simplify_graph(G, node_embeddings)

    print(f"Original graph: {len(G.nodes())} nodes, {len(G.edges())} edges")
    print(f"Simplified graph: {len(G_simplified.nodes())} nodes, {len(G_simplified.edges())} edges")

    print("Saving simplified graph...")
    nx.write_graphml(G_simplified, output_graph_path)
    print(f"Simplified graph saved to: {output_graph_path}")

if __name__ == "__main__":
    main()

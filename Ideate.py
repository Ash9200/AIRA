import networkx as nx
import pandas as pd
from openai import OpenAI
import json
import random
from tqdm import tqdm
from pydantic import BaseModel, Field
import datetime

def load_graph(file_path):
    return nx.read_graphml(file_path, node_type=str)

def load_centrality_rankings(file_path):
    return pd.read_csv(file_path)

def get_top_unique_nodes(rankings, N=10):
    centrality_types = ['In-Degree Centrality', 'Out-Degree Centrality', 'Betweenness Centrality', 'Eigenvector Centrality']
    unique_top_nodes = []
    for centrality_type in centrality_types:
        count = 0
        for node in rankings.nlargest(len(rankings), centrality_type)['Node']:
            if node not in unique_top_nodes:
                unique_top_nodes.append(node)
                count += 1
                if count == N:
                    break
    return unique_top_nodes

def find_random_paths(G, start, end, max_paths=5, cutoff=None):
    all_paths = []
    for path in tqdm(nx.all_simple_paths(G, start, end, cutoff=cutoff), desc="Finding paths", leave=False):
        all_paths.append(path)
        if len(all_paths) >= max_paths:
            break
    return all_paths

def subgraph_to_text(G, paths, start_node, end_node):
    text_representation = [
        f"Subgraph representation showing paths from '{start_node}' to '{end_node}':",
        f"START NODE: {start_node}",
        f"END NODE: {end_node}",
        "Paths:"
    ]
    for i, path in enumerate(paths, 1):
        path_representation = []
        for j in range(len(path) - 1):
            node1, node2 = path[j], path[j+1]
            edge_data = G[node1][node2]
            relationship = edge_data.get('relationship', 'related to')
            path_representation.append(f"{node1} --[{relationship}]--> {node2}")
        text_representation.append(f"Path {i}: " + " ".join(path_representation))
    return "\n".join(text_representation)

class ResearchIdea(BaseModel):
    hypothesis: str = Field(..., description="A specific, testable hypothesis or investigative statement based on the subgraph")
    explanation: str = Field(..., description="A detailed explanation and justification for the hypothesis")

def generate_research_hypothesis(subgraph_text, domain, start_node, end_node):
    prompt = "[prompt placeholder]"

    try:
        completion = client.beta.chat.completions.parse(
            model="[model placeholder]",
            messages=[
                {"role": "system", "content": "[prompt placeholder]"},
                {"role": "user", "content": prompt}
            ],
            response_format=ResearchIdea
        )
        return completion.choices[0].message.parsed
    except Exception as e:
        print(f"Error in generating research hypothesis: {e}")
        return None

def ideation_process(graph_path, rankings_path, domain, num_pairs=5, max_paths=5, path_cutoff=None):
    G = load_graph(graph_path)
    rankings = load_centrality_rankings(rankings_path)

    # Get top unique nodes
    top_nodes = get_top_unique_nodes(rankings, N=3)  # This will give us 12 unique nodes

    research_ideas = []
    used_pairs = set()  # Set to keep track of used pairs

    attempts = 0
    max_attempts = len(top_nodes) * (len(top_nodes) - 1) // 2  # Maximum possible unique pairs

    while len(research_ideas) < num_pairs and attempts < max_attempts:
        # Randomly select a pair that hasn't been used
        available_pairs = [(n1, n2) for n1 in top_nodes for n2 in top_nodes if n1 != n2 and (n1, n2) not in used_pairs and (n2, n1) not in used_pairs]

        if not available_pairs:
            print("All possible pairs have been exhausted.")
            break

        start_node, end_node = random.choice(available_pairs)
        used_pairs.add((start_node, end_node))
        attempts += 1

        paths = find_random_paths(G, start_node, end_node, max_paths=max_paths, cutoff=path_cutoff)

        if not paths:
            print(f"No path found between {start_node} and {end_node}. Trying a different pair.")
            continue

        subgraph_text = subgraph_to_text(G, paths, start_node, end_node)

        print(f"\nSubgraph Text Representation:")
        print(subgraph_text)
        print("-" * 50)

        idea = generate_research_hypothesis(subgraph_text, domain, start_node, end_node)
        if idea:
            research_ideas.append({
                'hypothesis': idea.hypothesis,
                'explanation': idea.explanation,
                'start_node': start_node,
                'end_node': end_node
            })

    if len(research_ideas) < num_pairs:
        print(f"Warning: Only generated {len(research_ideas)} ideas out of the requested {num_pairs}.")

    return research_ideas

if __name__ == "__main__":
    graph_path = ""
    rankings_path = ""

    domain = input("Please specify the domain or goal of the research: ")
    num_pairs = int(input("How many node pairs would you like to generate? "))
    max_paths = int(input("Maximum number of paths to find between each pair of nodes: "))
    path_cutoff = input("Enter a maximum path length (or press Enter for no limit): ")
    path_cutoff = int(path_cutoff) if path_cutoff else None

    results = ideation_process(graph_path, rankings_path, domain, num_pairs, max_paths, path_cutoff)

    # Generate timestamp
    timestamp = datetime.datetime.now().strftime("%m%d_%H%M")

    # Save results to a JSON file in the Analytical_Output folder
    output_filename = f''  
    with open(output_filename, 'w') as f:
        json.dump(results, f, indent=2)

    print("\nGenerated research ideas:")
    for i, item in enumerate(results, 1):
        print(f"Idea {i}:")
        print(f"Hypothesis: {item['hypothesis']}")
        print(f"Explanation: {item['explanation']}")
        print(f"Start Node: {item['start_node']}")
        print(f"End Node: {item['end_node']}")
        if i < len(results):
            print("\n" + "-"*50 + "\n")

    print(f"\nResults saved to {output_filename}")

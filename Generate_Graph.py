import uuid
import pandas as pd
import numpy as np
from tqdm import tqdm
from openai import OpenAI
import json
import networkx as nx
import PyPDF2
from pydantic import BaseModel, Field
from typing import List

class Triplet(BaseModel):
    node_1: str = Field(..., description="The first node (subject) of the triplet")
    edge: str = Field(..., description="The edge (relationship) between the two nodes")
    node_2: str = Field(..., description="The second node (object) of the triplet")

class TripletSet(BaseModel):
    triplets: List[Triplet] = Field(..., description="A list of exactly 10 triplets extracted from the text")

def read_file(file_path):
    with open(file_path, 'rb') as file:
        if file_path.lower().endswith('.json'):
            return json.load(file)
        elif file_path.lower().endswith('.pdf'):
            pdf_reader = PyPDF2.PdfReader(file)
            text = ""
            for page in pdf_reader.pages:
                text += page.extract_text()
            return text
        else:
            return file.read().decode('utf-8')

def load_existing_graph(output_folder):
    graph_path = os.path.join(output_folder, "knowledge_graph.graphml")
    if os.path.exists(graph_path):
        return nx.read_graphml(graph_path)
    return None

def split_text_into_chunks(text, chunk_size=2500, chunk_overlap=0):
    lines = text.split('\n')
    chunks = []
    current_chunk = []
    current_length = 0

    for line in lines:
        if current_length + len(line) > chunk_size and current_chunk:
            chunks.append('\n'.join(current_chunk))

            # Handle overlap conditionally
            if chunk_overlap > 0 and current_chunk[-1].strip():
                overlap_lines = current_chunk[-int(chunk_overlap / max(1, len(current_chunk[-1]))):]
            else:
                overlap_lines = []

            current_chunk = overlap_lines
            current_length = sum(len(l) for l in current_chunk)

        current_chunk.append(line)
        current_length += len(line)

    if current_chunk:
        chunks.append('\n'.join(current_chunk))

    return chunks

def generate_triplets(chunk, chunk_id, max_retries=3):
    SYSTEM_PROMPT = "[prompt placeholder]"

    USER_PROMPT = "[prompt placeholder]"

    for attempt in range(max_retries):
        try:
            completion = client.beta.chat.completions.parse(
                model="[model placeholder]",
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": USER_PROMPT}
                ],
                response_format=TripletSet,
            )

            # Extract triplets and add chunk_id during triplet creation
            triplets_with_chunk_id = [
                {**triplet.dict(), "chunk_id": chunk_id} for triplet in completion.choices[0].message.parsed.triplets
            ]

            return triplets_with_chunk_id

        except Exception as e:
            if attempt == max_retries - 1:
                print(f"Failed to generate triplets after {max_retries} attempts. Error: {e}")
                return []
            else:
                print(f"Attempt {attempt + 1} failed. Retrying...")

    return []

def process_file(file_path, chunk_size=2500, chunk_overlap=0):
    df = pd.DataFrame(columns=['node_1', 'edge', 'node_2', 'chunk_id'])

    text = read_file(file_path)
    chunks = split_text_into_chunks(text, chunk_size, chunk_overlap)

    for chunk_id, chunk in enumerate(tqdm(chunks, desc="Processing chunks")):
        triplets = generate_triplets(chunk, chunk_id)
        if triplets:
            chunk_df = pd.DataFrame(triplets)
            df = pd.concat([df, chunk_df], ignore_index=True)

    return df

def create_graph_from_triplets(df):
    G = nx.DiGraph()

    # Add edges to the graph
    for _, row in df.iterrows():
        G.add_edge(
            str(row["node_1"]).lower(),
            str(row["node_2"]).lower(),
            title=row["edge"].lower(),
            weight=1,
            relationship=row["edge"].lower(),
            chunk_id=row["chunk_id"]
        )

    # Group by node pairs and edge, and aggregate unique chunk IDs
    edge_counts = df.groupby(['node_1', 'node_2', 'edge']).agg({
        'chunk_id': lambda x: ','.join(map(str, set(x)))  # Convert list of chunk IDs to a string
    }).reset_index()

    # Update graph edge attributes
    for _, row in edge_counts.iterrows():
        if G.has_edge(row['node_1'], row['node_2']):
            G[row['node_1']][row['node_2']]['chunk_ids'] = row['chunk_id']

    return G

def process_and_create_graph(df, output_folder):
    print("Creating new graph from triplets...")
    new_G = create_graph_from_triplets(df)

    existing_G = load_existing_graph(output_folder)

    if existing_G:
        print("Merging new information with existing graph...")
        G = nx.compose(existing_G, new_G)
        for u, v, data in new_G.edges(data=True):
            if G.has_edge(u, v):
                existing_chunk_ids = G[u][v].get('chunk_ids', [])
                new_chunk_ids = data.get('chunk_ids', [])
                G[u][v]['chunk_ids'] = list(set(existing_chunk_ids + new_chunk_ids))
    else:
        print("No existing graph found. Using newly created graph.")
        G = new_G

    print("Saving updated graph...")
    nx.write_graphml(G, os.path.join(output_folder, "Knowledge_Graph.graphml"))

    return G

def main():
    input_folder = ''
    output_folder = ''
    log_file = os.path.join(output_folder, 'Processed_Files.json')
    os.makedirs(output_folder, exist_ok=True)

    chunk_size = 2500
    chunk_overlap = 0

    if os.path.exists(log_file):
        with open(log_file, 'r') as f:
            processed_files = set(json.load(f))
    else:
        processed_files = set()

    for filename in tqdm(os.listdir(input_folder), desc="Processing files"):
        if filename.lower().endswith(('.txt', '.json', '.pdf')) and filename not in processed_files:
            file_path = os.path.join(input_folder, filename)

            print(f"Processing file: {filename}")
            result_df = process_file(file_path, chunk_size, chunk_overlap)

            output_csv = os.path.join(output_folder, f"{os.path.splitext(filename)[0]}_triplets.csv")
            result_df.to_csv(output_csv, index=False)
            print(f"Saved triplets to {output_csv}")

            G = process_and_create_graph(result_df, output_folder)
            print(f"Graph processing complete for {filename}")

            processed_files.add(filename)

            # Update the JSON file after each processed file
            with open(log_file, 'w') as f:
                json.dump(list(processed_files), f)

            print(f"Updated processed files log: {filename} added")
            print("\n")

    print("All files processed")

if __name__ == "__main__":
    main()

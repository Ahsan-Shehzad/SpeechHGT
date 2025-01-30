import numpy as np
import scipy
from sklearn.cluster import KMeans
from sklearn.metrics import mutual_info_score
import HyperNetX as hnx
from scipy.spatial.distance import cdist
import pandas as pd

# Function to compute pairwise mutual information between features
def compute_mutual_information(features):
    num_features = features.shape[1]
    mi_matrix = np.zeros((num_features, num_features))

    # Compute mutual information for all pairs of features
    for i in range(num_features):
        for j in range(i+1, num_features):
            mi_value = mutual_info_score(features[:, i], features[:, j])
            mi_matrix[i, j] = mi_value
            mi_matrix[j, i] = mi_value

    return mi_matrix

# Function to perform k-means clustering on feature correlations
def cluster_features(features, n_clusters=5):
    # Perform k-means clustering on the correlation matrix
    correlation_matrix = np.corrcoef(features, rowvar=False)
    
    # We use the correlation matrix for clustering the features
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    kmeans.fit(correlation_matrix)
    return kmeans.labels_

# Function to construct the hypergraph
def construct_hypergraph(features, n_clusters=5):
    # Compute pairwise mutual information
    mi_matrix = compute_mutual_information(features)

    # Create nodes for each feature
    num_features = features.shape[1]
    nodes = [f"feature_{i}" for i in range(num_features)]

    # Create pairwise edges based on mutual information
    edges = []
    for i in range(num_features):
        for j in range(i + 1, num_features):
            if mi_matrix[i, j] > 0.1:  # Set a threshold for mutual information
                edges.append((nodes[i], nodes[j]))

    # Cluster the features based on the correlation matrix
    feature_labels = cluster_features(features, n_clusters=n_clusters)
    
    # Create hyperedges based on clustering
    hyperedges = defaultdict(list)
    for feature_index, label in enumerate(feature_labels):
        hyperedges[label].append(nodes[feature_index])

    # Construct the hypergraph
    hypergraph = hnx.Hypergraph()

    # Add nodes to the hypergraph
    for node in nodes:
        hypergraph.add_node(node)

    # Add edges (pairwise relationships) to the hypergraph
    for edge in edges:
        hypergraph.add_edge(*edge)

    # Add hyperedges (higher-order interactions) to the hypergraph
    for hyperedge in hyperedges.values():
        hypergraph.add_edge(*hyperedge)

    # Create the incidence matrix
    incidence_matrix = np.zeros((num_features, len(hyperedges)))
    hyperedge_indices = {tuple(hyperedge): idx for idx, hyperedge in enumerate(hyperedges.values())}

    for feature_idx, label in enumerate(feature_labels):
        feature_node = nodes[feature_idx]
        for hyperedge in hyperedges[label]:
            incidence_matrix[feature_idx, hyperedge_indices[tuple(hyperedges[label])]] = 1

    return hypergraph, incidence_matrix, nodes, edges, hyperedges

# Function to save the hypergraph structure
def save_hypergraph(hypergraph, incidence_matrix, nodes, edges, hyperedges, output_path):
    # Save the hypergraph structure
    hypergraph_file = f"{output_path}/hypergraph.gml"
    hnx.write(hypergraph, hypergraph_file)

    # Save the incidence matrix as a CSV
    incidence_matrix_file = f"{output_path}/incidence_matrix.csv"
    pd.DataFrame(incidence_matrix, columns=[f"hyperedge_{i}" for i in range(incidence_matrix.shape[1])],
                 index=nodes).to_csv(incidence_matrix_file)

    # Save nodes and edges to a file
    with open(f"{output_path}/nodes.txt", "w") as f:
        for node in nodes:
            f.write(f"{node}\n")

    with open(f"{output_path}/edges.txt", "w") as f:
        for edge in edges:
            f.write(f"{edge[0]} -- {edge[1]}\n")

    # Save hyperedges
    with open(f"{output_path}/hyperedges.txt", "w") as f:
        for hyperedge in hyperedges.values():
            f.write(f"{', '.join(hyperedge)}\n")

    print(f"Hypergraph saved to {output_path}")

# Main function to construct the hypergraph
def main():
    # Example extracted features (rows: samples, columns: features)
    # You can replace this with the actual extracted features from your dataset
    features = np.random.rand(100, 20)  # 100 samples, 20 features (modify as needed)

    # Set the output path for saving the hypergraph structure
    output_path = "path_to_output_directory"

    # Construct the hypergraph
    hypergraph, incidence_matrix, nodes, edges, hyperedges = construct_hypergraph(features)

    # Save the hypergraph structure to files
    save_hypergraph(hypergraph, incidence_matrix, nodes, edges, hyperedges, output_path)

if __name__ == "__main__":
    main()

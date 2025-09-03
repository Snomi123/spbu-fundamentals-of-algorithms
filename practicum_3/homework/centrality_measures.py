from typing import Any, Protocol
from itertools import combinations

import numpy as np
import networkx as nx


class CentralityMeasure(Protocol):
    def __call__(self, G: nx.Graph) -> dict[Any, float]:
        ...

def plot_graph(G: nx.Graph, node_weights: dict[Any, float], figsize=(14, 8), name: str = ""):
    plt.figure(figsize=figsize)
    pos = nx.spring_layout(G, seed=42)
    
    nodes = nx.draw_networkx_nodes(
        G, pos,
        node_color=list(node_weights.values()),
        cmap=plt.cm.viridis,
        node_size=500,
        alpha=0.8
    )
    
    nx.draw_networkx_edges(G, pos, alpha=0.3)
    nx.draw_networkx_labels(G, pos)
    
    plt.colorbar(nodes)
    plt.title(f"{name} Centrality")
    plt.axis("off")
    plt.show()

def create_custom_graph():
    G = nx.Graph()
    
    G.add_nodes_from(range(1, 16))

    edges = [
        (1, 2), (1, 3), (2, 3), (2, 4), (3, 5),
        (4, 5), (4, 6), (5, 7), (6, 7), (6, 8),
        (7, 9), (8, 9), (8, 10), (9, 11), (10, 11),
        (10, 12), (11, 13), (12, 13), (12, 14), (13, 15), (14, 15)
    ]
    G.add_edges_from(edges)
    
    return G
    
def closeness_centrality(G: AnyNxGraph) -> dict[Any, float]:
    centrality = {}
    n = len(G.nodes())
    
    for node in G.nodes():
        total_distance = 0
        for other in G.nodes():
            if node != other:
                try:
                    total_distance += nx.shortest_path_length(G, node, other)
                except nx.NetworkXNoPath:
                    continue
        centrality[node] = (n - 1) / total_distance if total_distance > 0 else 0
        
    return centrality


def betweenness_centrality(G: AnyNxGraph) -> dict[Any, float]: 
    centrality = {node: 0.0 for node in G.nodes()}
    nodes = list(G.nodes())
    
    for s, t in combinations(nodes, 2):
        try:
            paths = list(nx.all_shortest_paths(G, s, t))
            for path in paths:
                for node in path[1:-1]:  # Исключаем начальную и конечную вершины
                    centrality[node] += 1.0 / len(paths)
        except nx.NetworkXNoPath:
            continue
            
    return centrality


def eigenvector_centrality(G: AnyNxGraph) -> dict[Any, float]: 
    nodes = list(G.nodes())
    A = nx.to_numpy_array(G, nodelist=nodes)
    
    eigenvalues, eigenvectors = np.linalg.eig(A)
    
    idx = np.argmax(eigenvalues)
    principal_vector = eigenvectors[:, idx].real
    
    if np.linalg.norm(principal_vector) > 0:
        principal_vector = principal_vector / np.linalg.norm(principal_vector)
    
    return {node: principal_vector[i] for i, node in enumerate(nodes)}


def plot_centrality_measure(G: nx.Graph, measure: CentralityMeasure) -> None:
    values = measure(G)
    if values is not None:
        plot_graph(G, node_weights=values, name=measure.__name__)
    else:
        print(f"Реализуйте функцию {measure.__name__}")


if __name__ == "__main__":
    G = create_custom_graph()
    
    plot_centrality_measure(G, closeness_centrality)
    plot_centrality_measure(G, betweenness_centrality)
    plot_centrality_measure(G, eigenvector_centrality)


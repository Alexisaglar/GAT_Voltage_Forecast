import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import torch

from data_loader import create_dataset
from model import GATNet


def load_data_and_model(model_path, data_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = GATNet().to(device)
    model.load_state_dict(torch.load(model_path))
    model.eval()

    data_list, target_list = create_dataset(data_path)
    return model, data_list[1000], device  # Only return the first data point
    # return model, data_list, device  # Only return the first data point


def evaluate_model(model, data, device):
    model.eval()  # Set the model to evaluation mode
    with torch.no_grad():  # No gradients are needed
        data = data.to(device)
        output, attention_weights = model(data)
        edges = attention_weights[0].t().cpu().numpy()  # Transpose and convert to numpy
        scores = (
            attention_weights[1].squeeze().cpu().numpy()
        )  # Squeeze and convert to numpy
    return output, edges, scores


def visualize_graph(edges, scores):
    G = nx.Graph()  # Use nx.DiGraph() if edges are directed
    # G = nx.karate_club_graph()  # Use nx.DiGraph() if edges are directed
    for edge, weight in zip(edges, scores):
        G.add_edge(edge[0], edge[1], weight=weight)

    plt.figure(figsize=(14, 10))
    ax = plt.gca()

    pos = nx.spring_layout(G, k=0.9, iterations=600, weight="weight")
    # pos = nx.(G)
    weights = [G[u][v]["weight"] for u, v in G.edges()]
    edge_colors = plt.cm.YlOrRd(
        np.array(weights) / max(weights)
    )  # Normalize for coloring

    # Node sizes can also be scaled by some property (e.g., degree)
    node_sizes = [10 + 10 * G.degree(n) for n in G.nodes()]
    nx.draw_networkx_nodes(G, pos, node_size=node_sizes, node_color="skyblue")

    # Draw regular edges
    normal_edges = [(u, v) for u, v in G.edges() if u != v]
    nx.draw_networkx_edges(
        G,
        pos,
        edgelist=normal_edges,
        width=np.array(weights) * 5,
        edge_color=edge_colors,
        arrowsize=10,
    )

    # Specifically draw self-loops with a different, smaller connectionstyle
    loop_edges = [(u, v) for u, v in G.edges() if u == v]
    if loop_edges:
        loop_weights = [G[u][v]["weight"] for u, v in loop_edges]
        loop_edge_colors = plt.cm.YlOrRd(np.array(loop_weights) / max(weights))
        nx.draw_networkx_edges(
            G,
            pos,
            edgelist=loop_edges,
            width=np.array(loop_weights) * 1.8,
            edge_color=loop_edge_colors,
            arrowsize=10,
            connectionstyle="arc3,rad=0.005",
        )  # Smaller radius for loops

    # Labels and title
    nx.draw_networkx_labels(G, pos)

    # Enhanced colorbar
    sm = plt.cm.ScalarMappable(
        cmap=plt.cm.YlOrRd, norm=plt.Normalize(vmin=min(scores), vmax=max(scores))
    )
    sm.set_array([])
    plt.colorbar(sm, orientation="vertical", label="Attention Weights", ax=ax)

    plt.title("Graph Attention Network Visualization")
    plt.savefig(f'plots/attention_visualisation', dpi=300)
    plt.show()


def main():
    model_path = "checkpoints/best_model.pth"
    # data_path = 'raw_data/network_results.h5'
    data_path = "raw_data/33_bus_results.h5"
    model, first_data, device = load_data_and_model(model_path, data_path)
    print(first_data)
    # first_data = torch.tensor(first_data)
    output, edges, scores = evaluate_model(model, first_data, device)
    print(output)
    visualize_graph(edges, scores)


if __name__ == "__main__":
    main()

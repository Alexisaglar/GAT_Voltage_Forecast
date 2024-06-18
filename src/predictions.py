import torch
import matplotlib.pyplot as plt
import networkx as nx
from model import GATNet
from data_loader import create_dataset

def load_data_and_model(model_path, data_path):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = GATNet().to(device)
    model.load_state_dict(torch.load(model_path))
    model.eval()

    data_list, target_list = create_dataset(data_path)
    # return model, data_list[0], device  # Only return the first data point
    return model, data_list, device  # Only return the first data point

def evaluate_model(model, data, device):
    model.eval()  # Set the model to evaluation mode
    with torch.no_grad():  # No gradients are needed
        data = data.to(device)
        output, attention_weights = model(data)
        edges = attention_weights[0].t().cpu().numpy()  # Transpose and convert to numpy
        scores = attention_weights[1].squeeze().cpu().numpy()  # Squeeze and convert to numpy
    return output, edges, scores

def visualize_graph(edges, scores):
    G = nx.Graph()
    i = 0
    for edge, weight in zip(edges, scores):  # Iterate through edges and corresponding weights
        i+=1
        if i > 32:
            G.add_edge(edge[0], edge[1], weight=weight)  # Add directed edge with weight

    plt.figure(figsize=(12, 8))  # Create a new figure and set its size
    ax = plt.gca()  # Get tedgeshe current axes, creating them if necessary

    pos = nx.spring_layout(G, k=1, iterations=1000)  # Node positioning
    weights = [G[u][v]['weight'] for u, v in G.edges()]  # Edge weights for visualization

    # Draw the graph
    nx.draw(G, pos, with_labels=True, node_color='skyblue', node_size=100,
            edge_color=weights, edge_cmap=plt.cm.YlOrRd, width=3, arrowstyle='-|>', arrowsize=10, ax=ax)

    # Create colorbar
    sm = plt.cm.ScalarMappable(cmap=plt.cm.YlOrRd, norm=plt.Normalize(vmin=min(scores), vmax=max(scores)))
    sm.set_array([])
    plt.colorbar(sm, orientation='vertical', label='Attention Weights', ax=ax)

    plt.title('Graph Attention Network Visualization')
    plt.show()

def main():
    model_path = 'checkpoints/best_model.pth'
    # data_path = 'raw_data/network_results.h5'
    data_path = 'power_flow_data.h5'
    model, first_data, device = load_data_and_model(model_path, data_path)
    print(first_data)
    _, edges, scores = evaluate_model(model, first_data, device)
    visualize_graph(edges, scores)

if __name__ == "__main__":
    main()


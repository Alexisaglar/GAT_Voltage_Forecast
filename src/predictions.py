import torch
from torch_geometric.loader import DataLoader
import matplotlib.pyplot as plt
import numpy as np
import networkx as nx
from model import GATNet
from data_loader import create_dataset
from train import train_model
from test import test_model

def load_data_and_model(model_path, data_path):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = GATNet().to(device)
    model.load_state_dict(torch.load(model_path))
    model.eval()

    data_list, target_list = create_dataset(data_path)
    return model, data_list[0], device  # Only return the first data point

def evaluate_model(model, data, device):
    model.eval()  # Set the model to evaluation mode
    with torch.no_grad():  # No gradients are needed
        data = data.to(device)
        out, attention_weights = model(data)
    return out, attention_weights

def visualize_attention_weights(node_features, edge_index, attention_weights, layer_idx):
    G = nx.Graph()
    node_features = node_features.cpu().numpy()
    edge_index = edge_index.cpu().numpy()
    attention_weights = attention_weights.mean(dim=1).cpu().numpy()  # Average over heads

    for i, features in enumerate(node_features):
        G.add_node(i, features=features)

    for idx, (src, dst) in enumerate(edge_index.T):
        G.add_edge(src, dst, weight=attention_weights[idx])

    pos = nx.spring_layout(G, k=1, iterations=1000)
    # pos = nx.spring_layout(G)
    edge_weights = [G[u][v]['weight'] for u, v in G.edges]

    fig, ax = plt.subplots(figsize=(12, 8))
# Ensure background color is set to gray
    fig.patch.set_facecolor('gray')  # Set the background color for the figure
    ax.set_facecolor('gray')  # Set the background color for the axes

    nx.draw(G, pos, ax=ax, with_labels=True, node_color='lightblue', node_size=500, 
            font_size=10, width=edge_weights, edge_color=edge_weights, edge_cmap=plt.cm.YlOrRd)

    sm = plt.cm.ScalarMappable(cmap=plt.cm.YlOrRd, 
                               norm=plt.Normalize(vmin=min(edge_weights), vmax=max(edge_weights)))
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax)
    cbar.ax.set_facecolor('gray')  # Set the background color for the colorbar
    
    ax.set_title(f"Layer {layer_idx} Attention Weights", color='white')
    plt.show()

def main():
    model_path = 'checkpoints/best_model.pth'
    data_path = 'raw_data/network_results.h5'
    model, first_data, device = load_data_and_model(model_path, data_path)
    predictions, attention_weights = evaluate_model(model, first_data, device)

    for layer_idx, layer_attention in enumerate(attention_weights):  # Plot only the first graph in the test set
        visualize_attention_weights(first_data.x, first_data.edge_index, layer_attention[1], layer_idx)

if __name__ == "__main__":
    main()


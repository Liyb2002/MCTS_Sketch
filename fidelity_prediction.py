from torch.utils.data import Dataset, DataLoader
import os
import pickle
from tqdm import tqdm
import torch
import numpy as np

import Preprocessing.SBGCN.brep_read
import Preprocessing.proc_CAD.helper
import Encoders.helper

import whole_process_evaluate




import Preprocessing.dataloader
import Preprocessing.gnn_graph

import Encoders.gnn.gnn
import Encoders.helper

import Preprocessing.proc_CAD
import Preprocessing.proc_CAD.helper
from torch_geometric.loader import DataLoader

from tqdm import tqdm
from Preprocessing.config import device
import torch
import torch.nn as nn
import torch.optim as optim
import os
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D





graph_encoder = Encoders.gnn.gnn.SemanticModule()
graph_decoder = Encoders.gnn.gnn.Fidelity_Decoder()

graph_encoder.to(device)
graph_decoder.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(list(graph_encoder.parameters()) + list(graph_decoder.parameters()), lr=0.0004)
batch_size = 16

# ------------------------------------------------------------------------------# 

current_dir = os.getcwd()
save_dir = os.path.join(current_dir, 'checkpoints', 'fidelity_prediction')
os.makedirs(save_dir, exist_ok=True)

def load_models():
    graph_encoder.load_state_dict(torch.load(os.path.join(save_dir, 'graph_encoder.pth')))
    graph_decoder.load_state_dict(torch.load(os.path.join(save_dir, 'graph_decoder.pth')))


def save_models():
    torch.save(graph_encoder.state_dict(), os.path.join(save_dir, 'graph_encoder.pth'))
    torch.save(graph_decoder.state_dict(), os.path.join(save_dir, 'graph_decoder.pth'))

# ------------------------------------------------------------------------------# 



def compute_accuracy(predictions, ground_truth):
    """
    Computes accuracy for classification predictions.

    Args:
        predictions (torch.Tensor): Predicted logits, shape (batch_size, num_bins).
        ground_truth (torch.Tensor): Ground truth bin indices, shape (batch_size).

    Returns:
        correct (int): Number of correct predictions.
        total (int): Total number of predictions.
    """
    # Get the predicted bin index by finding the max logit
    predicted_bins = torch.argmax(predictions, dim=1)  # Shape: (batch_size,)

    # Compare with ground truth
    correct = (predicted_bins == ground_truth).sum().item()
    total = ground_truth.size(0)

    return correct, total


def compute_accuracy_relaxed(predictions, ground_truth):
    """
    Computes relaxed accuracy for classification predictions.
    A prediction is considered correct if it is within +-1 bin of the ground truth.

    Args:
        predictions (torch.Tensor): Predicted logits, shape (batch_size, num_bins).
        ground_truth (torch.Tensor): Ground truth bin indices, shape (batch_size).

    Returns:
        correct (int): Number of relaxed correct predictions.
        total (int): Total number of predictions.
        accuracy (float): The percentage of relaxed correct predictions.
    """
    # Get the predicted bin index by finding the max logit
    predicted_bins = torch.argmax(predictions, dim=1)  # Shape: (batch_size,)

    # Compare with ground truth within a tolerance of +-1 bin
    correct = ((predicted_bins == ground_truth) | 
               (predicted_bins == ground_truth + 1) | 
               (predicted_bins == ground_truth - 1)).sum().item()
    total = ground_truth.size(0)

    return correct, total

# ------------------------------------------------------------------------------# 

def calculate_bins_with_min_score(S_min=0.3, S_max=1.0, gamma=2, num_bins=10):
    # Transformed bin edges (equally spaced in [0, 1])
    transformed_bin_edges = torch.linspace(0, 1, num_bins + 1, device=device)

    # Rescale bin edges to the range [S_min, S_max]
    original_bin_edges = S_min + (transformed_bin_edges ** (1 / gamma)) * (S_max - S_min)
    # return original_bin_edges

    return torch.linspace(0, 1, num_bins + 1)



def compute_bin_score(cur_fidelity_score, bins):
    """
    Maps fidelity scores into hardcoded bins and computes the bin score (midpoint of the bin).

    Parameters:
        cur_fidelity_score (torch.Tensor): Tensor of fidelity scores.
        bins (list or torch.Tensor): List of bin edges.
        device (str): Device to move tensors to.

    Returns:
        torch.Tensor: Bin scores for each fidelity score.
    """
    # Ensure bins are a tensor and on the correct device
    bins = torch.tensor(bins, dtype=torch.float32)

    # Compute bin indices (0-based)
    bin_indices = torch.bucketize(cur_fidelity_score, bins) - 1

    # Handle edge case for scores exactly equal to the last bin's upper limit
    bin_indices = torch.clamp(bin_indices, 0, len(bins) - 2)

    return bin_indices



# ------------------------------------------------------------------------------# 



def train():

    # Set up dataloader
    dataset = whole_process_evaluate.Evaluation_Dataset('program_output')

    total_correct = 0
    total = 0

    best_val_accuracy = 0
    epochs = 30

    graphs = []
    gt_fidelity_score = []

    bins = calculate_bins_with_min_score()

    for data in tqdm(dataset, desc="Evaluating CAD Programs"):
        stroke_node_features, output_brep_edges, gt_brep_edges, cur_fidelity_score, stroke_cloud_loops, strokes_perpendicular, loop_neighboring_vertical, loop_neighboring_horizontal, loop_neighboring_contained, stroke_to_loop, stroke_to_edge = data
    
        gnn_graph = Preprocessing.gnn_graph.SketchLoopGraph(
            stroke_cloud_loops, 
            stroke_node_features, 
            strokes_perpendicular, 
            loop_neighboring_vertical, 
            loop_neighboring_horizontal, 
            loop_neighboring_contained,
            stroke_to_loop,
            stroke_to_edge
        )
        gnn_graph.to_device_withPadding(device)
        graphs.append(gnn_graph)



        cur_fidelity_score = cur_fidelity_score.to(device)
        binned_score = compute_bin_score(cur_fidelity_score, bins)  # Get the bin index (0-based)
        gt_fidelity_score.append(binned_score)  # Append the bin index as the ground truth

        # Vis
        # print("binned_score", binned_score)
        # Encoders.helper.vis_brep(output_brep_edges)
        # Encoders.helper.vis_brep(gt_brep_edges)

        # if len(graphs) > 40:
        #     break



    print(f"Total number of preprocessed graphs: {len(graphs)}")


    # Split the dataset into training and validation sets (80-20 split)
    split_index = int(0.8 * len(graphs))
    train_graphs, val_graphs = graphs[:split_index], graphs[split_index:]
    train_scores, val_scores = gt_fidelity_score[:split_index], gt_fidelity_score[split_index:]

    # Convert train and validation graphs to HeteroData
    hetero_train_graphs = [Preprocessing.gnn_graph.convert_to_hetero_data(graph) for graph in train_graphs]
    hetero_val_graphs = [Preprocessing.gnn_graph.convert_to_hetero_data(graph) for graph in val_graphs]

    # Create DataLoaders for training and validation graphs/masks
    graph_train_loader = DataLoader(hetero_train_graphs, batch_size=16, shuffle=False)
    score_train_loader = DataLoader(train_scores, batch_size=16, shuffle=False)

    graph_val_loader = DataLoader(hetero_val_graphs, batch_size=16, shuffle=False)
    score_val_loader = DataLoader(val_scores, batch_size=16, shuffle=False)


    # Training and validation loop
    epochs = 30  # Number of epochs
    best_accuracy = 0.0

    for epoch in range(epochs):
        train_loss = 0.0
        total_correct = 0
        total_samples = 0

        graph_encoder.train()
        graph_decoder.train()

        # Get total number of iterations for progress bar
        total_iterations = min(len(graph_train_loader), len(score_train_loader))

        # Training loop with progress bar
        for hetero_batch, batch_scores in tqdm(zip(graph_train_loader, score_train_loader), 
                                              desc=f"Epoch {epoch+1}/{epochs} - Training", 
                                              dynamic_ncols=True, 
                                              total=total_iterations):
        
            optimizer.zero_grad()

            x_dict = graph_encoder(hetero_batch.x_dict, hetero_batch.edge_index_dict)
            output = graph_decoder(x_dict)

            loss = criterion(output, batch_scores)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

            # Compute accuracy
            correct, total = compute_accuracy(output, batch_scores)
            total_correct += correct
            total_samples += total

        # Calculate epoch-level metrics
        train_accuracy = total_correct / total_samples
        train_loss = train_loss / total_samples
        print(f"Epoch {epoch+1}/{epochs} - Training Loss: {train_loss:.4f}, Accuracy: {train_accuracy:.4%}")



        graph_encoder.eval()
        graph_decoder.eval()

        val_loss = 0.0
        val_correct = 0
        val_samples = 0

        with torch.no_grad():
            for hetero_batch, batch_scores in tqdm(zip(graph_val_loader, score_val_loader), 
                                                  desc=f"Epoch {epoch+1}/{epochs} - Validation", 
                                                  dynamic_ncols=True, 
                                                  total=len(graph_val_loader)):
                # Forward pass
                x_dict = graph_encoder(hetero_batch.x_dict, hetero_batch.edge_index_dict)
                output = graph_decoder(x_dict)

                # Compute loss
                loss = criterion(output, batch_scores)
                val_loss += loss.item()

                # Compute accuracy
                correct, total = compute_accuracy(output, batch_scores)
                val_correct += correct
                val_samples += total

        # Calculate validation metrics
        val_accuracy = val_correct / val_samples
        val_loss = val_loss / val_samples
        print(f"Epoch {epoch+1}/{epochs} - Validation Loss: {val_loss:.4f}, Accuracy: {val_accuracy:.4%}")

        # Save the model if validation accuracy improves
        if val_accuracy > best_accuracy:
            best_accuracy = val_accuracy
            save_models()
            print("Best model saved.")

    print(f"Training complete. Best Validation Accuracy: {best_accuracy:.4%}")



def eval():
    """
    Evaluate the model on the validation dataset.
    """
    # Load models
    load_models()
    
    # Set up dataset
    dataset = whole_process_evaluate.Evaluation_Dataset('program_output')

    # Preprocess graphs
    graphs = []
    gt_fidelity_score = []
    bins = calculate_bins_with_min_score()

    for data in tqdm(dataset, desc="Preprocessing Validation Dataset"):
        stroke_node_features, output_brep_edges, gt_brep_edges, cur_fidelity_score, stroke_cloud_loops, strokes_perpendicular, loop_neighboring_vertical, loop_neighboring_horizontal, loop_neighboring_contained, stroke_to_loop, stroke_to_edge = data

        gnn_graph = Preprocessing.gnn_graph.SketchLoopGraph(
            stroke_cloud_loops, 
            stroke_node_features, 
            strokes_perpendicular, 
            loop_neighboring_vertical, 
            loop_neighboring_horizontal, 
            loop_neighboring_contained,
            stroke_to_loop,
            stroke_to_edge
        )
        gnn_graph.to_device_withPadding(device)
        graphs.append(gnn_graph)

        cur_fidelity_score = cur_fidelity_score.to(device)
        binned_score = compute_bin_score(cur_fidelity_score, bins)  # Get the bin index (0-based)
        gt_fidelity_score.append(binned_score)  # Append the bin index as the ground truth


        print("binned_score", binned_score)
        Encoders.helper.vis_brep(output_brep_edges)
        Encoders.helper.vis_brep(gt_brep_edges)


        if len(graphs) > 200:
            break

    print(f"Total number of validation graphs: {len(graphs)}")

    # Convert validation graphs to HeteroData
    hetero_val_graphs = [Preprocessing.gnn_graph.convert_to_hetero_data(graph) for graph in graphs]

    # Create DataLoaders for validation graphs/masks
    graph_val_loader = DataLoader(hetero_val_graphs, batch_size=16, shuffle=False)
    score_val_loader = DataLoader(gt_fidelity_score, batch_size=16, shuffle=False)

    graph_encoder.eval()
    graph_decoder.eval()

    val_loss = 0.0
    val_correct = 0
    val_samples = 0

    with torch.no_grad():
        for hetero_batch, batch_scores in tqdm(zip(graph_val_loader, score_val_loader), 
                                              desc="Evaluating Validation Dataset", 
                                              dynamic_ncols=True, 
                                              total=len(graph_val_loader)):
            # Forward pass
            x_dict = graph_encoder(hetero_batch.x_dict, hetero_batch.edge_index_dict)
            output = graph_decoder(x_dict)

            # Compute loss
            loss = criterion(output, batch_scores)
            val_loss += loss.item()

            # Compute accuracy
            correct, total = compute_accuracy_relaxed(output, batch_scores)
            val_correct += correct
            val_samples += total

    # Calculate validation metrics
    val_accuracy = val_correct / val_samples
    val_loss = val_loss / val_samples
    
    print(f"Validation Loss: {val_loss:.4f}, Accuracy: {val_accuracy:.4%}")



eval()
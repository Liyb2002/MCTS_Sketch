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
from collections import defaultdict




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
graph_decoder = Encoders.gnn.gnn.Fidelity_Decoder_bin()

graph_encoder.to(device)
graph_decoder.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(list(graph_encoder.parameters()) + list(graph_decoder.parameters()), lr=0.0004)
batch_size = 16

# ------------------------------------------------------------------------------# 

current_dir = os.getcwd()
save_dir = os.path.join(current_dir, 'checkpoints', 'fidelity_prediction_bin')
os.makedirs(save_dir, exist_ok=True)

def load_models():
    graph_encoder.load_state_dict(torch.load(os.path.join(save_dir, 'graph_encoder.pth'), weights_only=True))
    graph_decoder.load_state_dict(torch.load(os.path.join(save_dir, 'graph_decoder.pth'), weights_only=True))


def save_models():
    torch.save(graph_encoder.state_dict(), os.path.join(save_dir, 'graph_encoder.pth'))
    torch.save(graph_decoder.state_dict(), os.path.join(save_dir, 'graph_decoder.pth'))
    print("Models saved successfully!")  

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

    # correct = ((predicted_bins == ground_truth) | 
    #         (predicted_bins == ground_truth + 1) | 
    #         (predicted_bins == ground_truth - 1)).sum().item()

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
    """

    bins = torch.tensor(bins, dtype=torch.float32)
    bin_indices = torch.bucketize(cur_fidelity_score, bins) - 1
    bin_indices = torch.clamp(bin_indices, 0, len(bins) - 2)

    return bin_indices


def process_contrastive_batch(graphs, gt_scores):
    if len(graphs) < 2:
        print("Skipping batch (not enough graphs for contrastive evaluation).")
        return 0, 0  # No valid pairs

    # Convert to heterogeneous graphs
    hetero_graphs = [Preprocessing.gnn_graph.convert_to_hetero_data(graph) for graph in graphs]

    total_correct = 0
    total_pairs = 0

    with torch.no_grad():
        # Iterate over consecutive pairs (graph[0] vs graph[1], then graph[1] vs graph[2], etc.)
        for i in range(len(graphs) - 1):
            graph1, graph2 = hetero_graphs[i], hetero_graphs[i + 1]
            gt1, gt2 = gt_scores[i], gt_scores[i + 1]

            # Forward pass through the models
            x_dict1 = graph_encoder(graph1.x_dict, graph1.edge_index_dict)
            x_dict2 = graph_encoder(graph2.x_dict, graph2.edge_index_dict)

            pred1 = graph_decoder(x_dict1).argmax(dim=-1)  # Predicted bin for graph1
            pred2 = graph_decoder(x_dict2).argmax(dim=-1)  # Predicted bin for graph2

            # print("\n---  ---")
            # print("pred1:", pred1, "gt1", gt1)
            # print("pred2:", pred2, "gt2", gt2)

            # Check if the predicted ranking matches the ground truth ranking
            if (pred1 < pred2 and gt1 < gt2) or (pred1 >= pred2 and gt1 >= gt2):
                total_correct += 1

            total_pairs += 1
    
    print(f"Total Correct: {total_correct}, Total Pairs: {total_pairs}")
    return total_correct, total_pairs
# ------------------------------------------------------------------------------# 


def train():
    dataset = whole_process_evaluate.Evaluation_Dataset('program_output_dataset')
    total_samples = len(dataset)
    chunk_size = 500
    epochs_per_chunk = 20

    best_val_loss = float('inf')

    past_particle_value = 0
    prev_bin_score = 0
    bins = calculate_bins_with_min_score()

    print("total_samples",total_samples)

    # Loop over dataset in chunks
    for chunk_start in range(0, total_samples, chunk_size):
        chunk_end = min(chunk_start + chunk_size, total_samples)
        print(f"Processing graphs {chunk_start} to {chunk_end}...")

        graphs = []
        gt_state_value = []
        
        # Process one chunk of graphs
        for idx in tqdm(range(chunk_start, chunk_end), desc="Preprocessing graphs"):
            data = dataset[idx]
            (particle_value, stroke_node_features, output_brep_edges, gt_brep_edges,
             stroke_cloud_loops, strokes_perpendicular, loop_neighboring_vertical,
             loop_neighboring_horizontal, loop_neighboring_contained, stroke_to_loop,
             stroke_to_edge, is_all_edges_used) = data

        
            cur_bin_score = compute_bin_score(torch.tensor(particle_value, dtype=torch.float32), bins)

            if not is_all_edges_used or cur_bin_score == prev_bin_score:
                continue
            
            prev_bin_score = cur_bin_score
            
            past_particle_value = particle_value
            # Build the graph
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

            # print("cur_bin_score", cur_bin_score)        
            # Encoders.helper.vis_left_graph(gnn_graph['stroke'].x.cpu().numpy())
            # Encoders.helper.vis_brep(output_brep_edges.cpu().numpy())


            gnn_graph.to_device_withPadding(device)
            graphs.append(gnn_graph)

            # Convert the target value to a tensor
            particle_value_tensor = torch.tensor(particle_value, dtype=torch.float32)
            binned_score = compute_bin_score(particle_value_tensor, bins).to(device)
            gt_state_value.append(binned_score)

        print(f"Loaded {len(graphs)} graphs from {chunk_start} to {chunk_end}.")

        # Split the chunk into training and validation sets (80-20 split)
        split_index = int(0.8 * len(graphs))
        train_graphs, val_graphs = graphs[:split_index], graphs[split_index:]
        train_scores, val_scores = gt_state_value[:split_index], gt_state_value[split_index:]

        # Convert graphs to heterogeneous data format
        hetero_train_graphs = [Preprocessing.gnn_graph.convert_to_hetero_data(graph) for graph in train_graphs]
        hetero_val_graphs = [Preprocessing.gnn_graph.convert_to_hetero_data(graph) for graph in val_graphs]

        # Create DataLoaders for the current chunk
        graph_train_loader = DataLoader(hetero_train_graphs, batch_size=16, shuffle=True)
        score_train_loader = DataLoader(train_scores, batch_size=16, shuffle=True)
        graph_val_loader = DataLoader(hetero_val_graphs, batch_size=16, shuffle=False)
        score_val_loader = DataLoader(val_scores, batch_size=16, shuffle=False)


        # Train on the current chunk for a fixed number of epochs
        for epoch in range(epochs_per_chunk):
            train_loss = 0.0
            total_correct = 0
            total_samples_batch = 0

            graph_encoder.train()
            graph_decoder.train()

            total_iterations = min(len(graph_train_loader), len(score_train_loader))
            # Training loop for this chunk
            for hetero_batch, batch_scores in tqdm(zip(graph_train_loader, score_train_loader), 
                                                   desc=f"Epoch {epoch+1}/{epochs_per_chunk} - Training",
                                                   dynamic_ncols=True, total=total_iterations):
                optimizer.zero_grad()
                x_dict = graph_encoder(hetero_batch.x_dict, hetero_batch.edge_index_dict)
                output = graph_decoder(x_dict, hetero_batch)

                loss = criterion(output, batch_scores)
                loss.backward()
                optimizer.step()

                train_loss += loss.item()
                correct, total = compute_accuracy(output, batch_scores)
            
                # print("Predicted:", output)  # Get class predictions
                print("Predicted:", output.argmax(dim=-1).cpu().numpy())  # Get class predictions
                print("Ground Truth:", batch_scores.cpu().numpy())
                print("-------------")


                total_correct += correct
                total_samples_batch += total

            train_accuracy = total_correct / total_samples_batch
            train_loss = train_loss / total_samples_batch
            print(f"Epoch {epoch+1}/{epochs_per_chunk} - Training Loss: {train_loss:.4f}, Accuracy: {train_accuracy:.4%}")

            # Validation loop for this chunk
            graph_encoder.eval()
            graph_decoder.eval()
            val_loss = 0.0
            val_correct = 0
            val_samples_batch = 0

            with torch.no_grad():
                for hetero_batch, batch_scores in tqdm(zip(graph_val_loader, score_val_loader), 
                                                       desc=f"Epoch {epoch+1}/{epochs_per_chunk} - Validation",
                                                       dynamic_ncols=True, total=len(graph_val_loader)):
                    x_dict = graph_encoder(hetero_batch.x_dict, hetero_batch.edge_index_dict)
                    output = graph_decoder(x_dict, hetero_batch)

                    loss = criterion(output, batch_scores)
                    val_loss += loss.item()

                    correct, total = compute_accuracy(output, batch_scores)
                    val_correct += correct
                    val_samples_batch += total

            val_accuracy = val_correct / val_samples_batch
            val_loss = val_loss / val_samples_batch
            print(f"Epoch {epoch+1}/{epochs_per_chunk} - Validation Loss: {val_loss:.4f}, Accuracy: {val_accuracy:.4%}")

            # Save model if validation improves
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                save_models()
                print("Best model saved for current chunk.")

        print(f"Finished training chunk {chunk_start} to {chunk_end}.")

        # Free up memory from this chunk
        del graphs, gt_state_value, train_graphs, val_graphs
        del hetero_train_graphs, hetero_val_graphs
        del graph_train_loader, score_train_loader, graph_val_loader, score_val_loader
        torch.cuda.empty_cache()



def eval():
    dataset = whole_process_evaluate.Evaluation_Dataset('program_output_dataset')
    total_samples = len(dataset)
    chunk_size = 100  # Smaller chunk size for evaluation
    batch_size = 1
    prev_bin_score = 0

    bins = calculate_bins_with_min_score()

    print("total_samples", total_samples)

    load_models()  
    graph_encoder.eval()
    graph_decoder.eval()

    total_correct = 0
    total_samples_batch = 0
    total_loss = 0.0

    with torch.no_grad():  # Ensure no gradients are computed
        for chunk_start in range(0, total_samples, chunk_size):
            chunk_end = min(chunk_start + chunk_size, total_samples)
            print(f"Processing graphs {chunk_start} to {chunk_end}...")

            graphs = []
            gt_state_value = []

            # Process one chunk of graphs
            for idx in tqdm(range(chunk_start, chunk_end), desc="Preprocessing graphs"):
                data = dataset[idx]
                (particle_value, stroke_node_features, output_brep_edges, gt_brep_edges,
                 stroke_cloud_loops, strokes_perpendicular, loop_neighboring_vertical,
                 loop_neighboring_horizontal, loop_neighboring_contained, stroke_to_loop,
                 stroke_to_edge, is_all_edges_used) = data

                cur_bin_score = compute_bin_score(torch.tensor(particle_value, dtype=torch.float32), bins)

                if not is_all_edges_used or cur_bin_score == prev_bin_score:
                    continue
                
                prev_bin_score = cur_bin_score

                # Build the graph
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

                # Convert the target value to a tensor (bin classification)
                particle_value_tensor = torch.tensor(particle_value, dtype=torch.float32)
                binned_score = compute_bin_score(particle_value_tensor, bins).to(device)
                gt_state_value.append(torch.tensor(binned_score, dtype=torch.long).to(device))  # Convert to LongTensor for CrossEntropyLoss

            print(f"Loaded {len(graphs)} graphs from {chunk_start} to {chunk_end}.")

            # Convert graphs to heterogeneous data format
            hetero_graphs = [Preprocessing.gnn_graph.convert_to_hetero_data(graph) for graph in graphs]

            # Create DataLoaders for the current chunk
            graph_loader = DataLoader(hetero_graphs, batch_size=batch_size, shuffle=False)
            score_loader = DataLoader(gt_state_value, batch_size=batch_size, shuffle=False)

            # Evaluation loop
            for hetero_batch, batch_scores in tqdm(zip(graph_loader, score_loader), 
                                                   desc=f"Evaluating {chunk_start}-{chunk_end}",
                                                   dynamic_ncols=True, total=len(graph_loader)):

                x_dict = graph_encoder(hetero_batch.x_dict, hetero_batch.edge_index_dict)
                output = graph_decoder(x_dict)  # Shape: [batch_size, num_classes]

                loss = criterion(output, batch_scores)
                total_loss += loss.item()

                correct, total = compute_accuracy(output, batch_scores)
                total_correct += correct
                total_samples_batch += total

                # Print predictions and ground truth for debugging
                print("Predicted:", output.argmax(dim=-1).cpu().numpy())  # Get class predictions
                print("Ground Truth:", batch_scores.cpu().numpy())

    # Compute overall accuracy and loss
    final_accuracy = total_correct / total_samples_batch
    avg_loss = total_loss / total_samples_batch

    print(f"Evaluation Loss: {avg_loss:.4f}, Accuracy: {final_accuracy:.4%}")


def eval_contrastive():
    dataset = whole_process_evaluate.Evaluation_Dataset('program_output_dataset')
    total_samples = len(dataset)
    batch_size = 4  # Each batch should contain at least 2 graphs for contrastive evaluation
    bins = calculate_bins_with_min_score()

    print("total_samples", total_samples)

    load_models()
    graph_encoder.eval()
    graph_decoder.eval()

    total_correct = 0
    total_pairs = 0
    prev_bin_score = 0

    graphs = []
    gt_scores = []
    prev_stroke_count = None

    with torch.no_grad():
        print("Processing graphs in dynamic chunks...")

        for idx in tqdm(range(total_samples), desc="Preprocessing graphs"):
            data = dataset[idx]
            (particle_value, stroke_node_features, output_brep_edges, gt_brep_edges,
             stroke_cloud_loops, strokes_perpendicular, loop_neighboring_vertical,
             loop_neighboring_horizontal, loop_neighboring_contained, stroke_to_loop,
             stroke_to_edge, is_all_edges_used) = data

            cur_bin_score = compute_bin_score(torch.tensor(particle_value, dtype=torch.float32), bins)

            if not is_all_edges_used or cur_bin_score == prev_bin_score:
                continue

            
            prev_bin_score = cur_bin_score
            stroke_count = len(stroke_node_features)  # Category identifier

            # Start a new chunk if:
            # - We reached 1000 graphs
            # - The stroke count is different from the previous one
            if len(graphs) >= 50 or (prev_stroke_count is not None and stroke_count != prev_stroke_count):
                print(f"Processing batch with {len(graphs)} graphs (Stroke Count: {prev_stroke_count})")
                correct, pairs = process_contrastive_batch(graphs, gt_scores)

                total_correct += correct
                total_pairs += pairs

                # Reset buffers for next batch
                graphs = []
                gt_scores = []

            # Store the current graph and ground truth bin score
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

            # Compute binned ground truth score
            binned_score = torch.tensor(compute_bin_score(torch.tensor(particle_value, dtype=torch.float32), bins), dtype=torch.long).to(device)
            gt_scores.append(binned_score)

            prev_stroke_count = stroke_count  # Track category for batching

        # Process the final batch
        if graphs:
            print(f"Processing final batch with {len(graphs)} graphs (Stroke Count: {prev_stroke_count})")
            correct, pairs = process_contrastive_batch(graphs, gt_scores, batch_size, bins)
            total_correct += correct
            total_pairs += pairs

    # Compute overall contrastive accuracy
    final_accuracy = total_correct / total_pairs if total_pairs > 0 else 0

    print(f"Contrastive Evaluation Accuracy: {final_accuracy:.4%}")



# ------------------------------------------------------------------------------# 



train()
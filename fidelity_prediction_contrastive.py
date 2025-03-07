from torch.utils.data import Dataset, DataLoader
import os
import pickle
from tqdm import tqdm
import torch
import numpy as np
import torch.nn.functional as F

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

criterion = nn.MSELoss()
optimizer = optim.Adam(list(graph_encoder.parameters()) + list(graph_decoder.parameters()), lr=0.0004)
batch_size = 16

# ------------------------------------------------------------------------------# 

current_dir = os.getcwd()
save_dir = os.path.join(current_dir, 'checkpoints', 'fidelity_prediction_contrastive')
os.makedirs(save_dir, exist_ok=True)

def load_models():
    graph_encoder.load_state_dict(torch.load(os.path.join(save_dir, 'graph_encoder.pth')))
    graph_decoder.load_state_dict(torch.load(os.path.join(save_dir, 'graph_decoder.pth')))
    print("Models loaded successfully!")  


def save_models():
    torch.save(graph_encoder.state_dict(), os.path.join(save_dir, 'graph_encoder.pth'))
    torch.save(graph_decoder.state_dict(), os.path.join(save_dir, 'graph_decoder.pth'))
    print("Models saved successfully!")  

# ------------------------------------------------------------------------------# 

def contrastive_loss(pred1, pred2, gt1, gt2, margin=0.05):
    """
    Compute contrastive loss for value regression.
    Enforces ranking order: If gt1 < gt2, then pred1 < pred2.
    """
    target = torch.sign(gt2 - gt1)  # +1 if gt2 > gt1, -1 otherwise
    return F.margin_ranking_loss(pred1, pred2, target, margin=margin)


def compute_accuracy(output, target, tol=0.1):
    """
    Computes accuracy for regression outputs by counting a prediction as correct if
    its absolute difference from the target is less than tol.
    
    Parameters:
    - output: torch.Tensor of predictions
    - target: torch.Tensor of ground truth values
    - tol: float, tolerance threshold
    
    Returns:
    - correct: int, number of predictions within tolerance
    - total: int, total number of predictions
    """
    # Flatten in case tensors are not 1D
    output = output.view(-1)
    target = target.view(-1)
    
    # Count predictions that are within the tolerance
    correct = (torch.abs(output - target) < tol).sum().item()
    total = target.numel()
    
    return correct, total


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

            pred1 = graph_decoder(x_dict1, graph1)
            pred2 = graph_decoder(x_dict2, graph2)

            # Check if the predicted ranking matches the ground truth ranking
            if (pred1 < pred2 and gt1 < gt2) or (pred1 >= pred2 and gt1 >= gt2):
                total_correct += 1
            
            print("\n---  ---")
            print("pred1:", pred1[0], "gt1", gt1)
            print("pred2:", pred2[0], "gt2", gt2)

            total_pairs += 1
    
    print(f"Total Correct: {total_correct}, Total Pairs: {total_pairs}")
    return total_correct, total_pairs

# ------------------------------------------------------------------------------# 




def train():
    dataset = whole_process_evaluate.Evaluation_Dataset('program_output_dataset')
    total_samples = len(dataset)
    epochs_per_batch = 20

    best_val_loss = float('inf')

    print("total_samples", total_samples)

    graphs = []
    gt_state_values = []
    prev_stroke_count = None

    past_value = 0

    for idx in tqdm(range(total_samples), desc="Preprocessing graphs"):
        data = dataset[idx]
        (particle_value, stroke_node_features, output_brep_edges, gt_brep_edges,
         stroke_cloud_loops, strokes_perpendicular, loop_neighboring_vertical,
         loop_neighboring_horizontal, loop_neighboring_contained, stroke_to_loop,
         stroke_to_edge, is_all_edges_used) = data

        stroke_count = len(stroke_node_features)  # Category identifier

        if not is_all_edges_used or past_value == particle_value:
            continue

        # Start new batch if stroke count changes or batch size exceeds 20
        if (prev_stroke_count is not None and prev_stroke_count != stroke_count) or len(graphs) > 20:
            print(f"Processing batch with {len(graphs)} graphs (Stroke Count: {prev_stroke_count})")

            # Convert graphs to heterogeneous data format
            hetero_graphs = [Preprocessing.gnn_graph.convert_to_hetero_data(graph) for graph in graphs]

            # Create DataLoader
            graph_loader = DataLoader(hetero_graphs, batch_size=16, shuffle=True)
            score_loader = DataLoader(gt_state_values, batch_size=16, shuffle=True)

            for epoch in range(epochs_per_batch):
                total_loss = 0.0
                total_correct = 0
                total_pairs = 0

                graph_encoder.train()
                graph_decoder.train()

                for hetero_batch, batch_scores in tqdm(zip(graph_loader, score_loader), 
                                                       desc=f"Epoch {epoch+1}/{epochs_per_batch} - Contrastive Training",
                                                       dynamic_ncols=True, total=len(graph_loader)):

                    optimizer.zero_grad()
                    x_dict = graph_encoder(hetero_batch.x_dict, hetero_batch.edge_index_dict)
                    output = graph_decoder(x_dict, hetero_batch)  # Shape: [batch_size, 1]
                    predicted_values = output.squeeze(-1)  # Ensure it's a 1D tensor

                    batch_size = predicted_values.shape[0]
                    loss = 0.0
                    pair_count = 0

                    # Iterate over consecutive pairs for contrastive learning
                    for i in range(batch_size - 1):
                        pred1, pred2 = predicted_values[i], predicted_values[i + 1]
                        gt1, gt2 = batch_scores[i], batch_scores[i + 1]

                        # Apply contrastive loss if GT values are different
                        if gt1 != gt2:
                            target = torch.sign(gt1 - gt2).float()  # Ensure correct order

                            loss += F.margin_ranking_loss(pred1, pred2, target, margin=0.05)
                            pair_count += 1

                            # print("pred1", pred1, "gt1", gt1)
                            # print("pred2", pred2, "gt2", gt2)
                            # print("----------------")

                            # **Compute Accuracy**: Check if ordering is correct
                            correct = (target > 0 and pred1 >= pred2) or (target < 0 and pred1 <= pred2)
                            if correct:
                                total_correct += 1
                            total_pairs += 1

                    # Average the loss across all pairs
                    if pair_count > 0:
                        loss = loss / pair_count
                        loss.backward()
                        optimizer.step()

                    total_loss += loss.item()

                avg_loss = total_loss / len(graph_loader)
                accuracy = total_correct / total_pairs if total_pairs > 0 else 0  # Avoid division by zero
                print(f"Epoch {epoch+1}/{epochs_per_batch} - Contrastive Loss: {avg_loss:.4f}, Accuracy: {accuracy:.4%}")

                # Save model if validation improves
                if avg_loss < best_val_loss:
                    best_val_loss = avg_loss
                    save_models()
                    print("Best model saved for current batch.")

            # Free up memory
            del graphs, gt_state_values, hetero_graphs, graph_loader, score_loader
            torch.cuda.empty_cache()

            graphs, gt_state_values = [], []  # Reset for next batch

        prev_stroke_count = stroke_count  # Track category
        past_value = particle_value

        # Build the graph
        gnn_graph = Preprocessing.gnn_graph.SketchLoopGraph(
            stroke_cloud_loops, stroke_node_features, strokes_perpendicular,
            loop_neighboring_vertical, loop_neighboring_horizontal, loop_neighboring_contained,
            stroke_to_loop, stroke_to_edge
        )
        gnn_graph.to_device_withPadding(device)
        graphs.append(gnn_graph)

        # Store GT value for regression
        gt_state_values.append(torch.tensor(particle_value, dtype=torch.float32).to(device))




def eval():
    dataset = whole_process_evaluate.Evaluation_Dataset('program_output_dataset')
    total_samples = len(dataset)
    chunk_size = 1000

    batch_size = 1

    best_val_loss = float('inf')

    past_particle_value = 0

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

        
            if not is_all_edges_used or particle_value == past_particle_value:
                continue
            
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

        
            # if particle_value != past_particle_value:
            #     past_particle_value = particle_value
            #     Encoders.helper.vis_left_graph(gnn_graph['stroke'].x.cpu().numpy())
            #     Encoders.helper.vis_brep(output_brep_edges.cpu().numpy())


            gnn_graph.to_device_withPadding(device)
            graphs.append(gnn_graph)

            # Convert the target value to a tensor
            particle_value_tensor = torch.tensor(particle_value, dtype=torch.float32).to(device)
            gt_state_value.append(particle_value_tensor)

        print(f"Loaded {len(graphs)} graphs from {chunk_start} to {chunk_end}.")

        # Split the chunk into training and validation sets (80-20 split)

        # Convert graphs to heterogeneous data format
        hetero_train_graphs = [Preprocessing.gnn_graph.convert_to_hetero_data(graph) for graph in graphs]

        # Create DataLoaders for the current chunk
        graph_train_loader = DataLoader(hetero_train_graphs, batch_size=1, shuffle=True)
        score_train_loader = DataLoader(gt_state_value, batch_size=1, shuffle=True)


        # Train on the current chunk for a fixed number of epochs
        for epoch in range(1):
            train_loss = 0.0
            total_correct = 0
            total_samples_batch = 0

            graph_encoder.eval()
            graph_decoder.eval()

            total_iterations = min(len(graph_train_loader), len(score_train_loader))
            # Training loop for this chunk
            for hetero_batch, batch_scores in tqdm(zip(graph_train_loader, score_train_loader), 
                                                   dynamic_ncols=True, total=total_iterations):
                optimizer.zero_grad()
                x_dict = graph_encoder(hetero_batch.x_dict, hetero_batch.edge_index_dict)
                output = graph_decoder(x_dict)

                loss = criterion(output, batch_scores)
                loss.backward()
                optimizer.step()

                print("output", output[0])
                print("batch_scores", batch_scores[0])
                Encoders.helper.vis_left_graph(hetero_batch[0]['stroke'].x.cpu().numpy())

                train_loss += loss.item()
                correct, total = compute_accuracy(output, batch_scores)
                total_correct += correct
                total_samples_batch += total

            train_accuracy = total_correct / total_samples_batch
            train_loss = train_loss / total_samples_batch
            print(f"Epoch {epoch+1}/{epochs_per_chunk} - Training Loss: {train_loss:.4f}, Accuracy: {train_accuracy:.4%}")



def eval_contrastive():
    dataset = whole_process_evaluate.Evaluation_Dataset('program_output_dataset')
    total_samples = len(dataset)
    batch_size = 4  # Each batch should contain at least 2 graphs for contrastive evaluation

    print("total_samples", total_samples)

    load_models()
    graph_encoder.eval()
    graph_decoder.eval()

    total_correct = 0
    total_pairs = 0
    past_particle_value = 0

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

            if not is_all_edges_used or particle_value == past_particle_value:
                continue

            
            past_particle_value = particle_value
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
            particle_value_tensor = torch.tensor(particle_value, dtype=torch.float32).to(device)
            gt_scores.append(particle_value_tensor)

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
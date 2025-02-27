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

criterion = nn.MSELoss()
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

# ------------------------------------------------------------------------------# 



def train():
    dataset = whole_process_evaluate.Evaluation_Dataset('program_output_dataset')
    total_samples = len(dataset)
    chunk_size = 1000
    epochs_per_chunk = 10

    best_accuracy = 0.0
    past_particle_value = 0

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
                output = graph_decoder(x_dict)

                loss = criterion(output, batch_scores)
                loss.backward()
                optimizer.step()

                train_loss += loss.item()
                correct, total = compute_accuracy(output, batch_scores)
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
                    output = graph_decoder(x_dict)

                    loss = criterion(output, batch_scores)
                    val_loss += loss.item()

                    correct, total = compute_accuracy(output, batch_scores)
                    val_correct += correct
                    val_samples_batch += total

            val_accuracy = val_correct / val_samples_batch
            val_loss = val_loss / val_samples_batch
            print(f"Epoch {epoch+1}/{epochs_per_chunk} - Validation Loss: {val_loss:.4f}, Accuracy: {val_accuracy:.4%}")

            # Save model if validation improves
            if val_accuracy > best_accuracy:
                best_accuracy = val_accuracy
                save_models()
                print("Best model saved for current chunk.")

        print(f"Finished training chunk {chunk_start} to {chunk_end}.")

        # Free up memory from this chunk
        del graphs, gt_state_value, train_graphs, val_graphs
        del hetero_train_graphs, hetero_val_graphs
        del graph_train_loader, score_train_loader, graph_val_loader, score_val_loader
        torch.cuda.empty_cache()



# ------------------------------------------------------------------------------# 



train()
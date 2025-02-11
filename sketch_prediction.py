import Preprocessing.dataloader
import Preprocessing.gnn_graph

import Encoders.gnn.gnn
import Encoders.helper

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
graph_decoder = Encoders.gnn.gnn.Sketch_Decoder()

graph_encoder.to(device)
graph_decoder.to(device)

criterion = Encoders.gnn.gnn.FocalLoss(alpha=0.75, gamma=2.5)
optimizer = optim.Adam(list(graph_encoder.parameters()) + list(graph_decoder.parameters()), lr=0.0004)

# ------------------------------------------------------------------------------# 

current_dir = os.getcwd()
save_dir = os.path.join(current_dir, 'checkpoints', 'sketch_prediction')
os.makedirs(save_dir, exist_ok=True)

def load_models():
    graph_encoder.load_state_dict(torch.load(os.path.join(save_dir, 'graph_encoder.pth')))
    graph_decoder.load_state_dict(torch.load(os.path.join(save_dir, 'graph_decoder.pth')))


def save_models():
    torch.save(graph_encoder.state_dict(), os.path.join(save_dir, 'graph_encoder.pth'))
    torch.save(graph_decoder.state_dict(), os.path.join(save_dir, 'graph_decoder.pth'))


# ------------------------------------------------------------------------------# 


def compute_accuracy(valid_output, valid_batch_masks):
    batch_size = valid_output.shape[0] // 400
    correct = 0

    for i in range(batch_size):
        output_slice = valid_output[i * 400:(i + 1) * 400]
        mask_slice = valid_batch_masks[i * 400:(i + 1) * 400]

        max_output_value, max_output_index = torch.max(output_slice, dim=0)
        max_mask_value, max_mask_index = torch.max(mask_slice, dim=0)

        values_where_mask_is_1 = output_slice[mask_slice == 1]
        indices_where_mask_is_1 = torch.nonzero(mask_slice == 1).squeeze()

        # print(f"Graph {i}: Values where mask is 1: {values_where_mask_is_1.tolist()}, Indices: {indices_where_mask_is_1.tolist()}")
        # print(f"Graph {i}: max_output_index={max_output_index.item()}, max_mask_index={max_mask_index.item()}")

        if max_output_index == max_mask_index:
            correct += 1

    return correct


def compute_accuracy_with_lvl(valid_output, valid_batch_masks, hetero_batch):
    # Infer batch size and sequence length from the shapes
    batch_size = valid_output.shape[0] // 400

    # Initialize counters for each category
    category_count = [0, 0, 0, 0]  # Tracks the number of batches in each category
    correct_count = [0, 0, 0, 0]   # Tracks the correct predictions in each category

    for i in range(batch_size):
        # Slice for each example in the batch

        output_slice = valid_output[i * 400:(i + 1) * 400]
        mask_slice = valid_batch_masks[i * 400:(i + 1) * 400]
        
        stroke_node_features_slice = hetero_batch.x_dict['stroke'][i * 400:(i + 1) * 400]
        edge_features = hetero_batch.edge_index_dict['stroke', 'represents', 'loop']
        edge_features_slice = Encoders.helper.find_edge_features_slice(edge_features, i)


        # Find k: the number of rows where all elements are not -1
        k = 0
        for row in stroke_node_features_slice:
            if not torch.all(row == -1):
                k += 1
            else:
                break

        # Determine category based on k
        if k < 50:
            category_idx = 0  # Category 1
        elif 50 <= k < 70:
            category_idx = 1  # Category 2
        elif 70 <= k < 90:
            category_idx = 2  # Category 3
        else:
            category_idx = 3  # Category 4

        # Increment the batch count for the category
        category_count[category_idx] += 1

        # Get the index of the maximum value for both the output and mask
        _, max_output_index = torch.max(output_slice, dim=0)
        gt_indices = torch.nonzero(mask_slice > 0)[:, 0].tolist()

        # Check if the prediction is correct and increment the correct counter for the category
        if max_output_index.item() in gt_indices:
            correct_count[category_idx] += 1
        else:
            pass
            # Encoders.helper.vis_selected_loops(stroke_node_features_slice.cpu().numpy(), edge_features_slice, [max_output_index.item()])
            # Encoders.helper.vis_selected_loops(stroke_node_features_slice.cpu().numpy(), edge_features_slice, gt_indices)

    return category_count, correct_count


# ------------------------------------------------------------------------------# 


def train():
    # Load the dataset
    dataset = Preprocessing.dataloader.Program_Graph_Dataset('dataset/whole')
    print(f"Total number of shape data: {len(dataset)}")
    
    best_val_accuracy = 0
    epochs = 30
    
    graphs = []
    loop_selection_masks = []

    # Preprocess and build the graphs
    for data in tqdm(dataset, desc=f"Building Graphs"):
        # Extract the necessary elements from the dataset
        program, program_whole, stroke_cloud_loops, stroke_node_features, strokes_perpendicular, output_brep_edges, stroke_operations_order_matrix, loop_neighboring_vertical, loop_neighboring_horizontal,loop_neighboring_contained, stroke_to_loop, stroke_to_edge = data
        if program[-1] != 'sketch':
            continue

        kth_operation = Encoders.helper.get_kth_operation(stroke_operations_order_matrix, len(program)-1)
        chosen_strokes = (kth_operation == 1).nonzero(as_tuple=True)[0]  # Indices of chosen strokes
        
        loop_chosen_mask = []
        for loop in stroke_cloud_loops:
            if all(stroke in chosen_strokes for stroke in loop):
                loop_chosen_mask.append(1)  # Loop is chosen
            else:
                loop_chosen_mask.append(0)  # Loop is not chosen
        
        loop_selection_mask = torch.tensor(loop_chosen_mask, dtype=torch.float).reshape(-1, 1)
        if not (loop_selection_mask == 1).any():
            continue
        

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
        loop_selection_mask = loop_selection_mask.to(device)

        # Encoders.helper.vis_stroke_with_order(stroke_node_features)
        # Encoders.helper.vis_brep(output_brep_edges)
        # Encoders.helper.vis_selected_strokes(gnn_graph['stroke'].x.cpu().numpy(),chosen_strokes)

        # Prepare the pair
        graphs.append(gnn_graph)
        loop_selection_masks.append(loop_selection_mask)


    print(f"Total number of preprocessed graphs: {len(graphs)}")
    # Split the dataset into training and validation sets (80-20 split)
    split_index = int(0.8 * len(graphs))
    train_graphs, val_graphs = graphs[:split_index], graphs[split_index:]
    train_masks, val_masks = loop_selection_masks[:split_index], loop_selection_masks[split_index:]


    # Convert train and validation graphs to HeteroData
    hetero_train_graphs = [Preprocessing.gnn_graph.convert_to_hetero_data(graph) for graph in train_graphs]
    padded_train_masks = [Preprocessing.dataloader.pad_masks(mask) for mask in train_masks]

    hetero_val_graphs = [Preprocessing.gnn_graph.convert_to_hetero_data(graph) for graph in val_graphs]
    padded_val_masks = [Preprocessing.dataloader.pad_masks(mask) for mask in val_masks]

    # Create DataLoaders for training and validation graphs/masks
    graph_train_loader = DataLoader(hetero_train_graphs, batch_size=16, shuffle=False)
    mask_train_loader = DataLoader(padded_train_masks, batch_size=16, shuffle=False)

    graph_val_loader = DataLoader(hetero_val_graphs, batch_size=16, shuffle=False)
    mask_val_loader = DataLoader(padded_val_masks, batch_size=16, shuffle=False)



    # Training loop
    for epoch in range(30):
        train_loss = 0.0
        graph_encoder.train()
        graph_decoder.train()
        train_correct = 0
        train_total = 0


        total_iterations = min(len(graph_train_loader), len(mask_train_loader))
        for hetero_batch, batch_masks in tqdm(zip(graph_train_loader, mask_train_loader), 
                                              desc=f"Epoch {epoch+1}/{epochs} - Training", 
                                              dynamic_ncols=True, 
                                              total=total_iterations):

            optimizer.zero_grad()

            x_dict = graph_encoder(hetero_batch.x_dict, hetero_batch.edge_index_dict)
            output = graph_decoder(x_dict)

            batch_masks = batch_masks.to(output.device).view(-1, 1)
            valid_mask = (batch_masks != -1).float()
            valid_output = output * valid_mask
            valid_batch_masks = batch_masks * valid_mask
            loss = criterion(valid_output, valid_batch_masks)

            train_correct += compute_accuracy(valid_output, valid_batch_masks)
            train_total += 16


            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        train_accuracy = train_correct / train_total if train_total > 0 else 0
        train_loss /= len(train_graphs)
        val_loss = 0.0
        correct = 0
        total = 0

        graph_encoder.eval()
        graph_decoder.eval()
        with torch.no_grad():
            total_iterations_val = min(len(graph_val_loader), len(mask_val_loader))

            for hetero_batch, batch_masks in tqdm(zip(graph_val_loader, mask_val_loader), 
                                                  desc="Validation", 
                                                  dynamic_ncols=True, 
                                                  total=total_iterations_val):
                x_dict = graph_encoder(hetero_batch.x_dict, hetero_batch.edge_index_dict)
                output = graph_decoder(x_dict)

                batch_masks = batch_masks.to(output.device).view(-1, 1)
                valid_mask = (batch_masks != -1).float()
                valid_output = output * valid_mask
                valid_batch_masks = batch_masks * valid_mask
                loss = criterion(valid_output, valid_batch_masks)

                correct += compute_accuracy(valid_output, valid_batch_masks)
                total += 16

        
        val_loss /= len(val_graphs)
        

        accuracy = correct / total if total > 0 else 0
        print(f"Epoch {epoch+1}/{epochs} - Training Loss: {train_loss:.7f} - Validation Loss: {val_loss:.7f} - Train Accuracy: {train_accuracy:.5f} - Validation Accuracy: {accuracy:.5f}")

        if accuracy > best_val_accuracy:
            best_val_accuracy = accuracy
            save_models()
            print(f"Models saved at epoch {epoch+1} with validation accuracy: {accuracy:.5f}")



def eval():
    # load_models()
    # Load the dataset
    dataset = Preprocessing.dataloader.Program_Graph_Dataset('dataset/eval')
    print(f"Total number of shape data: {len(dataset)}")


    eval_graphs = []
    eval_loop_selection_masks = []
    eval_all_loop_selection_masks = []

    # Preprocess and build the graphs
    for data in tqdm(dataset, desc=f"Building Graphs"):
        # Extract the necessary elements from the dataset
        program, program_whole, stroke_cloud_loops, stroke_node_features, strokes_perpendicular, output_brep_edges, stroke_operations_order_matrix, loop_neighboring_vertical, loop_neighboring_horizontal,loop_neighboring_contained, stroke_to_loop, stroke_to_edge = data

        if program[-1] != 'sketch':
            continue

        kth_operation = Encoders.helper.get_kth_operation(stroke_operations_order_matrix, len(program)-1)
        all_sketch_strokes = Encoders.helper.get_all_operation_strokes(stroke_operations_order_matrix, program_whole, 'sketch')

        # Gets the strokes for the current sketch Operation
        chosen_strokes = (kth_operation == 1).nonzero(as_tuple=True)[0]  # Indices of chosen stroke
        loop_chosen_mask = []
        for loop in stroke_cloud_loops:
            if all(stroke in chosen_strokes for stroke in loop):
                loop_chosen_mask.append(1)  # Loop is chosen
            else:
                loop_chosen_mask.append(0)  # Loop is not chosen
        
        loop_selection_mask = torch.tensor(loop_chosen_mask, dtype=torch.float).reshape(-1, 1)
        if not (loop_selection_mask == 1).any():
            continue

        
        # Gets the strokes for all sketch Operation
        all_chosen_strokes = (all_sketch_strokes == 1).nonzero(as_tuple=True)[0]  # Indices of chosen stroke
        all_loop_chosen_mask = []
        for loop in stroke_cloud_loops:
            if all(stroke in all_chosen_strokes for stroke in loop):
                all_loop_chosen_mask.append(1)  # Loop is chosen
            else:
                all_loop_chosen_mask.append(0)  # Loop is not chosen
        all_loop_selection_mask = torch.tensor(all_loop_chosen_mask, dtype=torch.float).reshape(-1, 1)

        
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

        # Encoders.helper.vis_brep(final_brep_edges)
        all_selected_loops_idx = [idx for idx, value in enumerate(all_loop_chosen_mask) if value != 0]

        # Encoders.helper.vis_selected_strokes(gnn_graph['stroke'].x.cpu().numpy(), chosen_strokes )

        # Prepare the pair
        gnn_graph.to_device_withPadding(device)
        loop_selection_mask = loop_selection_mask.to(device)
        
        eval_graphs.append(gnn_graph)
        eval_loop_selection_masks.append(loop_selection_mask)
        eval_all_loop_selection_masks.append(all_loop_selection_mask)


    print(f"Total number of preprocessed graphs: {len(eval_graphs)}")


    # Convert train and validation graphs to HeteroData
    hetero_eval_graphs = [Preprocessing.gnn_graph.convert_to_hetero_data(graph) for graph in eval_graphs]
    padded_eval_masks = [Preprocessing.dataloader.pad_masks(mask) for mask in eval_loop_selection_masks]
    padded_eval_all_masks = [Preprocessing.dataloader.pad_masks(mask) for mask in eval_all_loop_selection_masks]

    # Create DataLoaders for training and validation graphs/masks
    graph_eval_loader = DataLoader(hetero_eval_graphs, batch_size=16, shuffle=False)
    mask_eval_loader = DataLoader(padded_eval_masks, batch_size=16, shuffle=False)
    mask_eval_all_loader = DataLoader(padded_eval_all_masks, batch_size=16, shuffle=False)



    # Eval
    graph_encoder.eval()
    graph_decoder.eval()

    eval_loss = 0.0
    total_category_count = [0, 0, 0, 0]
    total_correct_count = [0, 0, 0, 0] 

    with torch.no_grad():
        total_iterations_eval = min(len(graph_eval_loader), len(mask_eval_all_loader))

        for hetero_batch, batch_masks, in tqdm(zip(graph_eval_loader, mask_eval_all_loader), 
                                                desc="Evaluation", 
                                                dynamic_ncols=True, 
                                                total=total_iterations_eval):
                        

            x_dict = graph_encoder(hetero_batch.x_dict, hetero_batch.edge_index_dict)
            output = graph_decoder(x_dict)

            batch_masks = batch_masks.to(output.device).view(-1, 1)
            valid_mask = (batch_masks != -1).float()
            valid_output = output * valid_mask
            valid_batch_masks = batch_masks * valid_mask


            category_count, correct_count = compute_accuracy_with_lvl(valid_output, valid_batch_masks, hetero_batch)           

            for i in range(4):
                total_category_count[i] += category_count[i]
                total_correct_count[i] += correct_count[i]

            # Compute loss
            loss = criterion(valid_output, valid_batch_masks)
            eval_loss += loss.item()


    print("Category-wise Accuracy:")
    total_correct = 0
    total_samples = 0

    for i in range(4):
        if total_category_count[i] > 0:
            accuracy = total_correct_count[i] / total_category_count[i]
            total_correct += total_correct_count[i]
            total_samples += total_category_count[i]
            print(f"Category {i+1}: {accuracy:.4f}% (Correct: {total_correct_count[i]}/{total_category_count[i]})")
        else:
            print(f"Category {i+1}: No samples")

    # Calculate and print average evaluation loss
    average_eval_loss = eval_loss / total_iterations_eval
    print(f"Average Evaluation Loss: {average_eval_loss:.4f}")

    # Calculate and print overall average accuracy
    overall_accuracy = total_correct / total_samples 
    print(f"Overall Average Accuracy: {overall_accuracy:.4f}%")




#---------------------------------- Public Functions ----------------------------------#


train()
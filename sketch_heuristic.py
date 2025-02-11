import Preprocessing.dataloader
import Preprocessing.gnn_graph
import Preprocessing.gnn_graph_stroke

import random

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


# ------------------------------------------------------------------------------# 



def run():
    # Load the dataset
    dataset = Preprocessing.dataloader.Program_Graph_Dataset('dataset/messy')
    print(f"Total number of shape data: {len(dataset)}")
        

    correct = 0
    total = 0

    # Preprocess and build the graphs
    for data in tqdm(dataset, desc=f"Heuristic Searching...."):
        # Extract the necessary elements from the dataset
        stroke_cloud_loops, stroke_node_features, connected_stroke_nodes, loop_neighboring_vertical, loop_neighboring_horizontal, loop_neighboring_contained, loop_neighboring_coplanar, stroke_to_brep, stroke_operations_order_matrix, final_brep_edges = data

        second_last_column = stroke_operations_order_matrix[:, -2].reshape(-1, 1)
        chosen_strokes = (second_last_column == 1).nonzero(as_tuple=True)[0]  # Indices of chosen strokes
        loop_chosen_mask = []
        for loop in stroke_cloud_loops:
            if all(stroke in chosen_strokes for stroke in loop):
                loop_chosen_mask.append(1)  # Loop is chosen
            else:
                loop_chosen_mask.append(0)  # Loop is not chosen
        
        loop_selection_mask = torch.tensor(loop_chosen_mask, dtype=torch.float).reshape(-1, 1).to(device)
        if not (loop_selection_mask == 1).any():
            continue

        # Build the graph
        gnn_graph = Preprocessing.gnn_graph.SketchLoopGraph(
            stroke_cloud_loops, 
            stroke_node_features, 
            connected_stroke_nodes,
            loop_neighboring_vertical, 
            loop_neighboring_horizontal, 
            loop_neighboring_contained,
            stroke_to_brep
        )

        # Encoders.helper.vis_stroke_with_order(stroke_node_features)
        # Encoders.helper.vis_brep(final_brep_edges)
        # Encoders.helper.vis_whole_graph(gnn_graph, torch.argmax(loop_selection_mask))


        # Heuristic Search
        heuristic_search = BaselineHeuristicSearch(gnn_graph)
        output_matrix = heuristic_search.search()

        if torch.argmax(output_matrix) == torch.argmax(loop_selection_mask):
            correct += 1
        total += 1

    accuracy = correct / total if total > 0 else 0
    print(f" Accuracy: {accuracy:.5f} ")

    



import torch
import random

class BaselineHeuristicSearch:
    def __init__(self, graph):
        self.graph = graph  # SketchLoopGraph instance

    def search(self):
        # 1. Find loops with node features 0 and edges to loops with features 1
        loop_features = self.graph['loop'].x[:, 0]  # Assume the first feature indicates if the loop is used (1 or 0)
        
        # Get indices of loops with feature 0
        zero_feature_loops = torch.where(loop_features == 0)[0]
        one_feature_loops = torch.where(loop_features == 1)[0]

        # Find loops that are connected by 'contains' or 'neighboring_horizontal' edges to loops with feature 1
        candidate_loops = self._find_connected_loops(zero_feature_loops, one_feature_loops)

        if len(candidate_loops) > 0:
            # 2. Find the loop with the smallest average index
            best_loop = self._find_smallest_average_index(candidate_loops)

            # 3. Find the largest loop (that contains others if possible)
            largest_loop = self._find_largest_loop(candidate_loops, best_loop)
        else:
            # 4. If all features are 0, directly go to step 2 (find smallest average index among all loops)
            all_loops = torch.arange(loop_features.size(0))
            best_loop = self._find_smallest_average_index(all_loops)
            largest_loop = best_loop  # No containing logic needed in this case

        # Output: A matrix with (num_loop_nodes, 1), where only the chosen loop has value 1
        output = torch.zeros((loop_features.size(0), 1), dtype=torch.float)
        output[largest_loop, 0] = 1  # Set the chosen loop to 1

        return output

    def _find_connected_loops(self, zero_feature_loops, one_feature_loops):
        """Find loops with feature 0 that are connected to loops with feature 1 by 'contains' or 'neighboring_horizontal' edges."""
        contains_edges = self.graph['loop', 'contains', 'loop'].edge_index
        horizontal_edges = self.graph['loop', 'neighboring_horizontal', 'loop'].edge_index

        connected_loops = []

        # Check if the loop with feature 0 has an edge to any loop with feature 1
        for loop_idx in zero_feature_loops:
            # Check 'contains' edges
            contains_connection = contains_edges[0] == loop_idx
            if torch.any(torch.isin(contains_edges[1][contains_connection], one_feature_loops)):
                connected_loops.append(loop_idx)
                continue
            
            # Check 'neighboring_horizontal' edges
            horizontal_connection = horizontal_edges[0] == loop_idx
            if torch.any(torch.isin(horizontal_edges[1][horizontal_connection], one_feature_loops)):
                connected_loops.append(loop_idx)

        return connected_loops

    def _find_smallest_average_index(self, candidate_loops):
        """Find the loop with the smallest average index."""
        if len(candidate_loops) == 0:
            return None
        
        # Convert candidate_loops list to a tensor
        candidate_loops_tensor = torch.tensor(candidate_loops, dtype=torch.long)
        
        # Find the minimum index in the tensor
        return torch.min(candidate_loops_tensor).item()

    def _find_largest_loop(self, candidate_loops, best_loop):
        """Find the largest loop from the candidates, preferably one that contains others."""
        contains_edges = self.graph['loop', 'contains', 'loop'].edge_index
        largest_loop = best_loop

        # Convert candidate_loops list to a tensor
        candidate_loops_tensor = torch.tensor(candidate_loops, dtype=torch.long)

        # Check if any loop contains the others
        for loop in candidate_loops:
            contained_loops = contains_edges[0] == loop
            # Use torch.isin with both arguments as tensors
            if torch.all(torch.isin(contains_edges[1][contained_loops], candidate_loops_tensor)):
                largest_loop = loop
                break

        return largest_loop if largest_loop is not None else random.choice(candidate_loops)


#---------------------------------- Public Functions ----------------------------------#


run()
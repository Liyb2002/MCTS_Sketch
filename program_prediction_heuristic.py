import Preprocessing.dataloader
import Preprocessing.gnn_graph
import Preprocessing.gnn_graph_stroke

import Encoders.gnn.gnn
import Encoders.helper

from torch_geometric.loader import DataLoader as GraphDataLoader
from torch.utils.data import DataLoader, TensorDataset

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
    dataset = Preprocessing.dataloader.Program_Graph_Dataset('dataset/messy_order')
    print(f"Total number of shape data: {len(dataset)}")
    
    
    terminate_graphs = []
    non_terminate_graphs = []

    # Preprocess and build the graphs
    for data in tqdm(dataset, desc=f"Building Graphs"):
        # Extract the necessary elements from the dataset
        program, program_whole, stroke_cloud_loops, stroke_node_features, strokes_perpendicular, output_brep_edges, stroke_operations_order_matrix, loop_neighboring_vertical, loop_neighboring_horizontal,loop_neighboring_contained, stroke_to_loop, stroke_to_edge = data
        
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

        if program[-1] == 'terminate':
            terminate_graphs.append(gnn_graph)
        else:
            non_terminate_graphs.append(gnn_graph)
        
        if len(terminate_graphs) > 100:
            break
        



    print(f"Total number of terminate_graphs graphs: {len(terminate_graphs)}, Total number of non_terminate_graphs graphs: {len(non_terminate_graphs)}")


    # 1) terminate_graphs
    terminate_correct = 0
    for graph in tqdm(terminate_graphs, 
                    desc=f"Terminate_graphs", 
                    dynamic_ncols=True, 
                    total=len(terminate_graphs)):

        if (not graph._has_circle_shape) or (not graph._full_shape):
            terminate_correct +=1
    
    print("terminate_graphs accuracy:", terminate_correct/len(terminate_graphs))


    # 2) non_terminate_graphs
    non_terminate_correct = 0
    for graph in tqdm(non_terminate_graphs, 
                    desc=f"Non Terminate_graphs", 
                    dynamic_ncols=True, 
                    total=len(non_terminate_graphs)):

        if graph._has_circle_shape:
            non_terminate_correct +=1
    
    print("terminate_graphs accuracy:", non_terminate_correct/len(non_terminate_graphs))




#---------------------------------- Public Functions ----------------------------------#


run()
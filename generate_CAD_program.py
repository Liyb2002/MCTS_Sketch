import Preprocessing.dataloader
import Preprocessing.generate_dataset_baseline
import Preprocessing.gnn_graph

import Preprocessing.proc_CAD.generate_program
import Preprocessing.proc_CAD.Program_to_STL
import Preprocessing.proc_CAD.brep_read
import Preprocessing.proc_CAD.helper

import whole_process_helper.helper

import Models.loop_embeddings

import Encoders.gnn.gnn
import Encoders.gnn_stroke.gnn
import Encoders.helper

import particle
# import whole_process_evaluateo0

from torch.utils.data import DataLoader
from tqdm import tqdm

import pickle
import torch
import torch.nn as nn
import torch.optim as optim
import os
import shutil
import numpy as np
import random
import copy
import re
import time

from collections import deque





# --------------------- Dataset --------------------- #
dataset = Preprocessing.dataloader.Program_Graph_Dataset('dataset/whole', return_data_path=True)
data_loader = DataLoader(dataset, batch_size=1, shuffle=True)


# --------------------- Directory --------------------- #
current_dir = os.getcwd()
output_dir = os.path.join(current_dir, 'MCTS_dataset')



# --------------------- Compute_start_idx --------------------- #
def compute_start_idx():
    data_dirs = [d for d in os.listdir(output_dir) if os.path.isdir(os.path.join(output_dir, d))]

    pattern = re.compile(r'.*_(\d+)$')
    
    largest_number = 0
    
    # List all directories and retrieve the number at the end of the format
    for d in os.listdir(output_dir):
        if os.path.isdir(os.path.join(output_dir, d)):
            match = pattern.match(d)
            if match:
                number = int(match.group(1))  # Extract the number
                largest_number = max(largest_number, number)  # Keep track of the largest number

    
    return max(largest_number, 0)


def handle_failed_program(cur_output_dir, data_produced):
    """Removes everything in cur_output_dir and decrements data_produced if success_program is False."""
    
    # Check if directory exists
    if os.path.exists(cur_output_dir):
        # Remove all contents inside the directory
        shutil.rmtree(cur_output_dir)
        print(f"Removed all contents in {cur_output_dir}")

    # Decrement data_produced
    data_produced -= 1
    print(f"data_produced decremented to {data_produced}")

    return data_produced


# --------------------- Main Code --------------------- #
data_produced = compute_start_idx()
data_limit = 1
if os.path.exists(os.path.join(output_dir, f'data_{data_produced}')):
    shutil.rmtree(os.path.join(output_dir, f'data_{data_produced}'))
os.makedirs(os.path.join(output_dir, f'data_{data_produced}'), exist_ok=True)

print("data_produced", data_produced)

for data in tqdm(data_loader, desc="Generating CAD Programs"):
    program, stroke_node_features, data_path= data

    if data_produced > data_limit:
        break

    if program[-1][0] != 'terminate' or len(program) < 6:
        continue
    
    cur_output_dir = os.path.join(output_dir, f'data_{data_produced}')
    if os.path.exists(cur_output_dir):
        shutil.rmtree(cur_output_dir)
    os.makedirs(cur_output_dir, exist_ok=True)
    

    gt_brep_dir = os.path.join(data_path[0], 'canvas')
    brep_files = [file_name for file_name in os.listdir(gt_brep_dir)
            if file_name.startswith('brep_') and file_name.endswith('.step')]
    brep_files.sort(key=lambda x: int(x.split('_')[1].split('.')[0]))
    gt_brep_file_path = os.path.join(gt_brep_dir, brep_files[-1])


    base_particle = particle.Particle(gt_brep_file_path, data_produced, stroke_node_features.squeeze(0).cpu().numpy())
    base_particle.set_gt_program(program)
    base_particle.set_particle_id(0, cur_output_dir)



    reproducible_particles = [base_particle]
    all_particles = [base_particle]
    num_states = 1

    while len(reproducible_particles) != 0:

        if num_states > 100:
            break

        reproducible_particle = reproducible_particles.pop(0) 

        available_ops = reproducible_particle.reproduce()

        if len(available_ops) == 0:
            continue

        for op, prob, param in available_ops:

            new_particle = reproducible_particle.deepcopy_particle(num_states, prob)
            new_particle.current_op = op
            new_particle.generate_next_step(param)

            reproducible_particle.childNodes.append(new_particle)

            reproducible_particles.append(new_particle)
            all_particles.append(new_particle)

            num_states += 1


    # Now we need to find the leafNodes
    leafNodes_list = []
    for tree_node in all_particles:
        if tree_node.leafNode:
            leafNodes_list.append(tree_node)


    print("start rollout")
    # For all leafNodes, sample solutions
    for leaf_node in leafNodes_list:
        for i in range (0, 10):
            copied_particle = leaf_node.deepcopy_particle(leaf_node.particle_id * 100 + i, 1)
            copied_particle.sample_tree()
            leaf_node.value = max(leaf_node.value, copied_particle.value)
    
    
    for leaf_node in leafNodes_list:
        print("leaf Node fideleity score", leaf_node.value)
        
    # print("Start Tree Computation")
    # base_particle.print_tree()





    data_produced += 1




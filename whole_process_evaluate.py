
from torch.utils.data import Dataset, DataLoader
import os
import pickle
from tqdm import tqdm
import torch
import glob
import json
import random

import Preprocessing.SBGCN.brep_read
import Preprocessing.proc_CAD.helper
import Encoders.helper


import fidelity_score

# --------------------- Dataloader for output --------------------- #
class Evaluation_Dataset(Dataset):
    def __init__(self, dataset):
        self.data_path = os.path.join(os.getcwd(), dataset)
        self.data_dirs = [
            os.path.join(self.data_path, d) 
            for d in os.listdir(self.data_path) 
            if os.path.isdir(os.path.join(self.data_path, d))
        ]

        # List of sublist. Each sublist is all the particles in a data piece
        self.data_particles = []
        
        self.data_particles = [
        [
            os.path.join(data_dir, subfolder)
            for subfolder in os.listdir(data_dir)
            if os.path.isdir(os.path.join(data_dir, subfolder))
        ]
        for data_dir in self.data_dirs
        ]



        # all flatter all the particles
        self.flatted_particle_folders = [
            folder
            for sublist in self.data_particles
            for folder in sublist
        ]


        print(f"Total number of data pieces: {len(self.flatted_particle_folders)}")

    def __len__(self):
        return len(self.flatted_particle_folders)

    def __getitem__(self, idx):
        folder = self.flatted_particle_folders[idx]
    

        particle_value_file = os.path.join(folder, "particle_value.json")
        if not os.path.exists(particle_value_file):
            return self.__getitem__((idx + 1) % len(self.flatted_particle_folders))
        else:
            with open(particle_value_file, 'r') as f:
                particle_data = json.load(f)
        particle_value = particle_data.get('value', None)  # Extract 'value', default to None if missing
        if particle_value == 0 and random.random() < 0.8:
            return self.__getitem__((idx + 1) % len(self.flatted_particle_folders))


        canvas_dir = os.path.join(folder, 'canvas')

        # Find the eval file dynamically
        eval_files = glob.glob(os.path.join(canvas_dir, "*_eval_info.pkl"))
        if not eval_files:
            return self.__getitem__((idx + 1) % len(self.flatted_particle_folders))  # Try the next item
        eval_file = eval_files[0]  # Assuming there's only one

        with open(eval_file, 'rb') as f:
            shape_data = pickle.load(f)

        # Convert numpy arrays to tensors
        stroke_node_features = torch.tensor(shape_data['stroke_node_features'], dtype=torch.float32)

        # Find the highest index `brep_xxx.step`
        brep_files = glob.glob(os.path.join(canvas_dir, "brep_*.step"))
        if not brep_files:
            return self.__getitem__((idx + 1) % len(self.flatted_particle_folders))  # Skip if no BREP file found

        highest_brep_file = max(brep_files, key=lambda x: int(os.path.basename(x).split('_')[-1].split('.')[0]))

        # Load generated BREP file
        edge_features_list, cylinder_features = Preprocessing.SBGCN.brep_read.create_graph_from_step_file(highest_brep_file)
        output_brep_edges = Preprocessing.proc_CAD.helper.pad_brep_features(edge_features_list + cylinder_features)
        output_brep_edges = torch.tensor(output_brep_edges, dtype=torch.float32)

        # Convert numpy arrays to tensors
        gt_brep_edges = torch.tensor(shape_data['gt_brep_edges'], dtype=torch.float32)
        strokes_perpendicular = torch.tensor(shape_data['strokes_perpendicular'], dtype=torch.float32)
        loop_neighboring_vertical = torch.tensor(shape_data['loop_neighboring_vertical'], dtype=torch.long)
        loop_neighboring_horizontal = torch.tensor(shape_data['loop_neighboring_horizontal'], dtype=torch.long)
        loop_neighboring_contained = torch.tensor(shape_data['loop_neighboring_contained'], dtype=torch.long)
        stroke_to_loop = torch.tensor(shape_data['stroke_to_loop'], dtype=torch.long)
        stroke_to_edge = torch.tensor(shape_data['stroke_to_edge'], dtype=torch.long)

        return (
            particle_value,
            stroke_node_features, output_brep_edges, gt_brep_edges,
            shape_data['stroke_cloud_loops'], 
            strokes_perpendicular, loop_neighboring_vertical,
            loop_neighboring_horizontal, loop_neighboring_contained,
            stroke_to_loop, stroke_to_edge
        )




# --------------------- Main Code --------------------- #


def run_eval():
    # Set up dataloader
    dataset = Evaluation_Dataset('program_output_test', 3)
    data_loader = DataLoader(dataset, batch_size=1, shuffle=False)

    total_correct = 0
    total = 0

    prev_brep_edges = None

    for data in tqdm(data_loader, desc="Evaluating CAD Programs"):
        stroke_node_features, output_brep_edges, gt_brep_edges, cur_fidelity_score, stroke_cloud_loops, strokes_perpendicular, loop_neighboring_vertical, loop_neighboring_horizontal, loop_neighboring_contained, stroke_to_loop, stroke_to_edge = data

        stroke_node_features = stroke_node_features.squeeze(0)
        stroke_node_features = torch.round(stroke_node_features * 10000) / 10000

        output_brep_edges = output_brep_edges.squeeze(0)
        output_brep_edges = torch.round(output_brep_edges * 10000) / 10000

        gt_brep_edges = gt_brep_edges.squeeze(0)
        gt_brep_edges = torch.round(gt_brep_edges * 10000) / 10000

        print("cur_fidelity_score", cur_fidelity_score)
        Encoders.helper.vis_brep(output_brep_edges)
        Encoders.helper.vis_brep(gt_brep_edges)

        # unique_new_edges = brep_difference(prev_brep_edges, output_brep_edges)
        # Encoders.helper.vis_brep(unique_new_edges)

        
        total += 1
    print(f"Overall Average Accuracy: {total_correct / total:.4f}, with total_correct : {total_correct} and total: {total}")


# run_eval()
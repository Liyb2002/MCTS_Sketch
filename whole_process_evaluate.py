
from torch.utils.data import Dataset, DataLoader
import os
import pickle
from tqdm import tqdm
import torch

import Preprocessing.SBGCN.brep_read
import Preprocessing.proc_CAD.helper
import Encoders.helper


import fidelity_score

# --------------------- Dataloader for output --------------------- #
class Evaluation_Dataset(Dataset):
    def __init__(self, dataset, mode = 1):
        self.data_path = os.path.join(os.getcwd(), dataset)
        self.data_dirs = [
            os.path.join(self.data_path, d) 
            for d in os.listdir(self.data_path) 
            if os.path.isdir(os.path.join(self.data_path, d))
        ]

        # List of sublist. Each sublist is all the particles in a data piece
        self.data_particles = []
        
        # mode 1: sample a particle
        if mode == 1:
            for data_dir in self.data_dirs:
                found_folder = False
                for subfolder in os.listdir(data_dir):
                    if os.path.isdir(os.path.join(data_dir, subfolder)):
                        self.data_particles.append([os.path.join(data_dir, subfolder)])
                        found_folder = True
                        break
                if not found_folder:
                    self.data_particles.append([])
        
        # mode 2: find all particles
        if mode == 2:
            self.data_particles = [
            [
                os.path.join(data_dir, subfolder)
                for subfolder in os.listdir(data_dir)
                if os.path.isdir(os.path.join(data_dir, subfolder))
            ]
            for data_dir in self.data_dirs
        ]

        if mode == 3:
            # mode 3: only find particles that ends with _output
            for data_dir in self.data_dirs:
                found_folder = False
                for subfolder in os.listdir(data_dir):
                    # Check if the item is a directory and ends with '_output'
                    if os.path.isdir(os.path.join(data_dir, subfolder)) and subfolder.endswith('_output'):
                        self.data_particles.append([os.path.join(data_dir, subfolder)])
                        found_folder = True
                        break
                if not found_folder:
                    self.data_particles.append([])



        # all flatter all the particles
        self.flatted_particle_folders = [
            folder
            for sublist in self.data_particles
            for folder in sublist
        ]

        # Collect all data pieces: (folder_path, file_index)
        self.data_pieces = []

        if mode == 1 or mode == 2:
            for folder in self.flatted_particle_folders:
                canvas_dir = os.path.join(folder, 'canvas')
                if os.path.exists(canvas_dir) and os.path.isdir(canvas_dir):
                    shape_files = sorted(
                        f for f in os.listdir(canvas_dir) if f.endswith('_eval_info.pkl')
                    )
                    for shape_file in shape_files:
                        index = int(shape_file.split('_')[0])
                        self.data_pieces.append((folder, index))

        if mode == 3:
            for folder in self.flatted_particle_folders:
                canvas_dir = os.path.join(folder, 'canvas')
                if os.path.exists(canvas_dir) and os.path.isdir(canvas_dir):
                    shape_files = [
                        f for f in os.listdir(canvas_dir) if f.endswith('_eval_info.pkl')
                    ]
                    if shape_files:  # Ensure there are files to process
                        # Get the file with the highest index
                        max_shape_file = max(
                            shape_files, key=lambda x: int(x.split('_')[0])
                        )
                        index = int(max_shape_file.split('_')[0])
                        self.data_pieces.append((folder, index))

        print(f"Total number of data pieces: {len(self.data_pieces)}")

    def __len__(self):
        return len(self.data_pieces)

    def __getitem__(self, idx):
        folder, file_index = self.data_pieces[idx]
        canvas_dir = os.path.join(folder, 'canvas')

        # Load stroke node features
        eval_file = os.path.join(canvas_dir, f"{file_index}_eval_info.pkl")
        if not os.path.exists(eval_file):
            raise FileNotFoundError(f"{eval_file} not found.")
        with open(eval_file, 'rb') as f:
            shape_data = pickle.load(f)

        # Convert numpy arrays to tensors
        stroke_node_features = torch.tensor(shape_data['stroke_node_features'], dtype=torch.float32)

        # Load generated BREP file
        brep_file = os.path.join(canvas_dir, f"brep_{file_index}.step")
        edge_features_list, cylinder_features = Preprocessing.SBGCN.brep_read.create_graph_from_step_file(brep_file)
        output_brep_edges = Preprocessing.proc_CAD.helper.pad_brep_features(edge_features_list + cylinder_features)
        output_brep_edges = torch.tensor(output_brep_edges, dtype=torch.float32)

        # Convert numpy arrays to tensors
        gt_brep_edges = torch.tensor(shape_data['gt_brep_edges'], dtype=torch.float32)
        cur_fidelity_score = torch.tensor(shape_data['cur_fidelity_score'], dtype=torch.float32)
        strokes_perpendicular = torch.tensor(shape_data['strokes_perpendicular'], dtype=torch.float32)
        loop_neighboring_vertical = torch.tensor(shape_data['loop_neighboring_vertical'], dtype=torch.long)  # Adjacency typically uses torch.long
        loop_neighboring_horizontal = torch.tensor(shape_data['loop_neighboring_horizontal'], dtype=torch.long)
        loop_neighboring_contained = torch.tensor(shape_data['loop_neighboring_contained'], dtype=torch.long)
        stroke_to_loop = torch.tensor(shape_data['stroke_to_loop'], dtype=torch.long)  # Relationships should use torch.long
        stroke_to_edge = torch.tensor(shape_data['stroke_to_edge'], dtype=torch.long)


        return (
            stroke_node_features, output_brep_edges, gt_brep_edges,
            cur_fidelity_score,
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
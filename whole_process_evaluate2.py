
from torch.utils.data import Dataset, DataLoader
import os
import pickle
from tqdm import tqdm
import torch
import re

import fidelity_score

import Preprocessing.SBGCN.brep_read
import Preprocessing.proc_CAD.helper
import Encoders.helper

import fidelity_score

# --------------------- Dataloader for output --------------------- #
class Evaluation_Dataset(Dataset):
    def __init__(self, dataset):
        self.data_path = os.path.join(os.getcwd(), dataset)
        self.data_pieces = []

        # Get the list of directories in the dataset
        self.data_dirs = [
            os.path.join(self.data_path, d)
            for d in os.listdir(self.data_path)
            if os.path.isdir(os.path.join(self.data_path, d))
        ]

        for data_dir in self.data_dirs:
            particle_data = []
            for subfolder in os.listdir(data_dir):
                subfolder_path = os.path.join(data_dir, subfolder)

                # Check if the subfolder is an `_output` directory
                if os.path.isdir(subfolder_path) and subfolder.endswith('_output'):
                    gt_brep_path = os.path.join(subfolder_path, 'gt_brep.step')
                    canvas_dir = os.path.join(subfolder_path, 'canvas')

                    # Validate the existence of the canvas directory
                    if os.path.exists(canvas_dir) and os.path.isdir(canvas_dir):
                        # Find all files matching the pattern `brep_{num}.step`
                        files = os.listdir(canvas_dir)
                        brep_files = [f for f in files if re.match(r'brep_\d+\.step$', f)]

                        if brep_files:
                            # Extract the highest-numbered BREP file
                            brep_numbers = [int(re.search(r'brep_(\d+)\.step$', f).group(1)) for f in brep_files]
                            highest_num = max(brep_numbers)
                            highest_brep_file = f'brep_{highest_num}.step'
                            highest_brep_path = os.path.join(canvas_dir, highest_brep_file)

                            # Add the pair of ground truth and output BREP paths
                            particle_data.append([gt_brep_path, highest_brep_path])

            # Add this particle's data as a new data piece
            if particle_data:
                self.data_pieces.append(particle_data)

        print(f"Total number of particles: {len(self.data_pieces)}")
        print(f"Total number of data pieces: {sum(len(p) for p in self.data_pieces)}")

    def __len__(self):
        return len(self.data_pieces)

    def __getitem__(self, idx):
        return self.data_pieces[idx]




# --------------------- Main Code --------------------- #


def run_eval():
    # Set up dataloader
    dataset = Evaluation_Dataset('program_output_dataset')
    data_loader = DataLoader(dataset, batch_size=1, shuffle=False, collate_fn=lambda x: x[0])

    total_correct = 0
    total = 0

    for particle in tqdm(data_loader, desc="Evaluating CAD Programs"):

        best_fidelity_score = 0
        for gt_brep_path, output_brep_path in particle:

            try:
                cur_fidelity_score = fidelity_score.compute_fidelity_score(gt_brep_path, output_brep_path)
            except Exception as e:
                print(f"Error while computing fidelity score: {e}")
                cur_fidelity_score = 0

            if cur_fidelity_score > best_fidelity_score:
                best_fidelity_score = cur_fidelity_score

        if best_fidelity_score < 0.95:
            print("gt_brep_path", gt_brep_path)
        
        print("Best fidelity score for this particle:", best_fidelity_score)
        print("-----------")

    # print(f"Overall Average Accuracy: {total_correct / total:.4f}, with total_correct : {total_correct} and total: {total}")


run_eval()
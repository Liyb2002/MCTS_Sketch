
import Preprocessing.SBGCN.brep_read
import torch
import torch.nn as nn
import torch.optim as optim
from torch_geometric.data import DataLoader 
import os

import Preprocessing.SBGCN.SBGCN_network
import Preprocessing.SBGCN.decoder
from tqdm import tqdm
import Preprocessing.SBGCN.io_utils

def train_graph_embedding(dataset, num_epochs=1, batch_size=1, learning_rate=0.001):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


    # Define loss function and optimizer
    criterion = nn.BCELoss()
    model = Preprocessing.SBGCN.SBGCN_network.FaceEdgeVertexGCN()
    decoder_model = Preprocessing.SBGCN.decoder.SBGCN_Decoder()

    optimizer = optim.Adam(list(model.parameters()) + list(decoder_model.parameters()), lr=learning_rate)

    # Create DataLoader for batching
    dataloader = DataLoader(dataset, batch_size=batch_size, 
                                 shuffle=False)
    
    # Training loop
    for epoch in range(num_epochs):
        total_loss = 0.0

        
        # Iterate over batches
        for batch in tqdm(dataloader):

            step_path = batch[0]
            graph = Preprocessing.SBGCN.brep_read.create_graph_from_step_file(step_path)

            x_f, x_e, x_v = model(graph)
            reconstruct_matrix = decoder_model(x_f, x_e, x_v)
            gt_matrix = graph['face'].z
            loss = criterion(reconstruct_matrix, gt_matrix)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item() * len(batch)

        print(f"Epoch {epoch + 1}, Average Loss: {total_loss / len(dataset)}")

    Preprocessing.SBGCN.io_utils.save_model(model, "sbgcn_model")
    return model


def load_pretrained_SBGCN_model():
    model = Preprocessing.SBGCN.SBGCN_network.FaceEdgeVertexGCN()

    checkpoint_path = os.path.join(Preprocessing.SBGCN.io_utils.home_dir, "model_checkpoints", "sbgcn_model" , "sbgcn_model" + ".ckpt")
    loaded_model = Preprocessing.SBGCN.io_utils.load_model(model, checkpoint_path)
    if loaded_model is not None:
        return loaded_model

    return None






def run():
    step_path =  ['../proc_CAD/canvas/step_2.step']
    for i in range(100):
        step_path.append(step_path[-1])

    dataset = Preprocessing.SBGCN.brep_read.BRep_Dataset(step_path)

    model = train_graph_embedding(dataset)



import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import HeteroConv, GCNConv, EdgeConv
from torch_geometric.data import HeteroData

import Encoders.gnn_stroke.basic

class SemanticModule(nn.Module):
    def __init__(self, in_channels=6, mlp_channels=16):
        super(SemanticModule, self).__init__()
        self.local_head = Encoders.gnn_stroke.basic.GeneralHeteroConv(['intersects_mean', 'temp_previous_add',  'represented_by_mean'], in_channels, 32)

        self.layers = nn.ModuleList([
            Encoders.gnn_stroke.basic.ResidualGeneralHeteroConvBlock(['intersects_mean','temp_previous_add',  'represented_by_mean'], 32, 32),
            Encoders.gnn_stroke.basic.ResidualGeneralHeteroConvBlock(['intersects_mean','temp_previous_add',  'represented_by_mean'], 32, 32),
            Encoders.gnn_stroke.basic.ResidualGeneralHeteroConvBlock(['intersects_mean','temp_previous_add',  'represented_by_mean'], 32, 32),
            Encoders.gnn_stroke.basic.ResidualGeneralHeteroConvBlock(['intersects_mean','temp_previous_add','represented_by_mean'], 32, 32),
        ])


    def forward(self, x_dict, edge_index_dict):
        x_dict = self.local_head(x_dict, edge_index_dict)

        for layer in self.layers:
            x_dict = layer(x_dict, edge_index_dict)
        
        x_dict = {key: x.relu() for key, x in x_dict.items()}

        return x_dict



class ExtrudingStrokePrediction(nn.Module):
    def __init__(self, hidden_channels=64):
        super(ExtrudingStrokePrediction, self).__init__()

        self.edge_conv = Encoders.gnn_stroke.basic.ResidualGeneralHeteroConvBlock(['intersects_mean','temp_previous_add',  'represented_by_mean'], 33, 33)

        self.local_head = nn.Linear(33, 64) 
        self.decoder = nn.Sequential(
            nn.Linear(64, hidden_channels),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_channels, 1),
        )

    def forward(self, x_dict, edge_index_dict, prev_sketch_strokes):
        prev_sketch_strokes = prev_sketch_strokes.to(torch.float32)
        combined_stroke = torch.cat([x_dict['stroke'].to(torch.float32), prev_sketch_strokes], dim=1)

        zeros = torch.zeros((x_dict['brep'].shape[0], 1), device=x_dict['brep'].device, dtype=torch.float32)
        combined_brep = torch.cat([x_dict['brep'].to(torch.float32), zeros], dim=1)

        x_dict['stroke'] = combined_stroke
        x_dict['brep'] = combined_brep

        x_dict = self.edge_conv(x_dict, edge_index_dict)
        features = self.local_head(x_dict['stroke'])
        return torch.sigmoid(self.decoder(features))

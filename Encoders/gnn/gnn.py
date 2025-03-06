import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import HeteroConv, GCNConv, EdgeConv
from torch_geometric.data import HeteroData

import Encoders.gnn.basic

class SemanticModule(nn.Module):
    def __init__(self, in_channels=12):
        super(SemanticModule, self).__init__()
        self.local_head = Encoders.gnn.basic.GeneralHeteroConv(['represents_sum', 'represented_by_sum', 'neighboring_vertical_mean', 'neighboring_horizontal_mean', 'contains_sum', 'order_add', 'perpendicular_mean'], in_channels, 16)

        self.layers = nn.ModuleList([
            Encoders.gnn.basic.ResidualGeneralHeteroConvBlock(['represents_sum', 'represented_by_sum', 'neighboring_vertical_mean', 'neighboring_horizontal_mean', 'contains_sum', 'order_add', 'perpendicular_mean'], 16, 32),
            Encoders.gnn.basic.ResidualGeneralHeteroConvBlock(['represents_sum', 'represented_by_sum', 'neighboring_vertical_mean', 'neighboring_horizontal_mean', 'contains_sum', 'order_add', 'perpendicular_mean'], 32, 64),
            Encoders.gnn.basic.ResidualGeneralHeteroConvBlock(['represents_sum', 'represented_by_sum', 'neighboring_vertical_mean', 'neighboring_horizontal_mean', 'contains_sum', 'order_add', 'perpendicular_mean'], 64, 128),
            Encoders.gnn.basic.ResidualGeneralHeteroConvBlock(['represents_sum', 'represented_by_sum', 'neighboring_vertical_mean', 'neighboring_horizontal_mean', 'contains_sum', 'order_add', 'perpendicular_mean'], 128, 128),
            Encoders.gnn.basic.ResidualGeneralHeteroConvBlock(['represents_sum', 'represented_by_sum', 'neighboring_vertical_mean', 'neighboring_horizontal_mean', 'contains_sum', 'order_add', 'perpendicular_mean'], 128, 128),
            Encoders.gnn.basic.ResidualGeneralHeteroConvBlock(['represents_sum', 'represented_by_sum', 'neighboring_vertical_mean', 'neighboring_horizontal_mean', 'contains_sum', 'order_add', 'perpendicular_mean'], 128, 128),

        ])


    def forward(self, x_dict, edge_index_dict):

        x_dict = self.local_head(x_dict, edge_index_dict)

        for layer in self.layers:
            x_dict = layer(x_dict, edge_index_dict)
        
        x_dict = {key: x.relu() for key, x in x_dict.items()}

        return x_dict





class Sketch_Decoder(nn.Module):
    def __init__(self, hidden_channels=256):
        super(Sketch_Decoder, self).__init__()

        self.decoder = nn.Sequential(
            nn.Linear(128, hidden_channels),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.1),
            nn.Linear(hidden_channels, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.1),
            nn.Linear(64, 1),

        )

    def forward(self, x_dict):
        return torch.sigmoid(self.decoder(x_dict['loop']))


class Extrude_Decoder(nn.Module):
    def __init__(self, hidden_channels=256):
        super(Extrude_Decoder, self).__init__()

        self.decoder = nn.Sequential(
            nn.Linear(128, hidden_channels),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.1),
            nn.Linear(hidden_channels, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.1),
            nn.Linear(64, 1),

        )

    def forward(self, x_dict):
        return torch.sigmoid(self.decoder(x_dict['stroke']))


class Fillet_Decoder(nn.Module):
    def __init__(self, hidden_channels=256):
        super(Fillet_Decoder, self).__init__()

        self.decoder = nn.Sequential(
            nn.Linear(128, hidden_channels),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.1),
            nn.Linear(hidden_channels, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.1),
            nn.Linear(64, 1),

        )

    def forward(self, x_dict):
        return torch.sigmoid(self.decoder(x_dict['stroke']))


class Chamfer_Decoder(nn.Module):
    def __init__(self, hidden_channels=256):
        super(Chamfer_Decoder, self).__init__()

        self.decoder = nn.Sequential(
            nn.Linear(128, hidden_channels),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.1),
            nn.Linear(hidden_channels, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.1),
            nn.Linear(64, 1),

        )

    def forward(self, x_dict):
        return torch.sigmoid(self.decoder(x_dict['stroke']))


class Fidelity_Decoder(nn.Module):
    def __init__(self, hidden_channels=256, num_loop_nodes=400, num_stroke_nodes=400):
        super(Fidelity_Decoder, self).__init__()

        # Decoders for loop and stroke embeddings
        self.loop_decoder = nn.Sequential(
            nn.Linear(128, hidden_channels),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.1),
            nn.Linear(hidden_channels, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.1),
            nn.Linear(64, 1)  # Single output for continuous prediction
        )

        self.stroke_decoder = nn.Sequential(
            nn.Linear(128, hidden_channels),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.1),
            nn.Linear(hidden_channels, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.1),
            nn.Linear(64, 1)  # Single output for continuous prediction
        )

        self.num_loop_nodes = num_loop_nodes
        self.num_stroke_nodes = num_stroke_nodes

    def forward(self, x_dict, for_particle=False):
        if not for_particle: 
            loop_embeddings = x_dict['loop']
            stroke_embeddings = x_dict['stroke']

            # Compute batch size dynamically
            batch_size = loop_embeddings.size(0) // self.num_loop_nodes
            feature_dim = loop_embeddings.size(-1)

            # Reshape embeddings
            loop_embeddings = loop_embeddings.view(batch_size, self.num_loop_nodes, feature_dim)
            stroke_embeddings = stroke_embeddings.view(batch_size, self.num_stroke_nodes, feature_dim)
        else:
            loop_embeddings = x_dict['loop']
            stroke_embeddings = x_dict['stroke']
            feature_dim = loop_embeddings.size(-1)

            # Pad loop_embeddings
            if loop_embeddings.size(0) < self.num_loop_nodes:
                padding_size = self.num_loop_nodes - loop_embeddings.size(0)
                loop_embeddings = F.pad(loop_embeddings, (0, 0, 0, padding_size))
            else:
                loop_embeddings = loop_embeddings[:self.num_loop_nodes]

            # Pad stroke_embeddings
            if stroke_embeddings.size(0) < self.num_stroke_nodes:
                padding_size = self.num_stroke_nodes - stroke_embeddings.size(0)
                stroke_embeddings = F.pad(stroke_embeddings, (0, 0, 0, padding_size))
            else:
                stroke_embeddings = stroke_embeddings[:self.num_stroke_nodes]

            # Reshape to batch_size=1
            loop_embeddings = loop_embeddings.unsqueeze(0)
            stroke_embeddings = stroke_embeddings.unsqueeze(0)

        # Decode each node separately
        loop_scores = self.loop_decoder(loop_embeddings)  # Shape: [batch_size, num_loop_nodes, 1]
        stroke_scores = self.stroke_decoder(stroke_embeddings)  # Shape: [batch_size, num_stroke_nodes, 1]

        # Average over all nodes instead of summing (to normalize output scale)
        loop_graph_score = loop_scores.mean(dim=1)  # Shape: [batch_size, 1]
        stroke_graph_score = stroke_scores.mean(dim=1)  # Shape: [batch_size, 1]

        # Combine scores from loops and strokes
        combined_score = (loop_graph_score + stroke_graph_score) / 2  # Shape: [batch_size, 1]

        # Ensure output is in [0, 1] range
        return torch.sigmoid(combined_score)




class Fidelity_Decoder_bin(nn.Module):
    def __init__(self, hidden_channels=256, num_loop_nodes=400, num_stroke_nodes=400, num_classes=10):
        super(Fidelity_Decoder_bin, self).__init__()

        # Transform the last feature into a 128D vector using a linear layer
        self.last_feature_embedding = nn.Linear(1, 128)

        # Adjust stroke decoder input size (original 128 + 128 from last feature)
        self.stroke_decoder = nn.Sequential(
            nn.Linear(256, hidden_channels, bias=True),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.1),
            nn.Linear(hidden_channels, 64, bias=True),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.1),
            nn.Linear(64, num_classes, bias=True)
        )

        self.loop_decoder = nn.Sequential(
            nn.Linear(128, hidden_channels, bias=True),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.1),
            nn.Linear(hidden_channels, 64, bias=True),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.1),
            nn.Linear(64, num_classes, bias=True)
        )

        self.num_loop_nodes = num_loop_nodes
        self.num_stroke_nodes = num_stroke_nodes
        self.num_classes = num_classes

    def forward(self, x_dict, hetero_batch, for_particle=False):
        if not for_particle: 
            loop_embeddings = x_dict['loop']
            stroke_embeddings = x_dict['stroke']

            # Extract last feature as a float tensor
            last_feature = hetero_batch.x_dict['stroke'][:, -1].unsqueeze(-1).float()

            # Normalize last feature before embedding
            last_feature = (last_feature - last_feature.mean()) / (last_feature.std() + 1e-5)

            # Pass through a linear layer to get a 128D embedding
            last_feature_embedded = self.last_feature_embedding(last_feature)  # Shape: [num_strokes, 128]

            # Normalize stroke embeddings
            stroke_embeddings = (stroke_embeddings - stroke_embeddings.mean()) / (stroke_embeddings.std() + 1e-5)

            # Concatenate last feature embedding with stroke embeddings
            stroke_embeddings = torch.cat([stroke_embeddings, last_feature_embedded], dim=-1)  # Shape: [num_strokes, 256]

            # Compute batch size dynamically
            batch_size = loop_embeddings.size(0) // self.num_loop_nodes
            feature_dim = loop_embeddings.size(-1)

            # Reshape embeddings
            loop_embeddings = loop_embeddings.view(batch_size, self.num_loop_nodes, feature_dim)
            stroke_embeddings = stroke_embeddings.view(batch_size, self.num_stroke_nodes, 256)
        else:
            loop_embeddings = x_dict['loop']
            stroke_embeddings = x_dict['stroke']
            feature_dim = loop_embeddings.size(-1)

            # Extract last feature and transform
            last_feature = hetero_batch.x_dict['stroke'][:, -1].unsqueeze(-1).float()
            last_feature = (last_feature - last_feature.mean()) / (last_feature.std() + 1e-5)
            last_feature_embedded = self.last_feature_embedding(last_feature)

            # Normalize stroke embeddings before concatenation
            stroke_embeddings = (stroke_embeddings - stroke_embeddings.mean()) / (stroke_embeddings.std() + 1e-5)

            # Concatenate last feature embedding
            stroke_embeddings = torch.cat([stroke_embeddings, last_feature_embedded], dim=-1)

            # Pad loop_embeddings
            if loop_embeddings.size(0) < self.num_loop_nodes:
                padding_size = self.num_loop_nodes - loop_embeddings.size(0)
                loop_embeddings = F.pad(loop_embeddings, (0, 0, 0, padding_size))
            else:
                loop_embeddings = loop_embeddings[:self.num_loop_nodes]

            # Pad stroke_embeddings
            if stroke_embeddings.size(0) < self.num_stroke_nodes:
                padding_size = self.num_stroke_nodes - stroke_embeddings.size(0)
                stroke_embeddings = F.pad(stroke_embeddings, (0, 0, 0, padding_size))
            else:
                stroke_embeddings = stroke_embeddings[:self.num_stroke_nodes]

            # Reshape to batch_size=1
            loop_embeddings = loop_embeddings.unsqueeze(0)
            stroke_embeddings = stroke_embeddings.unsqueeze(0)

        # Decode each node separately
        loop_scores = self.loop_decoder(loop_embeddings)  # Shape: [batch_size, num_loop_nodes, num_classes]
        stroke_scores = self.stroke_decoder(stroke_embeddings)  # Shape: [batch_size, num_stroke_nodes, num_classes]

        # Aggregate over all nodes instead of summing (to normalize output scale)
        loop_graph_score = loop_scores.mean(dim=1)  # Shape: [batch_size, num_classes]
        stroke_graph_score = stroke_scores.mean(dim=1)  # Shape: [batch_size, num_classes]

        # Combine scores from loops and strokes
        combined_score = loop_graph_score + stroke_graph_score  # Shape: [batch_size, num_classes]

        return combined_score  # Raw logits for CrossEntropyLoss




class Stroke_type_Decoder(nn.Module):
    def __init__(self, hidden_channels=256):
        super(Stroke_type_Decoder, self).__init__()

        self.decoder = nn.Sequential(
            nn.Linear(128, hidden_channels),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.1),
            nn.Linear(hidden_channels, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.1),
            nn.Linear(64, 1),

        )

    def forward(self, x_dict):
        return torch.sigmoid(self.decoder(x_dict['stroke']))




class Program_Decoder(nn.Module):
    def __init__(self, embed_dim=128, num_heads=8, ff_dim=256, num_classes=10, dropout=0.1, num_layers=4):
        super(Program_Decoder, self).__init__()
        
        # Cross-attention layers for stroke and loop nodes
        self.cross_attn_blocks_stroke = nn.ModuleList([
            nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True) for _ in range(num_layers)
        ])
        self.cross_attn_blocks_loop = nn.ModuleList([
            nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True) for _ in range(num_layers)
        ])
        
        # Feed-forward and normalization layers for each block
        self.ff_blocks_stroke = nn.ModuleList([self._build_ff_block(embed_dim, ff_dim, dropout) for _ in range(num_layers)])
        self.norm_blocks_stroke = nn.ModuleList([nn.LayerNorm(embed_dim) for _ in range(num_layers)])
        
        self.ff_blocks_loop = nn.ModuleList([self._build_ff_block(embed_dim, ff_dim, dropout) for _ in range(num_layers)])
        self.norm_blocks_loop = nn.ModuleList([nn.LayerNorm(embed_dim) for _ in range(num_layers)])

        # Self-attention for program and concatenated graph features
        self.self_attn_program = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)
        self.self_attn_graph = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)
        
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(embed_dim, num_classes)

        # Program encoder with a CLS token
        self.program_encoder = ProgramEncoder()
        self.cls_token = nn.Parameter(torch.randn(1, 1, embed_dim))  # CLS token as a learnable parameter

    def _build_ff_block(self, embed_dim, ff_dim, dropout):
        """Creates a feed-forward block with dropout."""
        return nn.Sequential(
            nn.Linear(embed_dim, ff_dim),
            nn.ReLU(),
            nn.Linear(ff_dim, embed_dim),
            nn.Dropout(dropout)
        )

    def forward(self, x_dict, program_tokens):
        # Encode the program tokens and prepend the CLS token
        program_embedding = self.program_encoder(program_tokens)  # (batch_size, seq_len, embed_dim)
        attn_output_program, _ = self.self_attn_program(program_embedding, program_embedding, program_embedding)
        program_cls_output = attn_output_program[:, 0, :]  # CLS token for program

        # Process stroke node embeddings
        num_strokes = x_dict['stroke'].shape[0]
        batch_size_stroke = max(1, num_strokes // 400)  # Ensure batch_size is at least 1
        node_features_stroke = x_dict['stroke'].view(batch_size_stroke, min(400, num_strokes), 128)

        # Process loop node embeddings
        num_loops = x_dict['loop'].shape[0]
        batch_size_loop = max(1, num_loops // 400)  # Ensure batch_size is at least 1
        node_features_loop = x_dict['loop'].view(batch_size_loop, min(400, num_loops), 128)

        # Pass through each cross-attention and feed-forward block for stroke nodes
        out_stroke = program_embedding
        for attn_layer, ff_layer, norm_layer in zip(self.cross_attn_blocks_stroke, self.ff_blocks_stroke, self.norm_blocks_stroke):
            attn_output_stroke, _ = attn_layer(out_stroke, node_features_stroke, node_features_stroke)
            out_stroke = norm_layer(out_stroke + attn_output_stroke)
            out_stroke = norm_layer(out_stroke + ff_layer(out_stroke))
        
        # Pass through each cross-attention and feed-forward block for loop nodes
        out_loop = program_embedding
        for attn_layer, ff_layer, norm_layer in zip(self.cross_attn_blocks_loop, self.ff_blocks_loop, self.norm_blocks_loop):
            attn_output_loop, _ = attn_layer(out_loop, node_features_loop, node_features_loop)
            out_loop = norm_layer(out_loop + attn_output_loop)
            out_loop = norm_layer(out_loop + ff_layer(out_loop))

        # Concatenate stroke and loop embeddings for graph self-attention
        combined_graph_features = torch.cat([out_stroke, out_loop], dim=1)  # (batch_size, combined_seq_len, embed_dim)
        attn_output_graph, _ = self.self_attn_graph(combined_graph_features, combined_graph_features, combined_graph_features)
        graph_cls_output = attn_output_graph[:, 0, :]  # CLS token output from graph features

        # Weighted combination of program and graph CLS outputs
        combined_output =  program_cls_output + graph_cls_output

        # Classification
        logits = self.classifier(combined_output)

        return logits


#---------------------------------- Loss Function ----------------------------------#

class FocalLoss(nn.Module):
    def __init__(self, alpha=1.0, gamma=2.0):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma 

    def forward(self, probs, targets):        
        # Compute binary cross-entropy loss but do not reduce it
        BCE_loss = F.binary_cross_entropy(probs, targets, reduction='none')

        pt = torch.exp(-BCE_loss)  # Probability of the true class
        focal_loss = self.alpha * ((1 - pt) ** self.gamma) * BCE_loss

        return focal_loss.mean()


class ProgramEncoder(nn.Module):
    def __init__(self, vocab_size=20, embedding_dim=64, hidden_dim=128):
        super(ProgramEncoder, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=-1)
        self.positional_encoding = nn.Parameter(torch.randn(20, embedding_dim))  # Add learnable positional encoding
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, x):
        embedded = self.embedding(x) + self.positional_encoding[:x.size(1)]
        lstm_out, _ = self.lstm(embedded)
        final_output = self.fc(lstm_out)  # Transform each timestep for cross-attention
        return final_output



def entropy_penalty(logits):
    probs = torch.softmax(logits, dim=-1)
    entropy = -torch.sum(probs * torch.log(probs + 1e-8), dim=-1)
    return entropy.mean()
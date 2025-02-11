
import torch
import torch.nn as nn

from torch.nn import TransformerEncoder, TransformerEncoderLayer


class StrokeEmbeddingNetwork(nn.Module):
    def __init__(self, input_dim=6, embedding_dim=16):
        super(StrokeEmbeddingNetwork, self).__init__()
        self.fc1 = nn.Linear(input_dim, 32)
        self.fc2 = nn.Linear(32, embedding_dim)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x


class LoopEmbeddingNetwork(nn.Module):
    def __init__(self, input_dim=24, hidden_dim=32, output_dim=32, num_heads=2, num_layers=2):
        super(LoopEmbeddingNetwork, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        # Transformer encoder layer
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=input_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim,
            batch_first=True  
        )

        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Fully connected output layer
        self.fc_output = nn.Linear(input_dim, output_dim)  # The output layer to reduce to output_dim

    def forward(self, loop_features, mask_loop_features):
        # Apply mask to ensure that the transformer ignores the padded positions
        # Mask needs to be inverted for PyTorch's transformer (0s for valid, -inf for padding)
        mask = (mask_loop_features == 0).to(torch.bool)

        # Transformer encoder
        x = self.transformer_encoder(loop_features, src_key_padding_mask=mask)

        # Output layer
        face_embeddings = self.fc_output(x)  # shape: (batch_size, max_num_loops, output_dim)

        return face_embeddings




class LoopConnectivityDecoder(nn.Module):
    def __init__(self, embedding_dim=32, hidden_dim=64):
        super(LoopConnectivityDecoder, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(embedding_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )

    def forward(self, loop_embeddings, mask_loop_features):
        """
        Given loop embeddings, produce a connectivity matrix.
        Args:
            loop_embeddings (torch.Tensor): Tensor of shape (batch_size, max_num_loops, embedding_dim)
            mask_loop_features (torch.Tensor): Tensor of shape (batch_size, max_num_loops)
        Returns:
            connectivity_matrix (torch.Tensor): Tensor of shape (batch_size, max_num_loops, max_num_loops) with values in {0, 1}
        """
        batch_size, max_num_loops, embedding_dim = loop_embeddings.size()

        # Expand the embeddings to compute pairwise combinations in a batched manner
        loop_embeddings_i = loop_embeddings.unsqueeze(2).expand(-1, -1, max_num_loops, -1)  # Shape: (batch_size, max_num_loops, max_num_loops, embedding_dim)
        loop_embeddings_j = loop_embeddings.unsqueeze(1).expand(-1, max_num_loops, -1, -1)  # Shape: (batch_size, max_num_loops, max_num_loops, embedding_dim)

        # Concatenate along the last dimension to create pair embeddings
        pair_embeddings = torch.cat((loop_embeddings_i, loop_embeddings_j), dim=-1)  # Shape: (batch_size, max_num_loops, max_num_loops, 2 * embedding_dim)

        # Flatten the pair embeddings for processing by the feedforward network
        pair_embeddings = pair_embeddings.view(-1, 2 * embedding_dim)  # Shape: (batch_size * max_num_loops * max_num_loops, 2 * embedding_dim)

        # Predict connectivity for each pair
        connectivity_scores = self.fc(pair_embeddings).view(batch_size, max_num_loops, max_num_loops)  # Shape: (batch_size, max_num_loops, max_num_loops)

        # Apply mask to the connectivity matrix to ignore padding
        mask = mask_loop_features.unsqueeze(-1) * mask_loop_features.unsqueeze(-2)  # Shape: (batch_size, max_num_loops, max_num_loops)
        connectivity_matrix = connectivity_scores * mask

        return connectivity_matrix

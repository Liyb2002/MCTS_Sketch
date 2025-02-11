import torch
import torch.nn as nn
import torch.nn.functional as F

class SBGCN_Decoder(nn.Module):
    def __init__(self, input_dim = 32, hidden_dim = 64, output_dim = 32):
        super(SBGCN_Decoder, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, output_dim)

        self.output_dim = output_dim
    
    def forward(self, x, y, z):
        # Concatenate the three sets of embeddings along the feature dimension
        concatenated_embeddings = torch.cat((x, y, z), dim=0)
        
        # Apply fully connected layers
        out = F.relu(self.fc1(concatenated_embeddings))
        out = F.relu(self.fc2(out))
        out = self.fc3(out)

        out = torch.matmul(out, out.transpose(0, 1))  # Multiply the output by its transpose
        
        out = torch.sigmoid(out)
        
        return out

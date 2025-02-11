import torch
import torch_scatter
from torch.nn import Linear, Sequential, ModuleList, BatchNorm1d, Dropout, LeakyReLU, ReLU
import torch_geometric as tg
from Preprocessing.SBGCN.uvnet_encoders import UVNetCurveEncoder, UVNetSurfaceEncoder

class FaceEdgeVertexGCN(torch.nn.Module):
    def __init__(self,
                 f_in_width=12,
                 e_in_width=6,
                 v_in_width=3,
                 out_width=32,
                 k=4,
                 use_uvnet_features=False,
                 srf_in_dim=[0, 1, 2, 3, 4, 5, 8],
                 srf_emb_dim=64):
        super().__init__()
        self.use_uvnet_features = use_uvnet_features
        self.srf_in_dim = srf_in_dim
        if use_uvnet_features:
            self.srf_emb_dim = srf_emb_dim
            # Assuming you have a UVNetSurfaceEncoder class for encoding surface features
            self.surf_encoder = UVNetSurfaceEncoder(
                in_channels=len(srf_in_dim), output_dims=srf_emb_dim
            )
            f_in_width += srf_emb_dim
            e_in_width += srf_emb_dim

        # Embedding layers for face, edge, and vertex features
        self.embed_f_in = LinearBlock(f_in_width, out_width)
        self.embed_e_in = LinearBlock(e_in_width, out_width)
        self.embed_v_in = LinearBlock(v_in_width, out_width)

        # BipartiteResMRConv layers for message passing between face-edge and edge-vertex
        self.F2E = BipartiteResMRConv(out_width)
        self.E2V = BipartiteResMRConv(out_width)

        # Additional message passing layers
        self.ffLayers = ModuleList()
        for i in range(1):
            self.ffLayers.append(BipartiteResMRConv(out_width))
        
        self.F2F = BipartiteResMRConv(out_width)

        # BipartiteResMRConv layers for message passing between edge-face and vertex-edge
        self.E2F = BipartiteResMRConv(out_width)
        self.V2E = BipartiteResMRConv(out_width)

    def forward(self, data):
        x_f_raw = data['face'].x
        x_e_raw = data['edge'].x
        x_v_raw = data['vertex'].x
        index_id = data['face'].y

        # Apply input encoders
        x_f_embedding = self.embed_f_in(x_f_raw)
        x_e_embedding = self.embed_e_in(x_e_raw)
        x_v_embedding = self.embed_v_in(x_v_raw)

        # Upward pass (Face to Edge and Edge to Vertex)
        x_e = self.F2E(x_f_embedding, x_e_embedding, data['face', 'connects', 'edge'].edge_index.clone().detach(), index_id)
        x_v = self.E2V(x_e_embedding, x_v_embedding, data['edge', 'connects', 'vertex'].edge_index.clone().detach(), index_id)

        # Meta-Edge Spine
        for conv in self.ffLayers:
            x_f = conv(x_f_embedding, x_f_embedding, data['face', 'connects', 'face'].edge_index.clone().detach(), index_id)

        # Downward pass (Edge to Face and Vertex to Edge)
        x_f = self.E2F(x_e, x_f, data['edge', 'connects', 'face'].edge_index.clone().detach(), index_id)
        x_e = self.V2E(x_v, x_e, data['vertex', 'connects', 'edge'].edge_index.clone().detach(), index_id)

        return x_f, x_e, x_v

    def embed(self, graph):
        face_embeddings, edge_embeddings, vertex_embeddings = self.forward(graph)
        # concatenated_embeddings = torch.cat((face_embeddings, edge_embeddings, vertex_embeddings), dim=0)

        return face_embeddings, edge_embeddings, vertex_embeddings


class BipartiteResMRConv(torch.nn.Module):
    def __init__(self, width):
        super().__init__()
        self.mlp = LinearBlock(2*width, width)
    
    def forward(self, x_src, x_dst, e, index_id):
        
        maxes = torch.zeros_like(x_dst)  
        for edge_tuple in e:

            src_idx = index_id[0][edge_tuple[0]].clone().detach()
            dst_idx = index_id[0][edge_tuple[1]].clone().detach()

            diffs = torch.index_select(x_dst, 1, dst_idx) - torch.index_select(x_src, 1, src_idx)

            max_val, _ = torch_scatter.scatter_max(
                diffs, 
                dst_idx, 
                dim=1, 
                dim_size=x_dst.shape[1]
            )
            
            maxes += max_val

        return x_dst + self.mlp(torch.cat([x_dst, maxes], dim=2))


class LinearBlock(torch.nn.Module):
    def __init__(self, *layer_sizes, batch_norm=False, dropout=0.0, last_linear=False, leaky=True):
        super().__init__()

        layers = []
        for i in range(len(layer_sizes) - 1):
            c_in = layer_sizes[i]
            c_out = layer_sizes[i + 1]

            layers.append(Linear(c_in, c_out))
            if last_linear and i+1 >= len(layer_sizes) - 1:
                break
            if batch_norm:
                layers.append(BatchNorm1d(c_out))
            if dropout > 0:
                layers.append(Dropout(p=dropout))
            layers.append((LeakyReLU() if leaky else ReLU()))

        self.f = torch.nn.Sequential(*layers)

    def forward(self, x):
        return self.f(x)

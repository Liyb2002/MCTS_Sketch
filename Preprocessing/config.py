import torch


# device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ---------------------------------------------------------------------------- #

operation_to_index = {'terminate': 0, 'sketch': 1, 'extrude_addition': 2, 'extrude_subtraction': 3, 'fillet': 4}
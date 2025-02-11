import os
from pathlib import Path
import torch



home_dir = Path(__file__).parent

def save_model(model, model_name="sbgcn_model"):
    save_dir = os.path.join(home_dir, "model_checkpoints", model_name)

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # Define the full path for the model file
    model_file_name = model_name + ".ckpt"
    model_path = os.path.join(save_dir, model_file_name)

    # Save the model
    torch.save(model.state_dict(), model_path)
    print(f"Model saved to {model_path}")


def load_model(model, checkpoint_path):
    if os.path.isfile(checkpoint_path):
        print(f"Loaded model from {checkpoint_path}")
        model.load_state_dict(torch.load(checkpoint_path))
        return model
    else:
        return None

from models.transformer import BasicTransformer
import torch


def load_small_model(device):
    model = BasicTransformer(3015, 50, 2, 500, 1, 0.0).to(device=device)
    model.load_state_dict(torch.load("model_small.pt", map_location=device))
    return model


def load_large_model(device):
    model = BasicTransformer(3015, 50, 5, 500, 4, 0.0).to(device=device)
    model.load_state_dict(torch.load("model_large.pt", map_location=device))
    return model
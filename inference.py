import torch

import models
from training import MODEL_SAVE_PATH, device, val_loader, model

loaded_model = model.to(device=device)
loaded_model.load_state_dict(torch.load(MODEL_SAVE_PATH, map_location=device))


def eval(model, loader):
    with torch.no_grad():
        x = []
        y = []
        for batch, actual in val_loader:
            predicted = model(batch)
            x.append(actual)
            y.append(predicted)

eval(loaded_model, val_loader)
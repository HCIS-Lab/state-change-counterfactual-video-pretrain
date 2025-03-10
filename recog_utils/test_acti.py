from tqdm import tqdm
from utils.text_prompt import *
import torch

@torch.no_grad()
def eval_model(model, data_loader, device):
    # Evaluate the model on data from valloader
    correct = 0
    total = 0
    model.eval()

    assert not torch.is_grad_enabled(), "grad is enabled during inference"

    for img, labels in data_loader:
        images, labels = images.to(device), labels.to(device)
        preds = model(images)

        total += labels.size(0)
        _, predicted = torch.max(preds.data, 1)
        correct += (predicted == labels).sum().item()

    assert not torch.is_grad_enabled(), "grad is enabled during inference"
    return 0.0, 100 * correct / total
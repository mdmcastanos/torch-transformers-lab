import torch

def pick_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")

d = pick_device()
x = torch.randn(1024, 1024, device=d)
y = x @ x.T
print("device:", d, "mean:", y.mean().item())

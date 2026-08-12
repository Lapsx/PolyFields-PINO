import torch
import torch.nn as nn

device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = nn.Linear(1, 1).to(device)

v_tensor = torch.tensor([1.0], dtype=torch.float32)
v_tensor.requires_grad_(True)

inputs = v_tensor.to(device)

out = model(inputs)
S = torch.sum(out**2)

try:
    grad = torch.autograd.grad(S, v_tensor)[0]
    print("Success:", grad)
except Exception as e:
    print("Error:", e)

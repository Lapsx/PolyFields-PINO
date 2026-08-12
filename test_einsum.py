import torch
import torch.nn as nn

class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.w = nn.Parameter(torch.rand(8, 8, 16, 16, dtype=torch.cfloat))
    def forward(self, x):
        x_ft = torch.fft.rfft2(x)
        return torch.einsum("bixy,ioxy->boxy", x_ft[:, :, :16, :16], self.w)

model = nn.DataParallel(Model()).cuda()
x = torch.rand(32, 8, 64, 64).cuda()
out = model(x)
print(out.shape)

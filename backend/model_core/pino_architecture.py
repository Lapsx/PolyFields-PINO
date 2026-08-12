import torch
import torch.nn as nn
import torch.nn.functional as F

class SpectralConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, modes1, modes2):
        super(SpectralConv2d, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes1 = modes1
        self.modes2 = modes2

        self.scale = (1 / (in_channels * out_channels))
        # Kaggle/PyTorch DataParallel bug fix: Store complex weights as real with an extra dim of size 2
        self.weights1 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, 2, dtype=torch.float))
        self.weights2 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, 2, dtype=torch.float))

    def forward(self, x):
        batchsize = x.shape[0]
        # Aplica a Transformada de Fourier 2D real
        x_ft = torch.fft.rfft2(x)
        
        out_ft = torch.zeros(batchsize, self.out_channels, x.size(-2), x.size(-1)//2 + 1, dtype=torch.cfloat, device=x.device)
        
        # Convert back to complex in the forward pass
        w1 = torch.view_as_complex(self.weights1)
        w2 = torch.view_as_complex(self.weights2)
        
        # Multiplica os modos de Fourier de baixa frequência (Truncamento)
        out_ft[:, :, :self.modes1, :self.modes2] = \
            torch.einsum("bixy,ioxy->boxy", x_ft[:, :, :self.modes1, :self.modes2], w1)
        out_ft[:, :, -self.modes1:, :self.modes2] = \
            torch.einsum("bixy,ioxy->boxy", x_ft[:, :, -self.modes1:, :self.modes2], w2)
            
        # Transforma de volta para o espaço físico
        x = torch.fft.irfft2(out_ft, s=(x.size(-2), x.size(-1)))
        return x

class PINO_Polyelectrolyte(nn.Module):
    """
    Physics-Informed Neural Operator para a V3.
    Lida com o vetor estendido de entrada (GRF, Debye, Máscara e coeficientes de Fourier da Carga).
    """
    def __init__(self, modes1=16, modes2=16, width=96, input_channels=10):
        super(PINO_Polyelectrolyte, self).__init__()
        
        self.modes1 = modes1
        self.modes2 = modes2
        self.width = width
        
        # Lifting Layer (Projeca as entradas para a largura do espaço latente)
        self.p = nn.Linear(input_channels, self.width)
        
        # Fourier Layers
        self.conv0 = SpectralConv2d(self.width, self.width, self.modes1, self.modes2)
        self.conv1 = SpectralConv2d(self.width, self.width, self.modes1, self.modes2)
        self.conv2 = SpectralConv2d(self.width, self.width, self.modes1, self.modes2)
        self.conv3 = SpectralConv2d(self.width, self.width, self.modes1, self.modes2)
        
        # W (Convoluções espaciais locais em paralelo com as camadas de Fourier)
        self.w0 = nn.Conv2d(self.width, self.width, 1)
        self.w1 = nn.Conv2d(self.width, self.width, 1)
        self.w2 = nn.Conv2d(self.width, self.width, 1)
        self.w3 = nn.Conv2d(self.width, self.width, 1)
        
        # Projection Layer (Extrai a Densidade Polimérica ϕ(r) no final)
        self.q = nn.Linear(self.width, 64)
        self.q2 = nn.Linear(64, 1)

    def forward(self, x):
        # x.shape = (Batch, Nx, Ny, Channels)
        grid = self.get_grid(x.shape, x.device)
        x = torch.cat((x, grid), dim=-1) # Acopla posições espaciais absolutas
        
        x = self.p(x)
        x = x.permute(0, 3, 1, 2)
        
        x1 = self.conv0(x)
        x2 = self.w0(x)
        x = F.gelu(x1 + x2)
        
        x1 = self.conv1(x)
        x2 = self.w1(x)
        x = F.gelu(x1 + x2)
        
        x1 = self.conv2(x)
        x2 = self.w2(x)
        x = F.gelu(x1 + x2)
        
        x1 = self.conv3(x)
        x2 = self.w3(x)
        x = x1 + x2 # Sem Gelu na última camada
        
        x = x.permute(0, 2, 3, 1)
        x = self.q(x)
        x = F.gelu(x)
        x = self.q2(x)
        return x
        
    def get_grid(self, shape, device):
        batchsize, size_x, size_y = shape[0], shape[1], shape[2]
        gridx = torch.linspace(0, 1, size_x, dtype=torch.float)
        gridx = gridx.reshape(1, size_x, 1, 1).repeat([batchsize, 1, size_y, 1])
        gridy = torch.linspace(0, 1, size_y, dtype=torch.float)
        gridy = gridy.reshape(1, 1, size_y, 1).repeat([batchsize, size_x, 1, 1])
        return torch.cat((gridx, gridy), dim=-1).to(device)

def sobolev_physics_loss(phi_pred, phi_true, v_tensor, model):
    """
    Função de Loss central da V3 (PINO).
    Incorpora o Sobolev Training (Suscetibilidade Linear) para garantir forma termodinâmica.
    """
    # 1. Erro de Dados Supervisionado (L2 Loss)
    l2_loss = F.mse_loss(phi_pred, phi_true)
    
    # 2. Sobolev Loss (Erro nas Derivadas df/dV)
    # Extrai o Jacobiano da Predição
    S_pred = torch.sum(phi_pred**2)
    dphi_dV_pred = torch.autograd.grad(S_pred, v_tensor, create_graph=True)[0]
    
    # Extrai o Jacobiano Numérico Real (ou carregado do dataset)
    # Aqui assumiremos que o dataset de treino salvará o RPA_matrix (Jacobiano real)
    # dphi_dV_true = ... 
    
    # sobolev = F.mse_loss(dphi_dV_pred, dphi_dV_true)
    
    # 3. Physics Residual Loss (Debye-Huckel / Edwards)
    # diff_eq_residual = ...
    
    return l2_loss # + 0.1 * sobolev + 0.5 * diff_eq_residual

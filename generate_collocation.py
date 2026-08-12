import torch
import numpy as np

def generate_collocation_dataset(output_path="data/pino_v3_collocation.pt", num_samples=20000, N=100):
    print(f"[*] Gerando {num_samples} cenários físicos VAZIOS (Collocation Points)...")
    
    # Random parameters
    # b: 0.5 to 2.5
    # kappa: 0.5 to 2.5
    # u: -1.0 to 1.0
    params = torch.empty((num_samples, 3)).uniform_(0.0, 1.0)
    params[:, 0] = params[:, 0] * 2.0 + 0.5
    params[:, 1] = params[:, 1] * 2.0 + 0.5
    params[:, 2] = params[:, 2] * 2.0 - 1.0
    
    # Cargas do Polímero (c1 e c4) contínuas de 0 a 2.0
    c_params = torch.empty((num_samples, 2)).uniform_(0.0, 2.0)
    
    # Campos base (ruído inicial para o campo elétrico)
    # Para acelerar, vamos gerar campos uniformes aleatórios em vez de GRFs complexos, 
    # pois a física deve valer para qualquer campo V!
    print("[*] Sintetizando topologias V(x,z) aleatórias (Smooth Fields)...")
    V_fields = torch.empty((num_samples, N, N)).uniform_(-2.0, 2.0)
    
    # Suavizar um pouco os campos V no próprio PyTorch simulando um filtro de Fourier
    V_fft = torch.fft.fftn(V_fields, dim=(1, 2))
    kx = torch.fft.fftfreq(N).view(1, N, 1).repeat(num_samples, 1, N)
    kz = torch.fft.fftfreq(N).view(1, 1, N).repeat(num_samples, N, 1)
    k_sq = kx**2 + kz**2
    # Filtro Gaussiano de baixa frequência
    filter_mask = torch.exp(-k_sq * 50.0)
    V_fft = V_fft * filter_mask
    V_smooth = torch.fft.ifftn(V_fft, dim=(1, 2)).real
    
    # Normalizar para ter variância razoável
    V_smooth = V_smooth / torch.std(V_smooth, dim=(1,2), keepdim=True) * torch.empty((num_samples, 1, 1)).uniform_(0.5, 3.0)
    
    print(f"[*] Salvando dataset de Collocation em {output_path}...")
    torch.save({
        "V": V_smooth.float(),
        "params": torch.cat([params, c_params], dim=-1).float()
    }, output_path)
    print("[+] Dataset de Física gerado com sucesso! NENHUM RÓTULO DE SCFT FOI USADO.")

if __name__ == "__main__":
    generate_collocation_dataset()

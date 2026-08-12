import torch
import torch.nn as nn
from pino_architecture import PINO_Polyelectrolyte
import os
import torch.nn.functional as F

def physics_loss(phi_pred, V_batch, params):
    # params: [b, kappa, u, c1, c4]
    b_val = params[:, 0].view(-1, 1, 1)
    kappa_val = params[:, 1].view(-1, 1, 1)
    u_val = params[:, 2].view(-1, 1, 1)
    
    # 1. Conservação de Massa (A densidade global deve ser próxima a 1.0)
    dx = 1.0 / 100.0
    mass = torch.sum(phi_pred, dim=(1, 2)) * (dx * dx)
    loss_mass = torch.mean((mass - 1.0)**2)
    
    # 2. Positividade (Densidade não pode ser negativa)
    loss_positivity = torch.mean(torch.relu(-phi_pred)**2)
    
    # Extrair cargas
    c1_val = params[:, 3].view(-1, 1, 1)
    c4_val = params[:, 4].view(-1, 1, 1)
    
    # 3. Derivadas Espaciais via Transformada de Fourier (Spectral Derivatives)
    # A equação aproximada de Ginzburg-Landau para polímeros:
    # b^2 * \nabla^2 phi = V_eff * phi + u * phi^2 - mu * phi
    phi_fft = torch.fft.fftn(phi_pred, dim=(1, 2))
    N = phi_pred.shape[1]
    kx = torch.fft.fftfreq(N).view(1, N, 1).to(phi_pred.device)
    kz = torch.fft.fftfreq(N).view(1, 1, N).to(phi_pred.device)
    k_sq = (kx**2 + kz**2)
    
    laplacian_fft = - (2 * torch.pi * k_sq) * phi_fft
    laplacian_phi = torch.fft.ifftn(laplacian_fft, dim=(1, 2)).real
    
    # Gradiente de V (para o polímero alternado)
    V_fft = torch.fft.fftn(V_batch, dim=(1, 2))
    dV_dx = torch.fft.ifftn((1j * 2 * torch.pi * kx) * V_fft, dim=(1, 2)).real
    dV_dz = torch.fft.ifftn((1j * 2 * torch.pi * kz) * V_fft, dim=(1, 2)).real
    grad_V_sq = dV_dx**2 + dV_dz**2
    
    # Construir o Potencial Efetivo (V_eff)
    # Polímero Neutro: Sofre o campo V diretamente
    # Dibloco (c1): Atraído para ALTA MAGNITUDE (abs(V))
    # Alternado (c4): Comportamento de Dipolo, é atraído por altos GRADIENTES do campo elétrico (|nabla V|^2)
    V_eff = V_batch - (c1_val * 2.0 * torch.abs(V_batch)) - (c4_val * 0.1 * grad_V_sq)
    
    # PDE Residual (Simplificada, ignorando multiplicador de Lagrange mu por enquanto)
    # b_val * \nabla^2 phi - (V_eff + u_val * phi_pred) * phi_pred = 0
    pde_residual = (b_val * 0.1) * laplacian_phi - (V_eff + u_val * phi_pred) * phi_pred
    loss_pde = torch.mean(pde_residual**2)
    
    return loss_mass * 100.0 + loss_positivity * 50.0 + loss_pde * 1.0

def train_physics():
    print("[*] Iniciando Aprendizado da Física (Collocation PDE Loss)...")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"[*] Usando device: {device}")
    
    model = PINO_Polyelectrolyte(modes1=16, modes2=16, width=96, input_channels=8).to(device)
    
    # Função auxiliar para limpar prefixo 'module.' e converter complexos legados
    def load_clean_state_dict(m, path, d):
        state_dict = torch.load(path, map_location=d)
        clean_state_dict = {}
        for k, v in state_dict.items():
            k_clean = k.replace('module.', '')
            if v.is_complex():
                v = torch.view_as_real(v)
            clean_state_dict[k_clean] = v
        m.load_state_dict(clean_state_dict)

    if os.path.exists("weights/pino_v3_phase5_physics.pth"):
        load_clean_state_dict(model, "weights/pino_v3_phase5_physics.pth", device)
        print("[+] Pesos da Fase 5 (Física) encontrados! Retomando o treino de onde parou...")
    elif os.path.exists("weights/pino_v3_phase4_final.pth"):
        load_clean_state_dict(model, "weights/pino_v3_phase4_final.pth", device)
        print("[+] Pesos da Fase 4 (Slider Contínuo) carregados para iniciar o treino!")
    else:
        print("[!] Erro: Nenhum peso anterior encontrado!")
        return

    if torch.cuda.device_count() > 1:
        print(f"[*] Acelerando com {torch.cuda.device_count()} GPUs (DataParallel)!")
        model = nn.DataParallel(model)

    print("[*] Carregando pontos de Collocation...")
    data = torch.load("data/pino_v3_collocation.pt")
    V_data = data["V"]
    params_data = data["params"]
    
    dataset = torch.utils.data.TensorDataset(V_data, params_data)
    
    # Escalar batch size de acordo com a quantidade de GPUs
    base_batch_size = 8
    num_gpus = max(1, torch.cuda.device_count())
    effective_batch_size = base_batch_size * num_gpus
    print(f"[*] Tamanho do batch efetivo: {effective_batch_size}")
    loader = torch.utils.data.DataLoader(dataset, batch_size=effective_batch_size, shuffle=True)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    epochs = 5000
    
    # Scheduler para decair o LR pela metade a cada 1000 épocas
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1000, gamma=0.5)
    
    model.train()
    os.makedirs("weights", exist_ok=True)
    for ep in range(epochs):
        ep_loss = 0.0
        for V_batch, p_batch in loader:
            V_batch, p_batch = V_batch.to(device), p_batch.to(device)
            
            # Montar tensores
            N = V_batch.shape[1]
            B = V_batch.shape[0]
            
            # Montar canais: [V, c1, c2, b, kappa, u]
            inputs = torch.zeros(B, N, N, 6, device=device)
            inputs[..., 0] = V_batch
            inputs[..., 1] = p_batch[:, 3].view(B, 1, 1).expand(B, N, N) # c1_plane
            # canal 2 fica zerado por padrão
            inputs[..., 3] = p_batch[:, 0].view(B, 1, 1).expand(B, N, N) # b_plane
            inputs[..., 4] = p_batch[:, 4].view(B, 1, 1).expand(B, N, N) # c4_plane
            inputs[..., 5] = p_batch[:, 2].view(B, 1, 1).expand(B, N, N) # u_plane
            
            optimizer.zero_grad()
            phi_pred = model(inputs).squeeze(-1)
            
            loss = physics_loss(phi_pred, V_batch, p_batch)
            
            loss.backward()
            optimizer.step()
            
            ep_loss += loss.item()
            
        scheduler.step()
        print(f"Época [{ep+1}/{epochs}] - Physics Loss (PDE + Mass): {ep_loss/len(loader):.6f} - LR: {scheduler.get_last_lr()[0]:.2e}")
        
        # Salva o progresso a cada 200 épocas
        if (ep + 1) % 200 == 0:
            state_to_save = model.module.state_dict() if isinstance(model, nn.DataParallel) else model.state_dict()
            torch.save(state_to_save, f"weights/pino_v3_phase5_physics_ep{ep+1}.pth")
            # Salva também por cima do principal para que possa ser retomado (resume) facilmente se o Kaggle cair
            torch.save(state_to_save, "weights/pino_v3_phase5_physics.pth")
            print(f"[+] Checkpoint salvo na época {ep+1}!")

    final_state = model.module.state_dict() if isinstance(model, nn.DataParallel) else model.state_dict()
    torch.save(final_state, "weights/pino_v3_phase5_physics.pth")
    print("[+] Treino da Física Concluído! Pesos salvos em 'weights/pino_v3_phase5_physics.pth'")

if __name__ == "__main__":
    train_physics()

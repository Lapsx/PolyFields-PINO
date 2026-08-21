import torch
import torch.nn as nn
from pino_architecture import PINO_Polyelectrolyte, sobolev_physics_loss
import os

def train_physics():
    print("[*] Iniciando Unified PINO Training (Data L2 + Sobolev + Physics Collocation)...")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"[*] Usando device: {device}")
    
    model = PINO_Polyelectrolyte(modes1=16, modes2=16, width=96, input_channels=8).to(device)
    
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
        print("[+] Pesos anteriores encontrados! Retomando o treino de onde parou...")
    elif os.path.exists("weights/pino_v3_phase4_final.pth"):
        load_clean_state_dict(model, "weights/pino_v3_phase4_final.pth", device)
        print("[+] Pesos carregados para iniciar o treino!")
    else:
        print("[!] Aviso: Nenhum peso anterior encontrado, iniciando do zero.")

    # Removido nn.DataParallel pois ele quebra o autograd (create_graph=True) do Sobolev Loss
    # e causa "double free or corruption" no C++ engine do PyTorch.
    if torch.cuda.device_count() > 1:
        print("[!] Aviso: Várias GPUs detectadas no Kaggle, mas o nn.DataParallel QUEBRA as derivadas de alta ordem.")
        print("[*] Restringindo o treinamento para a GPU principal (cuda:0) para preservar o Grafo Matemático!")

    print("[*] Carregando dataset completo (Ground Truth + Derivadas)...")
    dataset_path = "data/pino_v3_dataset_complete.pt"
    if not os.path.exists(dataset_path):
         print(f"[!] Erro: {dataset_path} não encontrado. Execute scft_solver.py primeiro.")
         return
         
    data = torch.load(dataset_path, weights_only=False)
    V_data = data["V"]
    params_data = data["params"]
    phi_scft_data = data["phi_scft"]
    dphi_dV_data = data["dphi_dV"]
    
    dataset = torch.utils.data.TensorDataset(V_data, params_data, phi_scft_data, dphi_dV_data)
    
    effective_batch_size = 8
    print(f"[*] Tamanho do batch efetivo (Fixado em 1 GPU): {effective_batch_size}")
    loader = torch.utils.data.DataLoader(dataset, batch_size=effective_batch_size, shuffle=True)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    epochs = 5000
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1000, gamma=0.5)
    
    model.train()
    os.makedirs("weights", exist_ok=True)
    for ep in range(epochs):
        ep_l2 = 0.0
        ep_sob = 0.0
        ep_phys = 0.0
        ep_total = 0.0
        
        for V_batch, p_batch, phi_batch, dphi_batch in loader:
            V_batch = V_batch.to(device).requires_grad_(True)
            p_batch = p_batch.to(device)
            phi_batch = phi_batch.to(device)
            dphi_batch = dphi_batch.to(device)
            
            # Se o dataset só tem 3 parâmetros (b, kappa, u), preenche c1 e c4 com zeros
            if p_batch.shape[1] == 3:
                zeros_pad = torch.zeros(p_batch.shape[0], 2, device=device)
                p_batch = torch.cat([p_batch, zeros_pad], dim=1)
                
            N = V_batch.shape[1]
            B = V_batch.shape[0]
            
            # [V, c1, c2(zero), b, c4, u] — kappa está implícito em V (Yukawa)
            inputs = torch.zeros(B, N, N, 6, device=device)
            inputs[..., 0] = V_batch
            inputs[..., 1] = p_batch[:, 3].view(B, 1, 1).expand(B, N, N)
            inputs[..., 3] = p_batch[:, 0].view(B, 1, 1).expand(B, N, N)
            inputs[..., 4] = p_batch[:, 4].view(B, 1, 1).expand(B, N, N)
            inputs[..., 5] = p_batch[:, 2].view(B, 1, 1).expand(B, N, N)
            
            optimizer.zero_grad()
            phi_pred = model(inputs).squeeze(-1)
            
            loss, l2_item, sob_item, phys_item = sobolev_physics_loss(
                phi_pred, phi_batch, V_batch, dphi_batch, p_batch, model, epoch=ep
            )
            
            loss.backward()
            optimizer.step()
            
            ep_total += loss.item()
            ep_l2 += l2_item
            ep_sob += sob_item
            ep_phys += phys_item
            
        scheduler.step()
        n_b = len(loader)
        if (ep+1) % 10 == 0:
            print(f"Ep [{ep+1}/{epochs}] | Total: {ep_total/n_b:.4f} | L2: {ep_l2/n_b:.4f} | Sob: {ep_sob/n_b:.4f} | Phys: {ep_phys/n_b:.4f} | LR: {scheduler.get_last_lr()[0]:.2e}")
        
        if (ep + 1) % 100 == 0:
            state_to_save = model.module.state_dict() if isinstance(model, nn.DataParallel) else model.state_dict()
            torch.save(state_to_save, f"weights/pino_v3_phase5_physics_ep{ep+1}.pth")
            torch.save(state_to_save, "weights/pino_v3_phase5_physics.pth")
            print(f"[+] Checkpoint salvo na época {ep+1}!")

    print("[+] Treino Unificado Concluído! Pesos salvos.")

if __name__ == "__main__":
    train_physics()


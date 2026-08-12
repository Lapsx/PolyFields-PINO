import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import os

from pino_architecture import PINO_Polyelectrolyte

def sobolev_physics_loss(phi_pred, phi_true, V, model, weight_sobolev=0.1):
    # Loss padrão (MSE)
    mse_loss = nn.MSELoss()(phi_pred, phi_true)
    
    # Loss de Sobolev H1 (Derivadas de Primeira Ordem)
    dphi_dx_pred = torch.diff(phi_pred, dim=1)[:, :-1]
    dphi_dx_true = torch.diff(phi_true, dim=1)[:, :-1]
    
    dphi_dy_pred = torch.diff(phi_pred, dim=2)[:, :, :-1]
    dphi_dy_true = torch.diff(phi_true, dim=2)[:, :, :-1]
    
    h1_loss = nn.MSELoss()(dphi_dx_pred, dphi_dx_true) + nn.MSELoss()(dphi_dy_pred, dphi_dy_true)
    
    return mse_loss + weight_sobolev * h1_loss

def train_phase3():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"[*] Usando device: {device}")
    
    # Carregar modelo pré-treinado da Fase 2 (75 épocas)
    model = PINO_Polyelectrolyte(modes1=16, modes2=16, width=96, input_channels=8).to(device)
    try:
        model.load_state_dict(torch.load("weights/pino_v3_sobolev_epoch_75.pth", map_location=device))
        print("[+] Pesos da Fase 2 (Sobolev 75 épocas) carregados com sucesso!")
    except Exception as e:
        print("[!] Erro ao carregar pesos da Fase 2. Certifique-se de que o arquivo existe.")
        return
        
    print("[*] Carregando dataset Phase 3 (com perturbações eletrostáticas)...")
    dataset_path = "data/pino_v3_dataset_phase3.pt"
    if not os.path.exists(dataset_path):
        print("[!] Dataset Phase 3 não encontrado. Execute augment_dataset_phase3.py primeiro.")
        return
        
    data = torch.load(dataset_path)
    V_fields = data["V"]
    params = data["params"]
    phi_scft = data["phi_scft"]
    
    batch_size = 16
    epochs = 25 # Poucas épocas pois a rede já sabe quase tudo!
    
    dataset = TensorDataset(V_fields, params, phi_scft)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    optimizer = optim.AdamW(model.parameters(), lr=5e-4, weight_decay=1e-4) # LR menor para finetuning
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    
    model.train()
    for epoch in range(epochs):
        epoch_loss = 0.0
        for V_batch, params_batch, phi_batch in dataloader:
            V_batch = V_batch.to(device).requires_grad_(True)
            phi_batch = phi_batch.to(device)
            
            # Montar canais: [V, c1, c2, b, kappa, u]
            grid_size = 100
            inputs = torch.zeros(V_batch.shape[0], grid_size, grid_size, 6, device=device)
            inputs[..., 0] = V_batch
            
            # Injetar c1 (Diblock) no canal 1
            c1_plane = params_batch[:, 3].unsqueeze(1).unsqueeze(2).expand(-1, grid_size, grid_size).to(device)
            inputs[..., 1] = c1_plane
            
            # Injetar c4 (Alternado) no canal 4
            c4_plane = params_batch[:, 4].unsqueeze(1).unsqueeze(2).expand(-1, grid_size, grid_size).to(device)
            inputs[..., 4] = c4_plane
            
            b_plane = params_batch[:, 0].unsqueeze(1).unsqueeze(2).expand(-1, grid_size, grid_size).to(device)
            inputs[..., 3] = b_plane
            
            optimizer.zero_grad()
            
            # Forward
            phi_pred = model(inputs).squeeze(-1)
            
            # Physics-Informed Sobolev Loss
            loss = sobolev_physics_loss(phi_pred, phi_batch, V_batch, model)
            
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            
        scheduler.step()
        
        if (epoch+1) % 5 == 0:
            print(f"Época [{epoch+1}/{epochs}] - Loss Phase 3: {epoch_loss/len(dataloader):.6f}", flush=True)
            
    os.makedirs("weights", exist_ok=True)
    torch.save(model.state_dict(), "weights/pino_v3_phase3_final.pth")
    print("\n[+] Finetuning Phase 3 Concluído! Pesos salvos em 'weights/pino_v3_phase3_final.pth'")

if __name__ == "__main__":
    train_phase3()

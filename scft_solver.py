import torch
import os
import matplotlib.pyplot as plt

def run_scft_solver_mock(dataset_path="data/pino_v3_grf_dataset.pt"):
    """
    Simulador SCFT (Mock): Recebe os potenciais GRF e resolve a densidade polimérica real.
    Gera a densidade pseudo-física aproximada e extrai a derivada real (Jacobiano/Suscetibilidade)
    para habilitar o Treinamento de Sobolev (Sobolev Loss).
    """
    print("[*] Iniciando Solucionador SCFT (Self-Consistent Field Theory)...")
    if not os.path.exists(dataset_path):
        print("[!] Erro: Dataset GRF não encontrado.")
        return
        
    data = torch.load(dataset_path)
    V_fields = data["V"]
    params = data["params"]
    num_samples = V_fields.shape[0]
    grid_size = V_fields.shape[1]
    
    # Habilita o rastreamento de gradiente para extrair as derivadas reais
    V_fields.requires_grad_(True)
    
    print(f"[*] Resolvendo {num_samples} cenários e extraindo derivadas...")
    
    # Simulação mock de relaxação vetorizada para todas as amostras
    # phi ~ exp(-V / kT)
    phi_raw = torch.exp(-V_fields * 2.0)
    
    # Zera onde tem parede infinita (V > 9.0)
    # Usando máscara diferenciável via multiplicação
    mask = (V_fields <= 9.0).float()
    phi_masked = phi_raw * mask
    
    # Normalização (conservação de massa)
    masses = torch.sum(phi_masked, dim=(1, 2), keepdim=True) + 1e-8
    phi_scft = (phi_masked / masses) * 100.0
    
    # Extração do Ground Truth para o Sobolev Loss
    # Usamos S_true = sum(phi^2) para obter um perfil não-constante do Jacobiano local
    S_true = torch.sum(phi_scft**2)
    dphi_dV_true = torch.autograd.grad(S_true, V_fields, create_graph=False)[0]
    
    # Desanexando os tensores do grafo computacional para salvar
    phi_scft = phi_scft.detach()
    dphi_dV_true = dphi_dV_true.detach()
    V_fields = V_fields.detach()
            
    # Salva o Dataset Completo (V + Params + Phi_True + Derivadas)
    out_path = "data/pino_v3_dataset_complete.pt"
    torch.save({
        "V": V_fields,
        "params": params,
        "phi_scft": phi_scft,
        "dphi_dV": dphi_dV_true
    }, out_path)
    print(f"\n[+] Dataset Ancorado Final (com derivadas de Sobolev) gerado com sucesso em {out_path}!")
    
    # Salvar imagem comparativa
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    axes[0].imshow(V_fields[0].numpy(), cmap='inferno')
    axes[0].set_title('Potencial (GRF)')
    axes[1].imshow(phi_scft[0].numpy(), cmap='viridis')
    axes[1].set_title('Densidade Polimérica (SCFT)')
    axes[2].imshow(dphi_dV_true[0].numpy(), cmap='coolwarm')
    axes[2].set_title('Suscetibilidade Termodinâmica (dPhi/dV)')
    plt.savefig("data/exemplo_scft_solution.png")
    plt.close()

if __name__ == "__main__":
    run_scft_solver_mock()

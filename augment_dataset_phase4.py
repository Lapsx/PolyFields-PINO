import torch
import os
import numpy as np

def augment_dataset(input_path="data/pino_v3_dataset_complete.pt", output_path="data/pino_v3_dataset_phase4.pt"):
    print("[*] Carregando dataset original para Phase 4 (Slider Contínuo)...")
    if not os.path.exists(input_path):
        print(f"[!] Erro: {input_path} não encontrado!")
        return

    data = torch.load(input_path)
    V_fields = data["V"]        # (N, 100, 100)
    phi_scft = data["phi_scft"] # (N, 100, 100)
    params = data["params"]     # (N, 3)

    N_samples = V_fields.shape[0]
    
    print("[*] Duplicando dataset com intensidades eletrostáticas CONTÍNUAS...")
    
    # 1. Neutro
    V_0 = V_fields.clone()
    phi_0 = phi_scft.clone()
    c_0 = torch.zeros((N_samples, 2), dtype=torch.float32)

    # 2. Diblock Contínuo (c1 varia de 0.0 a 2.0)
    V_1 = V_fields.clone()
    # Sorteio uniforme de intensidades
    intensities_1 = torch.empty((N_samples, 1)).uniform_(0.0, 2.0)
    # Reformatação para multiplicação espacial (N_samples, 1, 1)
    int_1_spatial = intensities_1.unsqueeze(-1)
    
    # A perturbação escala com a intensidade. Para Diblock, a densidade deve subir nos DOIS polos (atração dupla)
    perturbation_1 = 1.0 + (0.6 * int_1_spatial) * torch.abs(torch.tanh(V_fields * 2.0))
    phi_1 = phi_scft * perturbation_1
    
    c_1 = torch.zeros((N_samples, 2), dtype=torch.float32)
    c_1[:, 0] = intensities_1.squeeze()

    # 3. Alternado Contínuo (c4 varia de 0.0 a 2.0)
    V_4 = V_fields.clone()
    intensities_4 = torch.empty((N_samples, 1)).uniform_(0.0, 2.0)
    int_4_spatial = intensities_4.unsqueeze(-1)
    
    # Para o alternado, ocorre um micro-pinning (ondulações que acompanham a força do campo)
    perturbation_4 = 1.0 + (0.4 * int_4_spatial) * torch.sin(10.0 * V_fields) * torch.abs(torch.tanh(V_fields))
    phi_4 = phi_scft * perturbation_4
    
    c_4 = torch.zeros((N_samples, 2), dtype=torch.float32)
    c_4[:, 1] = intensities_4.squeeze()

    # Concatenar
    V_final = torch.cat([V_0, V_1, V_4], dim=0)
    phi_final = torch.cat([phi_0, phi_1, phi_4], dim=0)
    
    params_final = torch.cat([
        torch.cat([params, c_0], dim=-1),
        torch.cat([params, c_1], dim=-1),
        torch.cat([params, c_4], dim=-1)
    ], dim=0)

    print(f"[+] Dataset Phase 4 criado com {V_final.shape[0]} amostras interpoláveis!")
    
    print(f"[*] Salvando dataset Phase 4 em {output_path}...")
    torch.save({
        "V": V_final,
        "phi_scft": phi_final,
        "params": params_final
    }, output_path)
    
    print("[+] Concluído! O polímero agora possui intensidade de carga variável.")

if __name__ == "__main__":
    augment_dataset()

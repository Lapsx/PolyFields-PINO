import torch
import os
import numpy as np

def augment_dataset(input_path="data/pino_v3_dataset_complete.pt", output_path="data/pino_v3_dataset_phase3.pt"):
    print("[*] Carregando dataset original para Phase 3 (Cargas no Polímero)...")
    if not os.path.exists(input_path):
        print(f"[!] Erro: {input_path} não encontrado!")
        return

    data = torch.load(input_path)
    V_fields = data["V"]        # (N, 100, 100)
    phi_scft = data["phi_scft"] # (N, 100, 100)
    params = data["params"]     # (N, 3)

    N_samples = V_fields.shape[0]
    
    # Vamos gerar 3 cópias do dataset:
    # 1. Neutro (c1=0, c4=0) -> sem mudança
    # 2. Diblock (c1=1, c4=0) -> perturbação baseada em baixas frequências
    # 3. Alternado (c1=0, c4=1) -> perturbação baseada em altas frequências

    print("[*] Duplicando dataset com perturbações eletrostáticas simuladas...")
    
    # 1. Neutro
    V_0 = V_fields.clone()
    phi_0 = phi_scft.clone()
    c_0 = torch.zeros((N_samples, 2), dtype=torch.float32)

    # 2. Diblock (c1=1)
    V_1 = V_fields.clone()
    # Física simplificada: O polímero Diblock sofre polarização no campo V. 
    # Ele se estica (aumenta o volume aparente / espalha a densidade) ou se concentra depedendo do sinal de V.
    # Proxy termodinâmico rápido para aprendizado da rede:
    perturbation_1 = 1.0 + 0.3 * torch.tanh(V_fields)
    phi_1 = phi_scft * perturbation_1
    c_1 = torch.zeros((N_samples, 2), dtype=torch.float32)
    c_1[:, 0] = 1.0

    # 3. Alternado (c4=1)
    V_4 = V_fields.clone()
    # Física simplificada: O polímero Alternado (cargas ++--++--) sofre um "micro-pinning". 
    # A densidade local fica mais "espetada" (concentração mais forte nos picos de V).
    perturbation_4 = 1.0 - 0.2 * torch.sin(5.0 * V_fields)
    phi_4 = phi_scft * perturbation_4
    c_4 = torch.zeros((N_samples, 2), dtype=torch.float32)
    c_4[:, 1] = 1.0

    # Concatenar tudo
    V_final = torch.cat([V_0, V_1, V_4], dim=0)
    phi_final = torch.cat([phi_0, phi_1, phi_4], dim=0)
    
    # params antigo tinha 3 canais (b, kappa, u). Agora adicionamos (c1, c4) -> 5 canais
    params_final = torch.cat([
        torch.cat([params, c_0], dim=-1),
        torch.cat([params, c_1], dim=-1),
        torch.cat([params, c_4], dim=-1)
    ], dim=0)

    print(f"[+] Dataset Phase 3 criado com {V_final.shape[0]} amostras!")
    
    print(f"[*] Salvando dataset Phase 3 em {output_path}...")
    torch.save({
        "V": V_final,
        "phi_scft": phi_final,
        "params": params_final
    }, output_path)
    
    print("[+] Concluído! O polímero agora possui elétrons em seu esqueleto.")

if __name__ == "__main__":
    augment_dataset()

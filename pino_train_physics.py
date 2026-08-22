import torch
import torch.nn as nn
from pino_architecture import (PINO_Polyelectrolyte, sobolev_physics_loss,
                               unpack_params, amplitude_to_channel)
import os
import time

# Kaggle mata o kernel em 12h. Salva por tempo (além de por época) para nunca
# perder mais que SAVE_EVERY_MIN de treino.
SAVE_EVERY_MIN = 20.0
TRAIN_STATE = "weights/training_state.pth"   # model + optimizer + época (NÃO usado pelo backend)
BACKEND_WEIGHTS = "weights/pino_v3_phase5_physics.pth"  # state_dict puro (lido por backend/main.py)


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

    # start_epoch preserva o schedule de LR e o warm-up da física entre sessões
    # do Kaggle. Sem isso, cada sessão de 12h recomeça em alpha_phys=0 e LR=1e-4.
    start_epoch = 0
    resume_state = None
    if os.path.exists(TRAIN_STATE):
        resume_state = torch.load(TRAIN_STATE, map_location=device, weights_only=False)
        model.load_state_dict(resume_state["model"])
        start_epoch = resume_state["epoch"]
        print(f"[+] Estado de treino encontrado! Retomando da época {start_epoch} "
              f"(LR e warm-up da física preservados).")
    elif os.path.exists(BACKEND_WEIGHTS):
        load_clean_state_dict(model, BACKEND_WEIGHTS, device)
        print("[+] Pesos anteriores encontrados! Retomando o treino de onde parou...")
        print("[!] Sem training_state.pth: contador de época reinicia em 0.")
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

    # Descarta amostras colapsadas: quando o poço efetivo é muito profundo e a
    # repulsão (u) é fraca, o estado fundamental localiza abaixo da resolução da
    # grade e vira um delta de Dirac num pixel. É artefato de rede, não física, e
    # a rede não tem como reproduzir sem aprender ruído.
    massa = phi_scft_data.sum(dim=(1, 2))
    frac_pico = phi_scft_data.amax(dim=(1, 2)) / (massa + 1e-12)
    ok = frac_pico <= 0.5
    n_drop = int((~ok).sum())
    if n_drop:
        print(f"[!] Descartando {n_drop}/{len(ok)} amostras colapsadas "
              f"(>50% da massa num pixel).")
        V_data, params_data = V_data[ok], params_data[ok]
        phi_scft_data, dphi_dV_data = phi_scft_data[ok], dphi_dV_data[ok]

    print(f"[*] Treinando com {len(V_data)} amostras | params: {params_data.shape[1]} colunas")
    dataset = torch.utils.data.TensorDataset(V_data, params_data, phi_scft_data, dphi_dV_data)
    
    effective_batch_size = 32
    print(f"[*] Tamanho do batch efetivo (Fixado em 1 GPU): {effective_batch_size}")
    loader = torch.utils.data.DataLoader(dataset, batch_size=effective_batch_size, shuffle=True)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=2e-4)
    epochs = 5000
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1000, gamma=0.5)

    if resume_state is not None:
        optimizer.load_state_dict(resume_state["optimizer"])
        scheduler.load_state_dict(resume_state["scheduler"])
        print(f"[+] Momentos do Adam e scheduler restaurados (LR atual: "
              f"{scheduler.get_last_lr()[0]:.2e}).")

    model.train()
    os.makedirs("weights", exist_ok=True)

    def save_all(ep_done, numerado=False):
        """Salva os dois arquivos rolantes; só arquiva um checkpoint numerado quando
        pedido.

        O salvamento por tempo dispara a cada ~10 épocas, então numerar todo save
        encheria /kaggle/working com ~40 arquivos de 151 MB por sessão (≈6 GB) sem
        utilidade — os rolantes já bastam para retomar. Numerado só no múltiplo de 100.
        """
        state = model.module.state_dict() if isinstance(model, nn.DataParallel) else model.state_dict()
        torch.save(state, BACKEND_WEIGHTS)                       # formato lido pelo backend
        torch.save({"model": state,
                    "optimizer": optimizer.state_dict(),
                    "scheduler": scheduler.state_dict(),
                    "epoch": ep_done}, TRAIN_STATE)
        if numerado:
            torch.save(state, f"weights/pino_v3_phase5_physics_ep{ep_done}.pth")

    last_save = time.time()
    for ep in range(start_epoch, epochs):
        ep_l2 = 0.0
        ep_sob = 0.0
        ep_phys = 0.0
        ep_total = 0.0
        
        for V_batch, p_batch, phi_batch, dphi_batch in loader:
            V_batch = V_batch.to(device).requires_grad_(True)
            p_batch = p_batch.to(device)
            phi_batch = phi_batch.to(device)
            dphi_batch = dphi_batch.to(device)
            
            N = V_batch.shape[1]
            B = V_batch.shape[0]

            # unpack_params aceita 3, 5 ou 6 colunas (datasets antigos incluídos)
            b_v, u_v, c1_v, c4_v, a_v = unpack_params(p_batch, device=device)
            exp = lambda t: t.expand(B, N, N)

            # [V, c1, A, b, c4, u] — kappa está implícito em V (Yukawa)
            inputs = torch.zeros(B, N, N, 6, device=device)
            inputs[..., 0] = V_batch
            inputs[..., 1] = exp(c1_v)                              # Diblock
            inputs[..., 2] = exp(amplitude_to_channel(a_v))         # amplitude física
            inputs[..., 3] = exp(b_v)                               # Kuhn
            inputs[..., 4] = exp(c4_v)                              # Alternado
            inputs[..., 5] = exp(u_v)                               # Flory-Huggins
            
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
            print(f"Ep [{ep+1}/{epochs}] | Total: {ep_total/n_b:.4f} | L2: {ep_l2/n_b:.4f} | PDE: {ep_sob/n_b:.4f} | Phys: {ep_phys/n_b:.4f} | LR: {scheduler.get_last_lr()[0]:.2e}")
        
        marco = (ep + 1) % 100 == 0
        if marco or (time.time() - last_save) > SAVE_EVERY_MIN * 60:
            save_all(ep + 1, numerado=marco)
            last_save = time.time()
            print(f"[+] Checkpoint salvo na época {ep+1}!")

    save_all(epochs, numerado=True)
    print("[+] Treino Unificado Concluído! Pesos salvos.")

if __name__ == "__main__":
    train_physics()


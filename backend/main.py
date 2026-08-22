from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import torch
import numpy as np
import base64
import io
import os
import sys
import matplotlib.pyplot as plt

# ==========================================
# 1. Ajuste de Paths para importar a FNO Paramétrica
# ==========================================
current_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.dirname(current_dir)
# Fonte única de verdade: pino_architecture.py na raiz do projeto
sys.path.insert(0, root_dir)

try:
    from pino_architecture import PINO_Polyelectrolyte
except ImportError:
    print("Aviso: Não foi possível importar PINO_Polyelectrolyte. Verifique se o caminho está correto.")

# ==========================================
# 2. Inicialização do Servidor e Modelo
# ==========================================
app = FastAPI(title="PINO Quantum Sandbox API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], 
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Subindo servidor na placa: {device}")

# Parâmetros Universais
N = 100
L = 8.0
dx = L / (N - 1)
a = 1.0 # Raio da Esfera

# Grid Espacial
x = np.linspace(-L/2, L/2, N)
z = np.linspace(-L/2, L/2, N)
X, Z = np.meshgrid(x, z, indexing='ij')
R = np.sqrt(X**2 + Z**2)

# Carregando o Cérebro Neural Paramétrico
model = PINO_Polyelectrolyte(modes1=16, modes2=16, width=96, input_channels=8).to(device)

try:
    if os.path.exists(os.path.join(root_dir, "weights/pino_v3_phase5_physics.pth")):
        model.load_state_dict(torch.load(os.path.join(root_dir, "weights/pino_v3_phase5_physics.pth"), map_location=device))
        print("Modelo PINO V3 (Fase 5 - Física Collocation / Ginzburg-Landau) carregado com sucesso!")
    elif os.path.exists(os.path.join(root_dir, "weights/pino_v3_phase4_final.pth")):
        model.load_state_dict(torch.load(os.path.join(root_dir, "weights/pino_v3_phase4_final.pth"), map_location=device))
        print("Modelo PINO V3 (Fase 4 - Carga Contínua Final) carregado com sucesso!")
    elif os.path.exists(os.path.join(root_dir, "weights/pino_v3_phase3_final.pth")):
        model.load_state_dict(torch.load(os.path.join(root_dir, "weights/pino_v3_phase3_final.pth"), map_location=device))
        print("Modelo PINO V3 (Fase 3 - Eletrostática Final) carregado com sucesso!")
    elif os.path.exists(os.path.join(current_dir, "model_core/pino_v3_sobolev_epoch_75.pth")):
        model.load_state_dict(torch.load(os.path.join(current_dir, "model_core/pino_v3_sobolev_epoch_75.pth"), map_location=device))
        print("Modelo PINO V3 (Fase 2 - Sobolev) carregado com sucesso! (Aguardando Treinos)")
    model.eval()
except Exception as e:
    print(f"Erro ao carregar os pesos do modelo: {e}")
    model = None

# ==========================================
# 3. Modelos de Comunicação (Pydantic)
# ==========================================
class Charge(BaseModel):
    x: int
    z: int
    q: float
    r: float

class PredictionRequest(BaseModel):
    charges: list[Charge]
    b: float
    kappa: float
    u: float
    polymer_charge: int = 0
    polymer_charge_intensity: float = 1.0

class CompareRequest(BaseModel):
    stateA: PredictionRequest
    stateB: PredictionRequest

# ==========================================
# Funções Auxiliares
# ==========================================
def compute_density(charges: list[Charge], b_val: float, kappa_val: float, u_val: float, polymer_charge_type: int = 0, polymer_charge_intensity: float = 1.0):
    V = np.zeros((N, N))
    epsilon = 0.1
    
    # Calcular centro de massa das cargas no eixo Z
    sum_q = 0.0
    sum_qz = 0.0
    
    for c in charges:
        pos_x = x[c.x]
        pos_z = z[N - 1 - c.z] # Inverte Z pois o canvas HTML cresce para baixo e o plano cartesiano para cima
        # Usando a mesma formulação Yukawa (Debye-Huckel) do treinamento V1
        dist = np.sqrt((X - pos_x)**2 + (Z - pos_z)**2)
        V += c.q * np.exp(-kappa_val * dist) / (dist + epsilon)

    V_clean = np.copy(V)
    
    # Normalizar V para o range de treinamento [-1.0, 1.0] (Para a rede não entrar em colapso)
    non_wall = (R >= a)
    if np.max(V_clean[non_wall]) > np.min(V_clean[non_wall]):
        v_min = np.min(V_clean[non_wall])
        v_max = np.max(V_clean[non_wall])
        # Escala primeiro para [0, 1] e depois para [-1, 1]
        V_clean[non_wall] = 2.0 * ((V_clean[non_wall] - v_min) / (v_max - v_min)) - 1.0
    elif np.max(V_clean[non_wall]) == 0:
        pass # Sem cargas, mantém 0 (o que está ok no range [-1, 1])
    else:
        # Se for constante mas diferente de 0, capa em -1 ou 1
        V_clean[non_wall] = np.clip(V_clean[non_wall], -1.0, 1.0)
        
    V_clean[R < a] = 10.0 # Paredes continuam em 10.0 (regra do SCFT solver)
    
    v_tensor = torch.tensor(V_clean, dtype=torch.float32)
    x_tensor = torch.tensor(X, dtype=torch.float32)
    z_tensor = torch.tensor(Z, dtype=torch.float32)
    
    b_plane = torch.full((N, N), b_val, dtype=torch.float32)
    kappa_plane = torch.full((N, N), kappa_val, dtype=torch.float32)
    u_plane = torch.full((N, N), u_val, dtype=torch.float32)
    
    c1_plane = torch.zeros((N, N), dtype=torch.float32)
    c4_plane = torch.zeros((N, N), dtype=torch.float32)
    
    if polymer_charge_type == 1:
        c1_plane = torch.full((N, N), polymer_charge_intensity, dtype=torch.float32)
    elif polymer_charge_type == 4:
        c4_plane = torch.full((N, N), polymer_charge_intensity, dtype=torch.float32)
    
    if model is not None:
        if len(charges) == 0:
            density = np.zeros((N, N))
            d1 = np.zeros((N, N))
            d2 = np.zeros((N, N))
        else:
            v_tensor.requires_grad_(True)
            # Reconstruct inputs so gradients flow through v_tensor
            v_in = v_tensor.unsqueeze(0).unsqueeze(-1)
            x_in = x_tensor.unsqueeze(0).unsqueeze(-1)
            z_in = z_tensor.unsqueeze(0).unsqueeze(-1)
            b_in = b_plane.unsqueeze(0).unsqueeze(-1)
            k_in = kappa_plane.unsqueeze(0).unsqueeze(-1)
            u_in = u_plane.unsqueeze(0).unsqueeze(-1)
            
            c1_in = c1_plane.unsqueeze(0).unsqueeze(-1)
            c4_in = c4_plane.unsqueeze(0).unsqueeze(-1)
            
            # Nova arquitetura: [V, c1, c2(zero), b, c4, u]
            c2_in = torch.zeros_like(v_in)
            
            inputs = torch.cat([v_in, c1_in, c2_in, b_in, c4_in, u_in], dim=-1).to(device)
            
            with torch.set_grad_enabled(True):
                out = model(inputs)
                phi = out[0]  # output shape (B, Nx, Ny) após squeeze(-1)
                
                # Objetivo S = int(phi^2) - mede localizacao do polimero
                S = torch.sum(phi**2)
                
                dphi_dV = torch.autograd.grad(S, v_tensor, create_graph=True)[0]
                
                grad_sum = torch.sum(dphi_dV)
                d2phi_dV2 = torch.autograd.grad(grad_sum, v_tensor)[0]
            
            density = phi.detach().cpu().numpy()
            # --- FILTRO FÍSICO NEWTON-RAPHSON PÓS-PROCESSAMENTO ---
            # Reconstrói V_eff em numpy para o passo da Física
            V_fft = np.fft.fft2(V_clean)
            kx_np = np.fft.fftfreq(N)
            kz_np = np.fft.fftfreq(N)
            KX, KZ = np.meshgrid(kx_np, kz_np, indexing='ij')
            
            dV_dx = np.real(np.fft.ifft2(1j * 2 * np.pi * KX * V_fft))
            dV_dz = np.real(np.fft.ifft2(1j * 2 * np.pi * KZ * V_fft))
            grad_V_sq = dV_dx**2 + dV_dz**2
            
            c1_val = polymer_charge_intensity if polymer_charge_type == 1 else 0.0
            c4_val = polymer_charge_intensity if polymer_charge_type == 4 else 0.0
            V_eff = V_clean - (c1_val * 2.0 * np.abs(V_clean)) - (c4_val * 0.1 * grad_V_sq)
            
            # Calcula Laplaciano da predição da rede
            phi_fft = np.fft.fft2(density)
            K_sq = KX**2 + KZ**2
            laplacian_phi = np.real(np.fft.ifft2(-(4.0 * np.pi**2 * K_sq) * phi_fft))
            
            # F(phi) = (b²/6)·Laplacian(phi) - (V_eff + u·phi)·phi  [Edwards ground state]
            R_pde = (b_val**2 / 6.0) * laplacian_phi - (V_eff + u_val * density) * density
            
            # dF/dphi = - V_eff - 2 * u * phi (Jacobiano local aproximado)
            dF_dphi = - V_eff - 2.0 * u_val * density
            # Previne divisão por zero
            dF_dphi[np.abs(dF_dphi) < 1e-6] = 1e-6 * np.sign(dF_dphi[np.abs(dF_dphi) < 1e-6] + 1e-12)
            
            # Newton Step com relaxação
            delta_phi = - R_pde / dF_dphi
            density = density + 0.1 * delta_phi
            density = np.clip(density, 0, None)
            
            # --- CONSERVAÇÃO CANÔNICA ---
            # Desloca o potencial químico do sistema para satisfazer a massa
            dx_local = 1.0 / 100.0
            current_mass = np.sum(density) * dx_local * dx_local
            if current_mass > 1e-6:
                density = density * (1.0 / current_mass)
            
            d1 = dphi_dV.detach().cpu().numpy()
            d2 = d2phi_dV2.detach().cpu().numpy()
    else:
        density = np.zeros((N, N))
        d1 = np.zeros((N, N))
        d2 = np.zeros((N, N))
    
    # Impedir densidades negativas (artefato da camada linear da IA)
    density = np.clip(density, 0, None)
    density[R < a] = np.nan
    d1[R < a] = np.nan
    d2[R < a] = np.nan
    
    return density, d1, d2

def create_plot_base64(X_grid, Z_grid, matrix, cmap, label, vmin=None, vmax=None):
    fig, ax = plt.subplots(figsize=(5, 4), dpi=100)
    fig.patch.set_facecolor('black')
    ax.set_facecolor('black')
    
    cax = ax.contourf(X_grid, Z_grid, matrix, levels=50, cmap=cmap, vmin=vmin, vmax=vmax)
    cbar = fig.colorbar(cax, ax=ax, fraction=0.046, pad=0.04)
    cbar.ax.tick_params(colors='white')
    cbar.outline.set_edgecolor('white')
    cbar.set_label(label, color='white', labelpad=10)
    
    circle = plt.Circle((0, 0), a, color='#1e293b', zorder=10)
    ax.add_artist(circle)
    
    ax.set_aspect('equal')
    ax.set_xlim(-L/2, L/2)
    ax.set_ylim(-L/2, L/2)
    
    ax.grid(color='white', alpha=0.3, linestyle='--')
    ax.tick_params(colors='white')
    for spine in ax.spines.values():
        spine.set_color('white')
        spine.set_alpha(0.5)
    
    buf = io.BytesIO()
    plt.savefig(buf, format='png', facecolor='black', bbox_inches='tight', dpi=100)
    plt.close(fig)
    
    buf.seek(0)
    return base64.b64encode(buf.read()).decode('utf-8')

# ==========================================
# 4. Rota de Inferência
# ==========================================
@app.post("/predict")
async def predict_density(request: PredictionRequest):
    density, d1, d2 = compute_density(request.charges, request.b, request.kappa, request.u, request.polymer_charge, request.polymer_charge_intensity)
    
    # 1. Gerar Imagens
    img_b64 = create_plot_base64(X, Z, density, 'inferno', 'Densidade Polimérica')
    
    # Normalizar escala de cores para as derivadas (simetria em 0)
    max_d1 = np.nanmax(np.abs(d1)) if not np.all(np.isnan(d1)) else 1.0
    if max_d1 == 0: max_d1 = 1.0
    img_d1 = create_plot_base64(X, Z, d1, 'RdBu_r', 'dφ/dV (Linear)', vmin=-max_d1, vmax=max_d1)
    
    max_d2 = np.nanmax(np.abs(d2)) if not np.all(np.isnan(d2)) else 1.0
    if max_d2 == 0: max_d2 = 1.0
    img_d2 = create_plot_base64(X, Z, d2, 'PiYG', 'd²φ/dV² (Não-Linear)', vmin=-max_d2, vmax=max_d2)
    
    # 2. Cálculos Quantitativos (Massa, Centro de Massa, Raio de Giração)
    valid_mask = ~np.isnan(density)
    valid_density = density[valid_mask]
    
    # Garantir que não há densidades negativas (artefato numérico)
    valid_density = np.clip(valid_density, 0, None)
    mass = float(np.sum(valid_density) * dx * dx)
    
    if mass > 1e-6:
        com_x = float(np.sum(X[valid_mask] * valid_density) * dx * dx / mass)
        com_z = float(np.sum(Z[valid_mask] * valid_density) * dx * dx / mass)
        rg_sq = np.sum(((X[valid_mask] - com_x)**2 + (Z[valid_mask] - com_z)**2) * valid_density) * dx * dx / mass
        rg = float(np.sqrt(rg_sq))
    else:
        com_x, com_z, rg = 0.0, 0.0, 0.0
        
    # 3. Heurística Termodinâmica de Fase (Termômetro)
    phase = "Estável (Intermediário)"
    if request.u > 0.2 and rg < 2.0:
        phase = "Colapsado (Globule)"
    elif request.u < -0.2:
        phase = "Inchado (Coil)"
        
    return {
        "image": img_b64,
        "image_d1": img_d1,
        "image_d2": img_d2,
        "metrics": {
            "mass": mass,
            "com_x": com_x,
            "com_z": com_z,
            "rg": rg,
            "phase": phase
        }
    }

# ==========================================
# 5. Rota de Comparação
# ==========================================
@app.post("/compare")
async def compare_states(request: CompareRequest):
    densityA, _, _ = compute_density(request.stateA.charges, request.stateA.b, request.stateA.kappa, request.stateA.u, request.stateA.polymer_charge, request.stateA.polymer_charge_intensity)
    densityB, _, _ = compute_density(request.stateB.charges, request.stateB.b, request.stateB.kappa, request.stateB.u, request.stateB.polymer_charge, request.stateB.polymer_charge_intensity)
    
    diff = densityB - densityA
    max_val = np.nanmax(np.abs(diff))
    if np.isnan(max_val) or max_val == 0:
        max_val = 1.0 # fallback
        
    img_b64 = create_plot_base64(X, Z, diff, 'RdBu_r', 'Diferença de Densidade (B - A)', vmin=-max_val, vmax=max_val)
    return {"image": img_b64}

class ExperimentRequest(BaseModel):
    charges: list[Charge]
    b: float
    kappa: float
    u: float
    polymer_charge: int = 0
    polymer_charge_intensity: float = 0.0
    sweep_type: str

# ==========================================
# 6. Rota dos Experimentos (Coil-Globule, Salinidade, Rigidez)
# ==========================================
@app.post("/experiment")
async def run_experiment(request: ExperimentRequest):
    pts = 40
    if request.sweep_type == "u":
        values = np.linspace(-1.0, 1.0, pts)
        xlabel = "Interação Efetiva (u)"
        title = "Diagrama de Fases (Transição Coil-Globule)"
        color = "#38bdf8"
    elif request.sweep_type == "kappa":
        values = np.linspace(0.1, 5.0, pts)
        xlabel = "Comprimento de Debye Inverso (kappa) - Salinidade"
        title = "Efeito de Salinidade na Estrutura Polimérica"
        color = "#fca5a5"
    elif request.sweep_type == "kappa_adsorption":
        values = np.linspace(0.1, 5.0, pts)
        xlabel = "Comprimento de Debye Inverso (kappa) - Salinidade"
        title = "Transição Adsorvido/Desorvido vs Salinidade"
        color = "#10b981"
    elif request.sweep_type == "b":
        values = np.linspace(0.5, 2.0, pts)
        xlabel = "Comprimento de Kuhn (b) - Rigidez"
        title = "Espectroscopia de Rigidez Polimérica"
        color = "#c084fc"
    else:
        values = np.linspace(-1.0, 1.0, pts)
        xlabel = "Interação Efetiva (u)"
        title = "Diagrama de Fases"
        color = "#38bdf8"
        
    metric_values = []
    
    for val in values:
        if request.sweep_type == "u":
            density, _, _ = compute_density(request.charges, request.b, request.kappa, float(val), request.polymer_charge, request.polymer_charge_intensity)
        elif request.sweep_type in ["kappa", "kappa_adsorption"]:
            density, _, _ = compute_density(request.charges, request.b, float(val), request.u, request.polymer_charge, request.polymer_charge_intensity)
        elif request.sweep_type == "b":
            density, _, _ = compute_density(request.charges, float(val), request.kappa, request.u, request.polymer_charge, request.polymer_charge_intensity)
            
        valid_mask = ~np.isnan(density)
        valid_density = density[valid_mask]
        valid_density = np.clip(valid_density, 0, None)
        
        mass = float(np.sum(valid_density) * dx * dx)
        if mass > 1e-6:
            com_x = float(np.sum(X[valid_mask] * valid_density) * dx * dx / mass)
            com_z = float(np.sum(Z[valid_mask] * valid_density) * dx * dx / mass)
            rg_sq = np.sum(((X[valid_mask] - com_x)**2 + (Z[valid_mask] - com_z)**2) * valid_density) * dx * dx / mass
            rg = float(np.sqrt(rg_sq))
        else:
            rg = 0.0
            
        metric_values.append(rg)
        
    fig, ax = plt.subplots(figsize=(6.5, 4), dpi=100)
    fig.patch.set_facecolor('black')
    ax.set_facecolor('black')
    
    ax.plot(values, metric_values, marker='o', color=color, linewidth=2, markersize=4)
    
    # Análise de Pontos Críticos (Derivada Numérica)
    metric_array = np.array(metric_values)
    val_array = np.array(values)
    
    # Suavização simples para evitar ruído e cálculo do gradiente
    if len(val_array) > 5 and np.max(metric_array) > 0.1:
        dmetric = np.gradient(metric_array, val_array)
        idx_crit = np.argmax(np.abs(dmetric))
        x_c = val_array[idx_crit]
        y_c = metric_array[idx_crit]
        
        # Linha vertical
        ax.axvline(x_c, color='#fbbf24', linestyle='--', alpha=0.8, linewidth=1.5)
        
        if request.sweep_type == "u":
            text_str = f"Ponto Theta ($\\theta$): u = {x_c:.2f}\n[Inflexão Solvatação]"
        elif request.sweep_type == "kappa":
            text_str = f"Transição Estrutural: $\\kappa$ = {x_c:.2f}\n[Limite Eletrostático]"
        elif request.sweep_type == "kappa_adsorption":
            text_str = f"Ponto de Desorção: $\\kappa$ = {x_c:.2f}\n[Fuga do Polímero]"
        elif request.sweep_type == "b":
            text_str = f"Fronteira Flex/Ríg: b = {x_c:.2f}\n[Limite de Entropia]"
        else:
            text_str = f"Ponto Crítico: {x_c:.2f}"
            
        # Posição do texto para não bater na linha (se x_c está muito à direita, coloca na esquerda e vice-versa)
        ha = 'left' if idx_crit < len(val_array)/2 else 'right'
        offset = 0.05 * (np.max(val_array) - np.min(val_array))
        text_x = x_c + offset if ha == 'left' else x_c - offset
        
        ax.text(text_x, np.mean(metric_array), text_str, color='#fbbf24', fontsize=9, ha=ha, va='center',
                bbox=dict(facecolor='black', alpha=0.7, edgecolor='none'))
    
    ax.set_title(title, color='white')
    ax.set_xlabel(xlabel, color='white')
    ax.set_ylabel("Raio de Giração (Rg)", color='white')
    
    ax.grid(color='white', alpha=0.2, linestyle='--')
    ax.tick_params(colors='white')
    for spine in ax.spines.values():
        spine.set_color('white')
        spine.set_alpha(0.5)
        
    buf = io.BytesIO()
    plt.savefig(buf, format='png', facecolor='black', bbox_inches='tight', dpi=100)
    plt.close(fig)
    buf.seek(0)
    
    img_b64 = base64.b64encode(buf.read()).decode('utf-8')
    return {"image": img_b64}

# ==========================================
# 7. Rota do Histórico de Loss
# ==========================================
@app.get("/loss")
async def get_loss_history():
    loss_path = os.path.join(current_dir, "model_core/parametric_loss_history.pt")
    if not os.path.exists(loss_path):
        raise HTTPException(status_code=404, detail="Loss history not found yet")
        
    try:
        history = torch.load(loss_path)
        train_loss = history['train']
        test_loss = history['test']
        
        fig, ax = plt.subplots(figsize=(4, 4), dpi=100)
        fig.patch.set_facecolor('#1e1030')
        ax.set_facecolor('#1e1030')
        
        ax.plot(train_loss, label='Train Loss', color='#818cf8', linewidth=2)
        ax.plot(test_loss, label='Test Loss', color='#fca5a5', linewidth=2)
        ax.set_yscale('log')
        ax.set_xlabel('Epochs', color='white')
        ax.set_ylabel('Loss (MSE)', color='white')
        
        ax.legend(facecolor='black', edgecolor='white', labelcolor='white')
        ax.grid(color='white', alpha=0.1, linestyle='--')
        ax.tick_params(colors='white')
        for spine in ax.spines.values():
            spine.set_color('white')
            spine.set_alpha(0.3)
            
        buf = io.BytesIO()
        plt.savefig(buf, format='png', facecolor='#1e1030', bbox_inches='tight', dpi=100)
        plt.close(fig)
        
        buf.seek(0)
        img_b64 = base64.b64encode(buf.read()).decode('utf-8')
        return {"image": img_b64}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# ==========================================
# 8. Rota de Varredura de Multipolos
# ==========================================
class MultipoleExperimentRequest(BaseModel):
    q_magnitude: float
    b: float
    kappa: float
    u: float
    sweep_type: str

@app.post("/experiment_multipole")
async def run_multipole_experiment(request: MultipoleExperimentRequest):
    if request.sweep_type == "u":
        values = np.linspace(-1.0, 1.0, 30)
        xlabel = "Interação Solvente (u)"
        title = f"Ponto Crítico vs Momento Multipolar (u_c)\nFixos: b={request.b:.2f}, \u03ba={request.kappa:.2f}"
    elif request.sweep_type == "kappa":
        values = np.linspace(0.1, 5.0, 30)
        xlabel = "Screening Eletrostático (\u03ba)"
        title = f"Ponto Crítico vs Momento Multipolar (\u03bac)\nFixos: u={request.u:.2f}, b={request.b:.2f}"
    else: 
        values = np.linspace(0.5, 2.0, 30)
        xlabel = "Rigidez Kuhn (b)"
        title = f"Ponto Crítico vs Momento Multipolar (b_c)\nFixos: u={request.u:.2f}, \u03ba={request.kappa:.2f}"

    poles_list = [2, 4, 6, 8, 10]
    critical_points = []
    
    for n_poles in poles_list:
        charges = []
        radius_grid = a * (N / L) # 12.5
        center = N / 2
        for i in range(n_poles):
            angle = i * (2 * np.pi / n_poles)
            cx = int(center + radius_grid * np.cos(angle))
            cz = int(center + radius_grid * np.sin(angle))
            q_val = request.q_magnitude if i % 2 == 0 else -request.q_magnitude
            charges.append(Charge(x=cx, z=cz, q=q_val, r=2.0))
        
        rg_values = []
        for val in values:
            b_val = val if request.sweep_type == "b" else request.b
            kappa_val = val if request.sweep_type == "kappa" else request.kappa
            u_val = val if request.sweep_type == "u" else request.u
            
            density, _, _ = compute_density(charges, b_val, kappa_val, u_val)
            
            valid_mask = ~np.isnan(density)
            valid_density = density[valid_mask]
            valid_density = np.clip(valid_density, 0, None)
            mass = float(np.sum(valid_density) * dx * dx)
            if mass > 1e-6:
                com_x = float(np.sum(X[valid_mask] * valid_density) * dx * dx / mass)
                com_z = float(np.sum(Z[valid_mask] * valid_density) * dx * dx / mass)
                rg_sq = np.sum(((X[valid_mask] - com_x)**2 + (Z[valid_mask] - com_z)**2) * valid_density) * dx * dx / mass
                rg = float(np.sqrt(rg_sq))
            else:
                rg = 0.0
            rg_values.append(rg)
            
        rg_array = np.array(rg_values)
        val_array = np.array(values)
        if len(val_array) > 5 and np.max(rg_array) > 0.1:
            drg = np.gradient(rg_array, val_array)
            idx_crit = np.argmax(np.abs(drg))
            critical_points.append(val_array[idx_crit])
        else:
            critical_points.append(np.nan)
            
    fig, ax = plt.subplots(figsize=(6.5, 4), dpi=100)
    fig.patch.set_facecolor('black')
    ax.set_facecolor('black')
    
    ax.plot(poles_list, critical_points, marker='s', color='#f43f5e', linewidth=2.5, markersize=8)
    
    ax.set_title(title, color='white')
    ax.set_xlabel("Número de Polos na Macromolécula (n)", color='white')
    ax.set_ylabel(f"Ponto Crítico ({request.sweep_type}_c)", color='white')
    ax.set_xticks(poles_list)
    
    ax.grid(color='white', alpha=0.2, linestyle='--')
    ax.tick_params(colors='white')
    for spine in ax.spines.values():
        spine.set_color('white')
        spine.set_alpha(0.5)
        
    buf = io.BytesIO()
    plt.savefig(buf, format='png', facecolor='black', bbox_inches='tight', dpi=100)
    plt.close(fig)
    buf.seek(0)
    
    img_b64 = base64.b64encode(buf.read()).decode('utf-8')
    return {"image": img_b64}

class BatchScreenRequest(BaseModel):
    items: list[PredictionRequest]

@app.post("/batch_screen")
async def batch_screen(request: BatchScreenRequest):
    results = []
    for item in request.items:
        density, _, _ = compute_density(item.charges, item.b, item.kappa, item.u, item.polymer_charge, item.polymer_charge_intensity)
        valid_mask = ~np.isnan(density)
        valid_density = np.clip(density[valid_mask], 0, None)
        mass = float(np.sum(valid_density) * dx * dx)
        
        if mass > 1e-6:
            com_x = float(np.sum(X[valid_mask] * valid_density) * dx * dx / mass)
            com_z = float(np.sum(Z[valid_mask] * valid_density) * dx * dx / mass)
            rg_sq = np.sum(((X[valid_mask] - com_x)**2 + (Z[valid_mask] - com_z)**2) * valid_density) * dx * dx / mass
            rg = float(np.sqrt(rg_sq))
        else:
            com_x, com_z, rg = 0.0, 0.0, 0.0
            
        phase = "Estável"
        if item.u > 0.2 and rg < 2.0:
            phase = "Colapsado"
        elif item.u < -0.2:
            phase = "Inchado"
            
        results.append({
            "b": item.b, "kappa": item.kappa, "u": item.u, 
            "polymer_charge": item.polymer_charge, 
            "intensity": item.polymer_charge_intensity,
            "mass": mass, "com_x": com_x, "com_z": com_z, "rg": rg, "phase": phase
        })
    return {"results": results}

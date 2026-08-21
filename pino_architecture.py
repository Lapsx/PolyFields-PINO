"""
PolyFields-PINO v3 — Arquitetura e Funções de Loss
====================================================
Physics-Informed Neural Operator para densidade de polieletrólitos em
confinamento eletrostático assimétrico (potenciais multipolares arbitrários).

Backbone : Fourier Neural Operator 2D (FNO), 4 camadas espectrais.
Física   : Equação de Edwards (Ground State Dominance) + Perda Sobolev H¹.

Canais de entrada (8 = 6 explícitos + 2 coordenadas de grade):
  [0] V(r)   — Potencial externo normalizado ∈ [-1, 1]; paredes: V = 10.0
  [1] c1(r)  — Intensidade de carga Diblock (escalar uniforme por amostra)
  [2] c2(r)  — Reservado / zero  (kappa está implícito em V via Yukawa)
  [3] b(r)   — Comprimento de Kuhn (escalar uniforme por amostra)
  [4] c4(r)  — Intensidade de carga Alternado (escalar uniforme por amostra)
  [5] u(r)   — Parâmetro de Flory-Huggins (escalar uniforme por amostra)
  [6] x      — Coordenada x normalizada ∈ [0, 1]  (grid embedding)
  [7] z      — Coordenada z normalizada ∈ [0, 1]  (grid embedding)

Saída: φ(r) ∈ ℝ⁺ — densidade polimérica via softplus, shape (B, Nx, Ny).

Equação de Edwards (Ground State Dominance):
  (b²/6) ∇²φ(r) = [V_eff(r) + u·φ(r)] · φ(r)
  V_eff = V − c1·2|V| − c4·0.1·|∇V|²

Nota sobre dphi_dV_true: mantido na assinatura da loss por compatibilidade com
o DataLoader. A susceptibilidade térmica δφ/δV é capturada implicitamente
pela perda Sobolev H¹: ||∇(φ_pred − φ_true)||²_L2  via teorema de Parseval.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ===========================================================================
# 1. BLOCO ESPECTRAL — núcleo do FNO
# ===========================================================================

class SpectralConv2d(nn.Module):
    """
    Convolução espectral 2D: opera nos modos de baixa frequência da rfft2.

    Pesos armazenados como float32 de forma (..., 2) em vez de cfloat para
    compatibilidade com nn.DataParallel no Kaggle (bug de serialização de
    tensores complexos). Convertidos para complex via view_as_complex no forward.
    """
    def __init__(self, in_channels, out_channels, modes1, modes2):
        super().__init__()
        self.in_channels  = in_channels
        self.out_channels = out_channels
        self.modes1 = modes1
        self.modes2 = modes2
        scale = 1.0 / (in_channels * out_channels)
        self.weights1 = nn.Parameter(
            scale * torch.rand(in_channels, out_channels, modes1, modes2, 2))
        self.weights2 = nn.Parameter(
            scale * torch.rand(in_channels, out_channels, modes1, modes2, 2))

    def forward(self, x):
        B, C, H, W = x.shape
        x_ft   = torch.fft.rfft2(x)                                  # (B, C, H, W//2+1)
        out_ft = torch.zeros(B, self.out_channels, H, W // 2 + 1,
                             dtype=torch.cfloat, device=x.device)
        w1 = torch.view_as_complex(self.weights1)
        w2 = torch.view_as_complex(self.weights2)
        # Modos positivos e negativos (quadrantes de baixa frequência)
        out_ft[:, :, :self.modes1, :self.modes2] = torch.einsum(
            "bixy,ioxy->boxy", x_ft[:, :, :self.modes1, :self.modes2], w1)
        out_ft[:, :, -self.modes1:, :self.modes2] = torch.einsum(
            "bixy,ioxy->boxy", x_ft[:, :, -self.modes1:, :self.modes2], w2)
        return torch.fft.irfft2(out_ft, s=(H, W))                    # (B, C, H, W)


# ===========================================================================
# 2. ARQUITETURA PINO — FNO 2D com grid embedding
# ===========================================================================

class PINO_Polyelectrolyte(nn.Module):
    """
    FNO 2D para predição de φ(r).

    Recebe 6 canais físicos de entrada; as coordenadas de grade (x, z) são
    concatenadas internamente, totalizando input_channels = 8 canais no lifting.

    Ativação de saída: softplus  (φ ≥ 0, gradientes sempre definidos).
    Descartado: softmax espacial — mata gradientes em ~1/N² = 1/10.000.
    """
    def __init__(self, modes1=16, modes2=16, width=96, input_channels=8):
        super().__init__()
        self.modes1 = modes1
        self.modes2 = modes2
        self.width  = width

        # Lifting: projeta os 8 canais para o espaço latente
        self.p = nn.Linear(input_channels, width)

        # 4 camadas FNO: espectral + 1×1 local em paralelo
        self.conv0 = SpectralConv2d(width, width, modes1, modes2)
        self.conv1 = SpectralConv2d(width, width, modes1, modes2)
        self.conv2 = SpectralConv2d(width, width, modes1, modes2)
        self.conv3 = SpectralConv2d(width, width, modes1, modes2)
        self.w0    = nn.Conv2d(width, width, 1)
        self.w1    = nn.Conv2d(width, width, 1)
        self.w2    = nn.Conv2d(width, width, 1)
        self.w3    = nn.Conv2d(width, width, 1)

        # Projection: espaço latente → φ(r) escalar
        self.q  = nn.Linear(width, 64)
        self.q2 = nn.Linear(64, 1)

    def forward(self, x):
        # x: (B, Nx, Ny, 6)
        grid = self.get_grid(x.shape, x.device)
        x = torch.cat((x, grid), dim=-1)   # → (B, Nx, Ny, 8)

        x = self.p(x)                       # → (B, Nx, Ny, width)
        x = x.permute(0, 3, 1, 2)          # → (B, width, Nx, Ny)

        x = F.gelu(self.conv0(x) + self.w0(x))
        x = F.gelu(self.conv1(x) + self.w1(x))
        x = F.gelu(self.conv2(x) + self.w2(x))
        x = self.conv3(x) + self.w3(x)     # sem GELU na última camada FNO

        x = x.permute(0, 2, 3, 1)          # → (B, Nx, Ny, width)
        x = F.gelu(self.q(x))
        x = self.q2(x)                     # → (B, Nx, Ny, 1)

        return F.softplus(x).squeeze(-1)   # → (B, Nx, Ny), φ ∈ ℝ⁺

    def get_grid(self, shape, device):
        B, Nx, Ny = shape[0], shape[1], shape[2]
        gx = torch.linspace(0, 1, Nx, device=device).view(1, Nx, 1, 1).repeat(B, 1, Ny, 1)
        gz = torch.linspace(0, 1, Ny, device=device).view(1, 1, Ny, 1).repeat(B, Nx, 1, 1)
        return torch.cat((gx, gz), dim=-1)  # (B, Nx, Ny, 2)


# ===========================================================================
# 3. HELPERS FÍSICOS INTERNOS
# ===========================================================================

def _spectral_ops(N, device):
    """
    Frequências e operador Laplaciano para uma grade N×N.
    Retorna tensores broadcast-compatíveis com (B, N, N).

      kx        : (1, N, 1) — frequências na direção x  [via fftfreq, ∈ (-0.5, 0.5)]
      kz        : (1, 1, N) — frequências na direção z
      k_sq_4pi2 : (1, N, N) — 4π²(kx² + kz²)  [para ∇² = -4π²k² no espaço de Fourier]
    """
    kx = torch.fft.fftfreq(N, device=device).view(1, N, 1)
    kz = torch.fft.fftfreq(N, device=device).view(1, 1, N)
    k_sq_4pi2 = 4.0 * torch.pi**2 * (kx**2 + kz**2)
    return kx, kz, k_sq_4pi2


def _compute_v_eff(v, c1_val, c4_val, kx, kz):
    """
    Potencial efetivo com perturbações de carga interna:
        V_eff = V − c1·2|V| − c4·0.1·|∇V|²

    O gradiente de V é calculado espectralmente:
        F{∂V/∂x}(k) = i·2π·kx · F{V}(k)
    """
    V_fft = torch.fft.fftn(v, dim=(1, 2))
    dV_dx = torch.fft.ifftn(1j * 2.0 * torch.pi * kx * V_fft, dim=(1, 2)).real
    dV_dz = torch.fft.ifftn(1j * 2.0 * torch.pi * kz * V_fft, dim=(1, 2)).real
    grad_V_sq = dV_dx**2 + dV_dz**2
    return v - c1_val * 2.0 * torch.abs(v) - c4_val * 0.1 * grad_V_sq


def _edwards_residual(phi, v_eff, u_val, b_val, k_sq_4pi2):
    """
    Resíduo da equação de Edwards (Ground State Dominance):
        R(φ) = (b²/6) ∇²φ − (V_eff + u·φ) · φ

    Laplaciano via Parseval: F{∇²φ}(k) = −4π²k² · F{φ}(k)
    """
    phi_fft = torch.fft.fftn(phi, dim=(1, 2))
    lap_phi = torch.fft.ifftn(-k_sq_4pi2 * phi_fft, dim=(1, 2)).real
    return (b_val**2 / 6.0) * lap_phi - (v_eff + u_val * phi) * phi


# ===========================================================================
# 4. PERDA SUPERVISIONADA — dados com ground truth SCFT
# ===========================================================================

def sobolev_physics_loss(phi_pred, phi_true, v_tensor, dphi_dV_true,
                          params, model, epoch=0):
    """
    Perda PINO composta para dados com Ground Truth SCFT (Fases 1–5).

    Componentes:
      1. L2 supervisionado (MSE em φ)
      2. Sobolev H¹: ||∇(φ_pred − φ_true)||²_L2 via Parseval
         Penaliza erros nos gradientes espaciais de φ (captura δφ/δV implicitamente).
      3. PDE de Edwards: (b²/6)∇²φ = (V_eff + u·φ)·φ — mascarada nas paredes
         Penaliza violação das leis de mecânica estatística.
      4. Conservação de massa (suave): (∑φ_pred − 100)² / N²
         Reforça ∫φ dr = 100 durante o treino (complementa o filtro Newton-Raphson
         do backend e o dataset do SCFT que já normaliza a massa).

    Schedule (adequado para iniciar a partir dos pesos pré-treinados da Fase 4):
      Épocas    0–200 : warm-up  — só L2 + Sobolev  (alpha_phys = 0)
      Épocas  200–1000: ramp-up  — física entra linearmente (alpha_phys: 0 → 1)
      Épocas  1000+   : treino completo              (alpha_phys = 1)

    dphi_dV_true: campo proxy (≈ −2φ) — mantido na assinatura por compatibilidade
                  com o DataLoader existente. NÃO é usado na loss.

    Returns:
        (loss_total, l2_item, sobolev_item, physics_item)
    """
    scale = 10_000.0   # N² = 100² — tira o otimizador da morte sub-numérica
    N   = phi_pred.shape[1]
    dev = phi_pred.device

    kx, kz, k_sq_4pi2 = _spectral_ops(N, dev)

    # ── 1. L2 supervisionado ─────────────────────────────────────────────
    l2_loss = F.mse_loss(phi_pred, phi_true) * scale

    # ── 2. Sobolev H¹ — ||∇(φ_pred − φ_true)||² via Parseval ────────────
    err_fft      = torch.fft.fftn(phi_pred - phi_true, dim=(1, 2))
    sobolev_loss = torch.mean(k_sq_4pi2 * torch.abs(err_fft)**2) * scale

    # ── 3. PDE de Edwards mascarada nas paredes ──────────────────────────
    b_val  = params[:, 0].view(-1, 1, 1)
    u_val  = params[:, 2].view(-1, 1, 1)
    c1_val = params[:, 3].view(-1, 1, 1)
    c4_val = params[:, 4].view(-1, 1, 1)

    v_eff   = _compute_v_eff(v_tensor, c1_val, c4_val, kx, kz)
    pde_res = _edwards_residual(phi_pred, v_eff, u_val, b_val, k_sq_4pi2)

    # Condições de contorno de Dirichlet: φ = 0 nas paredes (V ≥ 9.0).
    # A PDE não é válida nesses pixels — mascarar evita resíduos espúrios.
    domain = (v_tensor < 9.0).float()
    pde_res = pde_res * domain

    # ── 4. Conservação de massa (suave) ─────────────────────────────────
    mass_pred = phi_pred.sum(dim=(1, 2))                              # (B,)
    mass_loss = (F.mse_loss(mass_pred, torch.full_like(mass_pred, 100.0))
                 / (N ** 2) * scale)

    physics_loss = torch.mean(pde_res ** 2) * scale + 0.1 * mass_loss

    # ── Schedule ────────────────────────────────────────────────────────
    alpha_phys = min(1.0, max(0.0, (epoch - 200) / 800.0))
    loss_total = 1.0 * l2_loss + 0.1 * sobolev_loss + alpha_phys * physics_loss

    return loss_total, l2_loss.item(), sobolev_loss.item(), physics_loss.item()


# ===========================================================================
# 5. PERDA DE COLOCAÇÃO — Phase 10 (sem ground truth SCFT)
# ===========================================================================

def prepare_collocation_inputs(V_raw, params_col, device):
    """
    Pré-processa o dataset de colocação (data/pino_v3_collocation.pt) para o
    formato de entrada do modelo, aplicando a mesma normalização do pipeline GRF/SCFT.

    Args:
        V_raw     : (B, N, N) tensor — potenciais brutos (não normalizados)
        params_col: (B, 5)    tensor — [b, kappa, u, c1, c4]
        device    : torch.device

    Returns:
        inputs : (B, N, N, 6) — tensor pronto para PINO_Polyelectrolyte.forward()
        V_norm : (B, N, N)    — potencial normalizado usado na perda de colocação

    Exemplo de uso no loop de treino (Phase 10):
        inputs_col, v_col = prepare_collocation_inputs(V_raw_batch, p_col_batch, device)
        phi_col  = model(inputs_col)
        loss_col = collocation_pde_loss(phi_col, v_col, p_col_batch.to(device))
        loss = loss_labeled + 0.1 * loss_col
    """
    B, N = V_raw.shape[0], V_raw.shape[1]

    # Normaliza V para [-1, 1] por amostra; paredes (V > 9.0) fixas em 10.0
    wall_mask = (V_raw > 9.0)
    v_min = V_raw.masked_fill(wall_mask,  float('inf')).view(B, -1).min(dim=1)[0].view(B, 1, 1)
    v_max = V_raw.masked_fill(wall_mask, float('-inf')).view(B, -1).max(dim=1)[0].view(B, 1, 1)
    rng   = (v_max - v_min).clamp(min=1e-8)
    V_norm = (2.0 * (V_raw - v_min) / rng - 1.0).masked_fill(wall_mask, 10.0)
    V_norm    = V_norm.to(device)

    # Monta tensor de inputs: [V, c1, 0, b, c4, u]
    pc = params_col.to(device)
    inputs = torch.zeros(B, N, N, 6, device=device)
    inputs[..., 0] = V_norm
    inputs[..., 1] = pc[:, 3].view(B, 1, 1).expand(B, N, N)   # c1 (Diblock)
    # canal 2 = 0: kappa implícito em V via Yukawa
    inputs[..., 3] = pc[:, 0].view(B, 1, 1).expand(B, N, N)   # b  (Kuhn)
    inputs[..., 4] = pc[:, 4].view(B, 1, 1).expand(B, N, N)   # c4 (Alternado)
    inputs[..., 5] = pc[:, 2].view(B, 1, 1).expand(B, N, N)   # u  (Flory-Huggins)

    return inputs, V_norm


def collocation_pde_loss(phi_col, v_col_norm, params_col):
    """
    Perda de colocação (Phase 10): equação de Edwards para amostras SEM ground truth.
    O modelo é penalizado apenas pela violação da física — sem MSE supervisionado.

    Ao contrário da perda supervisionada, a conservação de massa aqui tem peso
    maior (0.5) porque é a única restrição de magnitude disponível sem φ_true.

    Args:
        phi_col    : (B, N, N) — predição do modelo nos pontos de colocação
        v_col_norm : (B, N, N) — V normalizado (output de prepare_collocation_inputs)
        params_col : (B, 5)    — [b, kappa, u, c1, c4] no device correto

    Returns:
        loss : escalar — PDE residual + conservação de massa
    """
    scale = 10_000.0
    N   = phi_col.shape[1]
    dev = phi_col.device

    kx, kz, k_sq_4pi2 = _spectral_ops(N, dev)

    b_val  = params_col[:, 0].view(-1, 1, 1).to(dev)
    u_val  = params_col[:, 2].view(-1, 1, 1).to(dev)
    c1_val = params_col[:, 3].view(-1, 1, 1).to(dev)
    c4_val = params_col[:, 4].view(-1, 1, 1).to(dev)

    v_eff   = _compute_v_eff(v_col_norm, c1_val, c4_val, kx, kz)
    pde_res = _edwards_residual(phi_col, v_eff, u_val, b_val, k_sq_4pi2)

    domain  = (v_col_norm < 9.0).float()
    pde_res = pde_res * domain

    pde_loss  = torch.mean(pde_res ** 2) * scale

    mass_pred = phi_col.sum(dim=(1, 2))
    mass_loss = (F.mse_loss(mass_pred, torch.full_like(mass_pred, 100.0))
                 / (N ** 2) * scale)

    return pde_loss + 0.5 * mass_loss

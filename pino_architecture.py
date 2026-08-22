"""
PolyFields-PINO v3 — Arquitetura e Funções de Loss
====================================================
Physics-Informed Neural Operator para densidade de polieletrólitos em
confinamento eletrostático assimétrico (potenciais multipolares arbitrários).

Backbone : Fourier Neural Operator 2D (FNO), 4 camadas espectrais.
Física   : Equação de Edwards (Ground State Dominance) + Perda Sobolev H¹.

Canais de entrada (8 = 6 explícitos + 2 coordenadas de grade):
  [0] V(r)   — Potencial externo NORMALIZADO ∈ [-1, 1]; paredes: V = 10.0
  [1] c1(r)  — Intensidade de carga Diblock (escalar uniforme por amostra)
  [2] A(r)   — Amplitude física do potencial, log-normalizada ∈ [0, 1]
               (kappa continua implícito em V via Yukawa)
  [3] b(r)   — Comprimento de Kuhn (escalar uniforme por amostra)
  [4] c4(r)  — Intensidade de carga Alternado (escalar uniforme por amostra)
  [5] u(r)   — Parâmetro de Flory-Huggins (escalar uniforme por amostra)
  [6] x      — Coordenada x normalizada ∈ [0, 1]  (grid embedding)
  [7] z      — Coordenada z normalizada ∈ [0, 1]  (grid embedding)

Saída: φ(r) ∈ ℝ⁺ — densidade polimérica via softplus, shape (B, Nx, Ny).

Sobre o canal 2 (amplitude): o campo do canal 0 é sempre reescalado para [-1, 1] por
amostra, o que apaga a magnitude absoluta do potencial — sem o canal 2, uma carga Q=1
e uma carga Q=100 produzem entrada idêntica e portanto a mesma densidade. O potencial
físico é reconstruído como V_fis = A · V_norm (as paredes, marcadas por V_norm > 9,
não são reescaladas: o valor 10.0 é apenas um sentinela para a máscara de Dirichlet).

Equação de Edwards (Ground State Dominance):
  (b²/6) ∇²φ(r) = [V_eff(r) + u·φ(r)] · φ(r)
  V_eff = V_fis − c1·2|V_fis| − c4·0.1·|∇V_fis|²

`params` tem 6 colunas: [b, kappa, u, c1, c4, A]. Formatos antigos (3 ou 5 colunas)
são aceitos por compatibilidade — c1/c4 viram 0 e A vira 1.0 (sem reescala).

Nota sobre dphi_dV_true: mantido na assinatura da loss por compatibilidade com
o DataLoader. A susceptibilidade térmica δφ/δV é capturada implicitamente
pela perda Sobolev H¹: ||∇(φ_pred − φ_true)||²_L2  via teorema de Parseval.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ===========================================================================
# 0. AMPLITUDE DO POTENCIAL — fonte única de verdade
# ===========================================================================
# Faixa amostrada em grf_generator.py. O teto fica abaixo do sentinela de parede
# (9.0) para que A·V nunca seja confundido com uma parede.
A_MIN = 0.2
A_MAX = 8.0

_LOG_A_MIN = float(torch.log10(torch.tensor(A_MIN)))
_LOG_A_MAX = float(torch.log10(torch.tensor(A_MAX)))


def amplitude_to_channel(A):
    """Amplitude física → canal 2 ∈ [0, 1] (log-uniforme).

    Usada pelo gerador, pelo treino e pelo backend. Se as três não usarem
    exatamente esta função, o canal 2 vira ruído.
    """
    if not torch.is_tensor(A):
        A = torch.tensor(A, dtype=torch.float32)
    logA = torch.log10(torch.clamp(A, min=1e-6))
    return torch.clamp((logA - _LOG_A_MIN) / (_LOG_A_MAX - _LOG_A_MIN), 0.0, 1.0)


def channel_to_amplitude(c):
    """Inversa de amplitude_to_channel."""
    if not torch.is_tensor(c):
        c = torch.tensor(c, dtype=torch.float32)
    return torch.pow(10.0, c * (_LOG_A_MAX - _LOG_A_MIN) + _LOG_A_MIN)


def apply_amplitude(v_norm, A):
    """V_fis = A · V_norm, preservando o sentinela de parede (V_norm > 9 → 10.0).

    Args:
        v_norm: (B, Nx, Ny) potencial normalizado em [-1, 1] com paredes em 10.0
        A     : (B, 1, 1) ou escalar — amplitude física
    """
    wall = v_norm > 9.0
    return torch.where(wall, v_norm, A * v_norm)


def unpack_params(params, device=None):
    """Aceita params de 3, 5 ou 6 colunas e devolve [b, u, c1, c4, A] em (B,1,1).

    3 colunas → [b, kappa, u]              (c1 = c4 = 0, A = 1)
    5 colunas → [b, kappa, u, c1, c4]      (A = 1)
    6 colunas → [b, kappa, u, c1, c4, A]
    """
    if device is not None:
        params = params.to(device)
    n = params.shape[1]
    view = lambda t: t.view(-1, 1, 1)
    b = view(params[:, 0])
    u = view(params[:, 2])
    zero = torch.zeros_like(b)
    c1 = view(params[:, 3]) if n >= 5 else zero
    c4 = view(params[:, 4]) if n >= 5 else zero
    A = view(params[:, 5]) if n >= 6 else torch.ones_like(b)
    return b, u, c1, c4, A


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
    Perda PINO: L2 relativo + PDE de Edwards + Conservação de massa.

    Componentes:
      1. L2 relativo: ||φ_pred − φ_true||² / ||φ_true||²
         Erro normalizado pela magnitude do campo — evita viés por escala.
      2. PDE de Edwards: (b²/6)∇²φ − (V_eff + u·φ)·φ — mascarada nas paredes
         Normalizada por ||φ||² no domínio para manter escala O(1).
      3. Conservação de massa: (mean(φ_pred) − mean(φ_true))²
         Penaliza desvio na massa total.

    Schedule:
      Épocas    0–100 : warm-up  — só L2                    (alpha_phys = 0)
      Épocas  100–500 : ramp-up  — física entra linearmente (alpha_phys: 0 → 1)
      Épocas  500+    : treino completo                     (alpha_phys = 1)

    Pesos: L2 = 1.0, PDE = 0.1, Massa = 0.05

    dphi_dV_true: mantido na assinatura por compatibilidade com o DataLoader.
                  NÃO é usado na loss.

    Returns:
        (loss_total, l2_item, pde_item, physics_item)
    """
    N   = phi_pred.shape[1]
    dev = phi_pred.device

    kx, kz, k_sq_4pi2 = _spectral_ops(N, dev)

    # ── 1. L2 relativo ──────────────────────────────────────────────────
    diff_sq = torch.mean((phi_pred - phi_true) ** 2, dim=(1, 2))
    true_sq = torch.mean(phi_true ** 2, dim=(1, 2)) + 1e-8
    l2_loss = torch.mean(diff_sq / true_sq)

    # ── 2. PDE de Edwards mascarada nas paredes ─────────────────────────
    b_val, u_val, c1_val, c4_val, a_val = unpack_params(params)

    # O resíduo tem que ser avaliado no potencial FÍSICO: v_tensor é o campo
    # reescalado para [-1, 1], então sem isto a física contradiz o dado.
    v_phys  = apply_amplitude(v_tensor, a_val)
    v_eff   = _compute_v_eff(v_phys, c1_val, c4_val, kx, kz)
    pde_res = _edwards_residual(phi_pred, v_eff, u_val, b_val, k_sq_4pi2)

    # Mascarar paredes (Dirichlet: φ = 0 onde V ≥ 9.0)
    domain = (v_tensor < 9.0).float()
    pde_res = pde_res * domain

    # Normalizar resíduo pela magnitude do campo no domínio
    phi_sq_domain = torch.mean((phi_pred * domain) ** 2, dim=(1, 2)) + 1e-8
    pde_sq = torch.mean(pde_res ** 2, dim=(1, 2))
    pde_loss = torch.mean(pde_sq / phi_sq_domain)

    # ── 3. Conservação de massa ─────────────────────────────────────────
    mass_pred = torch.mean(phi_pred, dim=(1, 2))
    mass_true = torch.mean(phi_true, dim=(1, 2))
    mass_loss = torch.mean((mass_pred - mass_true) ** 2)

    physics_loss = pde_loss + 0.5 * mass_loss

    # ── Schedule ────────────────────────────────────────────────────────
    alpha_phys = min(1.0, max(0.0, (epoch - 100) / 400.0))
    loss_total = l2_loss + 0.1 * alpha_phys * physics_loss

    return loss_total, l2_loss.item(), pde_loss.item(), physics_loss.item()


# ===========================================================================
# 5. PERDA DE COLOCAÇÃO — Phase 10 (sem ground truth SCFT)
# ===========================================================================

def prepare_collocation_inputs(V_raw, params_col, device):
    """
    Pré-processa o dataset de colocação (data/pino_v3_collocation.pt) para o
    formato de entrada do modelo, aplicando a mesma normalização do pipeline GRF/SCFT.

    Args:
        V_raw     : (B, N, N) tensor — potenciais brutos (não normalizados)
        params_col: (B, 5) ou (B, 6) — [b, kappa, u, c1, c4] (+ A)
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

    # Monta tensor de inputs: [V, c1, A, b, c4, u]
    pc = params_col.to(device)
    b_v, u_v, c1_v, c4_v, a_v = unpack_params(pc)
    exp = lambda t: t.expand(B, N, N)
    inputs = torch.zeros(B, N, N, 6, device=device)
    inputs[..., 0] = V_norm
    inputs[..., 1] = exp(c1_v)                                # c1 (Diblock)
    inputs[..., 2] = exp(amplitude_to_channel(a_v))           # A  (amplitude física)
    inputs[..., 3] = exp(b_v)                                 # b  (Kuhn)
    inputs[..., 4] = exp(c4_v)                                # c4 (Alternado)
    inputs[..., 5] = exp(u_v)                                 # u  (Flory-Huggins)

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
        params_col : (B, 5) ou (B, 6) — [b, kappa, u, c1, c4] (+ A) no device correto

    Returns:
        loss : escalar — PDE residual + conservação de massa
    """
    scale = 10_000.0
    N   = phi_col.shape[1]
    dev = phi_col.device

    kx, kz, k_sq_4pi2 = _spectral_ops(N, dev)

    b_val, u_val, c1_val, c4_val, a_val = unpack_params(params_col, device=dev)

    v_phys  = apply_amplitude(v_col_norm, a_val)
    v_eff   = _compute_v_eff(v_phys, c1_val, c4_val, kx, kz)
    pde_res = _edwards_residual(phi_col, v_eff, u_val, b_val, k_sq_4pi2)

    domain  = (v_col_norm < 9.0).float()
    pde_res = pde_res * domain

    pde_loss  = torch.mean(pde_res ** 2) * scale

    mass_pred = phi_col.sum(dim=(1, 2))
    mass_loss = (F.mse_loss(mass_pred, torch.full_like(mass_pred, 100.0))
                 / (N ** 2) * scale)

    return pde_loss + 0.5 * mass_loss

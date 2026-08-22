import torch
import os
import matplotlib.pyplot as plt


def solve_scft_batch(V_fields, params, N_iter=3000, dt=0.005, target_mass=100.0,
                     mixing=0.5, tol=1e-7, batch_size=64, device='cpu'):
    """
    Solver SCFT batched via Imaginary Time Propagation no regime Ground State Dominance.

    Resolve o estado fundamental de:
        H ψ = μ ψ,   com   H = -(b²/6)∇² + ω(r)

    onde o campo auto-consistente é ω(r) = V_ext(r) + u · φ(r)
    e a densidade polimérica é φ(r) = ψ²(r) · (target_mass / ∫ψ² dr).

    Método: Strang splitting do propagador exp(-Δτ H):
        exp(-Δτ/2 ω) · exp(-Δτ T) · exp(-Δτ/2 ω)

    onde T = -(b²/6)∇² é diagonal no espaço de Fourier e ω(r) é diagonal
    no espaço real. Mixing linear estabiliza a auto-consistência.

    Args:
        V_fields:    (N_samples, Nx, Ny)  potenciais externos (GRF + paredes V>9).
        params:      (N_samples, 3)       [b, kappa, u]. kappa está implícito em V.
        N_iter:      Iterações máximas do propagador imaginário.
        dt:          Passo de tempo imaginário Δτ.
        target_mass: ∫φ dr (conservação canônica).
        mixing:      Fator de mixing linear para ω ∈ (0, 1].
                     0.5 = média entre estado novo e antigo. Estável para u ∈ [-1, 1].
        tol:         ||Δφ||/||φ|| < tol → convergiu.
        batch_size:  Amostras processadas simultaneamente (limitado por VRAM).
        device:      'cpu' ou 'cuda'.

    Returns:
        phi_all:     (N_samples, Nx, Ny)  densidades SCFT convergidas.
    """
    N_samples = V_fields.shape[0]
    Nx = V_fields.shape[1]
    Ny = V_fields.shape[2]

    # ─── Frequências espectrais (compartilhadas entre todos os samples) ───
    kx = torch.fft.fftfreq(Nx, device=device)
    ky = torch.fft.fftfreq(Ny, device=device)
    KX, KY = torch.meshgrid(kx, ky, indexing='ij')
    k_sq = 4.0 * torch.pi**2 * (KX**2 + KY**2)  # (Nx, Ny)

    phi_all = torch.zeros(N_samples, Nx, Ny)
    n_batches = (N_samples + batch_size - 1) // batch_size

    for bi in range(n_batches):
        s = bi * batch_size
        e = min(s + batch_size, N_samples)
        B = e - s

        V = V_fields[s:e].to(device)                     # (B, Nx, Ny)
        b_val = params[s:e, 0].to(device).view(B, 1, 1)  # Kuhn length
        u_val = params[s:e, 2].to(device).view(B, 1, 1)  # Flory-Huggins

        wall_mask = (V > 9.0)

        # ─── Propagador cinético: exp(-Δτ · (b²/6) · 4π²k²) ───
        # Pré-computado por batch pois b varia por amostra
        kin_coeff = b_val**2 / 6.0                                       # (B, 1, 1)
        kin_prop = torch.exp(-dt * kin_coeff * k_sq.unsqueeze(0))        # (B, Nx, Ny)

        # ─── Inicialização: fator de Boltzmann clampado ───
        V_init = V.clone()
        V_init[wall_mask] = 0.0
        psi = torch.exp(-V_init.clamp(-3.0, 3.0))
        psi[wall_mask] = 0.0

        # Normalizar ψ: ||ψ||₂ = 1
        psi_norm = torch.sqrt(torch.sum(psi**2, dim=(1, 2), keepdim=True) + 1e-12)
        psi = psi / psi_norm

        # ─── Densidade e campo auto-consistente iniciais ───
        phi = psi**2
        phi = phi / (torch.sum(phi, dim=(1, 2), keepdim=True) + 1e-12) * target_mass
        omega = V + u_val * phi

        # ─── Loop de Imaginary Time Propagation ───
        for it in range(N_iter):
            phi_old = phi

            # Strang splitting: V/2 → T → V/2
            # 1. Half-step potencial (espaço real)
            exp_half_V = torch.exp((-dt / 2.0 * omega).clamp(-50.0, 50.0))
            psi = psi * exp_half_V

            # 2. Full-step cinético (espaço de Fourier)
            psi_fft = torch.fft.fft2(psi)
            psi_fft = psi_fft * kin_prop
            psi = torch.fft.ifft2(psi_fft).real

            # 3. Half-step potencial (mesmo ω — não muda mid-step)
            psi = psi * exp_half_V

            # ψ do ground state é real e não-negativo; enforce BCs
            psi = psi.clamp(min=0.0)
            psi[wall_mask] = 0.0

            # Renormalizar ψ
            psi_norm = torch.sqrt(torch.sum(psi**2, dim=(1, 2), keepdim=True) + 1e-12)
            psi = psi / psi_norm

            # ─── Atualizar φ com mixing linear ───
            phi_new = psi**2
            phi_new = phi_new / (torch.sum(phi_new, dim=(1, 2), keepdim=True) + 1e-12) * target_mass
            phi = mixing * phi_new + (1.0 - mixing) * phi_old

            # Atualizar campo auto-consistente
            omega = V + u_val * phi

            # ─── Checar convergência ───
            delta = torch.norm((phi - phi_old).reshape(B, -1), dim=1)
            delta = delta / (torch.norm(phi.reshape(B, -1), dim=1) + 1e-12)

            if delta.max().item() < tol:
                break

        phi_all[s:e] = phi.cpu()

        # Log de progresso
        if (bi + 1) % max(1, n_batches // 10) == 0 or bi == 0:
            print(f"  -> Batch {bi+1}/{n_batches}: "
                  f"δ_max={delta.max().item():.2e}, "
                  f"iter={it+1}, "
                  f"φ_range=[{phi.min().item():.4f}, {phi.max().item():.4f}]")

    return phi_all


def run_scft_solver(dataset_path="data/pino_v3_grf_dataset.pt"):
    """
    Solucionador SCFT real: recebe os potenciais GRF e resolve a densidade polimérica
    via Imaginary Time Propagation no regime de Ground State Dominance.

    Substitui o mock exp(-2V) por um solver auto-consistente que incorpora:
    - Entropia conformacional via operador cinético (b²/6)∇²ψ
    - Volume excluído / Flory-Huggins (u·φ) via campo auto-consistente ω
    - Conservação de massa (normalização canônica ∫φ = target_mass)
    - Condição de contorno de Dirichlet nas paredes (ψ = 0 onde V > 9)

    O solver produz o dataset completo no formato:
        {"V": ..., "params": ..., "phi_scft": ..., "dphi_dV": ...}

    dphi_dV é um proxy de linear response (δφ/δV ≈ -2φ) mantido para
    compatibilidade do DataLoader. A loss H¹ Sobolev não utiliza este campo.
    """
    print("[*] Iniciando Solucionador SCFT (Self-Consistent Field Theory)...")
    print("[*] Método: Imaginary Time Propagation + Strang Splitting")
    print("[*] Regime: Ground State Dominance (cadeia longa)")

    if not os.path.exists(dataset_path):
        print(f"[!] Erro: {dataset_path} não encontrado. Execute grf_generator.py primeiro.")
        return

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"[*] Device: {device}")

    data = torch.load(dataset_path, weights_only=False)
    V_fields = data["V"]
    params = data["params"]
    num_samples = V_fields.shape[0]
    grid_size = V_fields.shape[1]

    print(f"[*] Resolvendo {num_samples} cenários SCFT (grid {grid_size}x{grid_size})...")
    print(f"[*] Parâmetros: b ∈ [{params[:, 0].min():.2f}, {params[:, 0].max():.2f}], "
          f"u ∈ [{params[:, 2].min():.2f}, {params[:, 2].max():.2f}]")

    phi_scft = solve_scft_batch(
        V_fields, params,
        N_iter=800,
        dt=0.005,
        target_mass=100.0,
        mixing=0.5,
        tol=1e-5,
        batch_size=64 if device.type == 'cuda' else 16,
        device=device
    )

    # ─── Derivada δφ/δV (proxy de linear response) ───
    # No ground state, a resposta linear é δφ/δV(r) ≈ -2φ(r)/ΔE
    # onde ΔE é o gap espectral. Sem computar ΔE, salvamos -2φ como proxy.
    # A loss H¹ Sobolev usa gradientes espaciais de φ (via FFT), não este campo.
    dphi_dV = -2.0 * phi_scft * (V_fields < 9.0).float()

    # ─── Salvar dataset ───
    out_path = "data/pino_v3_dataset_complete.pt"
    torch.save({
        "V": V_fields.detach(),
        "params": params,
        "phi_scft": phi_scft.detach(),
        "dphi_dV": dphi_dV.detach()
    }, out_path)

    # ─── Estatísticas ───
    masses = phi_scft.sum(dim=(1, 2))
    print(f"\n[+] Dataset SCFT salvo em {out_path}")
    print(f"    Amostras: {num_samples}")
    print(f"    φ range: [{phi_scft.min().item():.6f}, {phi_scft.max().item():.6f}]")
    print(f"    φ massa média: {masses.mean().item():.2f} ± {masses.std().item():.2f}")
    print(f"    φ fração com parede: {(V_fields > 9.0).any(dim=(1,2)).sum().item()}/{num_samples}")

    # ─── Plot comparativo ───
    fig, axes = plt.subplots(2, 3, figsize=(15, 9))

    # Amostra 0
    axes[0, 0].imshow(V_fields[0].numpy(), cmap='inferno')
    axes[0, 0].set_title('V_ext (GRF) — Amostra 0')
    axes[0, 0].axis('off')

    im1 = axes[0, 1].imshow(phi_scft[0].numpy(), cmap='viridis')
    axes[0, 1].set_title(f'φ SCFT — massa={phi_scft[0].sum().item():.1f}')
    axes[0, 1].axis('off')
    plt.colorbar(im1, ax=axes[0, 1], fraction=0.046)

    im2 = axes[0, 2].imshow(dphi_dV[0].numpy(), cmap='coolwarm')
    axes[0, 2].set_title('δφ/δV (Linear Response)')
    axes[0, 2].axis('off')
    plt.colorbar(im2, ax=axes[0, 2], fraction=0.046)

    # Amostra com parede (se existir)
    wall_idx = (V_fields > 9.0).any(dim=(1, 2)).nonzero(as_tuple=True)[0]
    if len(wall_idx) > 0:
        wi = wall_idx[0].item()
    else:
        wi = 1

    axes[1, 0].imshow(V_fields[wi].numpy(), cmap='inferno')
    axes[1, 0].set_title(f'V_ext (GRF) — Amostra {wi} (com parede)')
    axes[1, 0].axis('off')

    im3 = axes[1, 1].imshow(phi_scft[wi].numpy(), cmap='viridis')
    axes[1, 1].set_title(f'φ SCFT — massa={phi_scft[wi].sum().item():.1f}')
    axes[1, 1].axis('off')
    plt.colorbar(im3, ax=axes[1, 1], fraction=0.046)

    # Histograma de massas
    axes[1, 2].hist(masses.numpy(), bins=50, color='#818cf8', edgecolor='black', alpha=0.8)
    axes[1, 2].axvline(100.0, color='red', linestyle='--', label='Target mass')
    axes[1, 2].set_xlabel('Massa total ∫φ dr')
    axes[1, 2].set_ylabel('Contagem')
    axes[1, 2].set_title('Distribuição de Massa (Conservação Canônica)')
    axes[1, 2].legend()

    plt.tight_layout()
    plt.savefig("data/exemplo_scft_solution.png", dpi=150)
    plt.close()
    print("[+] Imagem comparativa salva em data/exemplo_scft_solution.png")


if __name__ == "__main__":
    run_scft_solver()

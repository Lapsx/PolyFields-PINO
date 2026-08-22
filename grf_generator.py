"""
Gerador de cenários para o PolyFields-PINO v3
==============================================
Produz `data/pino_v3_grf_dataset.pt` com:

    V      : (N, 100, 100) potencial NORMALIZADO em [-1, 1], paredes/obstáculos = 10.0
    params : (N, 6)        [b, kappa, u, c1, c4, A]

Três eixos de variação que não existiam na versão anterior:

1. TIPO DE POLÍMERO (c1, c4)
   A versão anterior gerava params de 3 colunas, então c1 e c4 entravam zerados no
   treino e tanto o seletor de tipo quanto o slider de intensidade do webapp moviam
   canais que a rede nunca viu variar. Aqui o dataset é balanceado em três terços:
   neutro (c1=c4=0), Diblock (c1>0) e Alternado (c4>0).

2. GEOMETRIA DOS OBSTÁCULOS
   A versão anterior só criava parede reta colada no topo (`grf[:wall_pos,:] = 10.0`),
   enquanto o backend sempre apresenta uma esfera circular centrada — a rede tinha
   canais de grade (x, z) explícitos e podia memorizar "parede fica em cima".
   Aqui há três geometrias: sem obstáculo, parede reta (em qualquer das 4 bordas) e
   1-2 obstáculos circulares em posição e raio aleatórios.

3. AMPLITUDE DO POTENCIAL (A)
   O campo é sempre reescalado para [-1, 1], o que apaga a magnitude absoluta: no
   backend, Q=1 e Q=100 produziam entrada idêntica e a mesma densidade. A amplitude
   física vai na coluna 5 de params e entra na rede pelo canal 2 (antes reservado).
   O potencial físico é V_fis = A · V_norm — reconstruído pelo solver, pela loss e
   pelo backend com `apply_amplitude`, de pino_architecture.py.
"""

import torch
import numpy as np
import os
import matplotlib.pyplot as plt
from scipy.stats import qmc

from pino_architecture import A_MIN, A_MAX

# Tipos de polímero (mesmos códigos do <select> do frontend)
TIPO_NEUTRO = 0
TIPO_DIBLOCK = 1
TIPO_ALTERNADO = 4


def generate_grf_2d(size, alpha, tau, device='cpu'):
    """Gaussian Random Field 2D com espectro de Matérn, via FFT.

    alpha: suavidade (rugosidade);  tau: comprimento de correlação.
    Saída normalizada em [-1, 1].
    """
    kx = torch.fft.fftfreq(size) * size
    ky = torch.fft.fftfreq(size) * size
    kx, ky = torch.meshgrid(kx, ky, indexing='ij')
    k_sq = kx**2 + ky**2

    spectrum = 1.0 / (tau**2 + k_sq)**(alpha / 2.0)
    spectrum[0, 0] = 0.0                      # remove a média

    noise = torch.randn(size, size, dtype=torch.cfloat)
    grf = torch.fft.ifft2(noise * spectrum).real

    grf = (grf - grf.min()) / (grf.max() - grf.min() + 1e-12)
    return (grf * 2.0 - 1.0).to(device)


def _add_straight_wall(grf, rng):
    """Parede reta em uma das quatro bordas (antes: sempre no topo)."""
    size = grf.shape[0]
    thick = rng.integers(8, size // 4)
    side = rng.integers(0, 4)
    if side == 0:   grf[:thick, :] = 10.0
    elif side == 1: grf[-thick:, :] = 10.0
    elif side == 2: grf[:, :thick] = 10.0
    else:           grf[:, -thick:] = 10.0
    return grf


def _add_circular_obstacles(grf, rng):
    """1 ou 2 discos de raio e posição aleatórios — a geometria do webapp.

    O backend usa esfera de raio a=1 numa caixa L=8 sobre grade 100, ou seja
    ~12.5 px. A faixa 6-22 px cobre isso com folga dos dois lados.
    """
    size = grf.shape[0]
    ii, jj = torch.meshgrid(torch.arange(size).float(),
                            torch.arange(size).float(), indexing='ij')
    for _ in range(int(rng.integers(1, 3))):
        r = float(rng.uniform(6.0, 22.0))
        ci = float(rng.uniform(r, size - r))
        cj = float(rng.uniform(r, size - r))
        grf[((ii - ci)**2 + (jj - cj)**2) < r**2] = 10.0
    return grf


def generate_v3_dataset(num_samples=15000, grid_size=100, output_dir='data', seed=42):
    os.makedirs(output_dir, exist_ok=True)
    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)
    print(f"[*] Gerando {num_samples} cenários (GRF + geometria + carga)...")

    V_fields = torch.zeros(num_samples, grid_size, grid_size)
    params = torch.zeros(num_samples, 6)      # [b, kappa, u, c1, c4, A]

    # LHS: alpha, tau, geometria, b, kappa, u, log10(A), intensidade de carga
    #
    # u ∈ [U_MIN, 1.0] com U_MIN > 0: o funcional de Edwards/Flory em 2D com termo
    # quártico ATRATIVO (u < 0) não tem energia mínima — a solução auto-focaliza e,
    # na rede discreta, colapsa num único pixel. Medido com o solver atual:
    #     u = -0.20 → 99.9% da massa em 1 célula
    #     u = -0.05 → 98.2% da massa em 1 célula
    #     u =  0.00 → 14.9% (ainda degenerado)
    #     u = +0.05 → 4.7% da massa no pico, 130 células ativas  ← saudável
    # A faixa antiga era [-1, 1], ou seja METADE de todo dataset gerado até aqui era
    # um delta de Dirac. Como um delta fica preso no argmin(V), esses casos também
    # eram cegos a c1, c4 e A — nenhuma perturbação monótona move o argmin.
    U_MIN = 0.05
    sampler = qmc.LatinHypercube(d=8, seed=seed)
    lhs = sampler.random(n=num_samples)
    l_bounds = [2.0, 1.0, 0.0, 0.5, 0.1, U_MIN, np.log10(A_MIN), 0.2]
    u_bounds = [4.0, 10.0, 1.0, 2.0, 5.0,   1.0, np.log10(A_MAX), 2.0]
    lhs = qmc.scale(lhs, l_bounds, u_bounds)

    # Tipo de polímero balanceado em três terços exatos, depois embaralhado —
    # sorteio independente deixaria os terços desiguais por acaso.
    tipos = np.repeat([TIPO_NEUTRO, TIPO_DIBLOCK, TIPO_ALTERNADO],
                      np.ceil(num_samples / 3).astype(int))[:num_samples]
    rng.shuffle(tipos)

    n_reta = n_circ = n_livre = 0
    for i in range(num_samples):
        alpha, tau, geom = lhs[i, 0], lhs[i, 1], lhs[i, 2]
        grf = generate_grf_2d(grid_size, alpha, tau)

        # ~35% sem obstáculo, ~30% parede reta, ~35% obstáculos circulares
        if geom < 0.35:
            n_livre += 1
        elif geom < 0.65:
            grf = _add_straight_wall(grf, rng); n_reta += 1
        else:
            grf = _add_circular_obstacles(grf, rng); n_circ += 1

        V_fields[i] = grf

        params[i, 0] = lhs[i, 3]              # b     (Kuhn)
        params[i, 1] = lhs[i, 4]              # kappa (rótulo; implícito em V)
        params[i, 2] = lhs[i, 5]              # u     (Flory-Huggins)
        params[i, 5] = 10.0 ** lhs[i, 6]      # A     (amplitude física)

        intensidade = lhs[i, 7]
        if tipos[i] == TIPO_DIBLOCK:
            params[i, 3] = intensidade        # c1
        elif tipos[i] == TIPO_ALTERNADO:
            params[i, 4] = intensidade        # c4

        if (i + 1) % 1000 == 0:
            print(f"  -> {i+1}/{num_samples}")

    save_path = os.path.join(output_dir, "pino_v3_grf_dataset.pt")
    torch.save({"V": V_fields, "params": params, "tipo": torch.tensor(tipos)}, save_path)

    print(f"\n[+] Salvo em {save_path}")
    print(f"    geometria : {n_livre} sem obstáculo | {n_reta} parede reta | {n_circ} circular")
    print(f"    tipo      : {(tipos==TIPO_NEUTRO).sum()} neutro | "
          f"{(tipos==TIPO_DIBLOCK).sum()} diblock | {(tipos==TIPO_ALTERNADO).sum()} alternado")
    print(f"    b     ∈ [{params[:,0].min():.2f}, {params[:,0].max():.2f}]")
    print(f"    u     ∈ [{params[:,2].min():.2f}, {params[:,2].max():.2f}]")
    print(f"    c1    ∈ [{params[:,3].min():.2f}, {params[:,3].max():.2f}]  "
          f"(não-nulos: {(params[:,3]>0).sum().item()})")
    print(f"    c4    ∈ [{params[:,4].min():.2f}, {params[:,4].max():.2f}]  "
          f"(não-nulos: {(params[:,4]>0).sum().item()})")
    print(f"    A     ∈ [{params[:,5].min():.2f}, {params[:,5].max():.2f}]")
    print(f"    fração de células em parede: {(V_fields > 9.0).float().mean().item()*100:.1f}%")

    # Amostra visual: uma de cada geometria
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))
    tem_parede = (V_fields > 9.0).any(dim=(1, 2))
    idx_livre = (~tem_parede).nonzero()[0].item()
    idx_parede = tem_parede.nonzero()
    for ax, idx, tit in zip(axes,
                            [idx_livre, idx_parede[0].item(), idx_parede[-1].item()],
                            ['sem obstáculo', 'com obstáculo (a)', 'com obstáculo (b)']):
        im = ax.imshow(V_fields[idx].numpy(), cmap='inferno')
        ax.set_title(f"{tit}\nA={params[idx,5]:.2f}  c1={params[idx,3]:.2f}  c4={params[idx,4]:.2f}",
                     fontsize=9)
        ax.axis('off')
        plt.colorbar(im, ax=ax, fraction=0.046)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "exemplo_grf.png"), dpi=130)
    plt.close()
    print(f"[+] Amostras em {output_dir}/exemplo_grf.png")


if __name__ == "__main__":
    generate_v3_dataset(num_samples=15000)

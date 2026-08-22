"""
Verificação rápida do dataset antes de queimar 12h de GPU
=========================================================
Roda em segundos e falha ruidosamente se o dataset não for o esperado. Existe
porque cada erro abaixo já aconteceu neste projeto e só apareceria depois de uma
sessão inteira de treino desperdiçada.

Uso:
    python verifica_dataset.py [caminho]     # default: data/pino_v3_dataset_complete.pt

Sai com código 1 se algo estiver errado, para dar para encadear com `&&` no notebook.
"""

import sys
import os
import torch

CAMINHO = sys.argv[1] if len(sys.argv) > 1 else "data/pino_v3_dataset_complete.pt"

falhas = []
avisos = []


def checa(cond, msg_falha, msg_ok):
    if cond:
        print(f"  [ok]   {msg_ok}")
    else:
        print(f"  [FALHA] {msg_falha}")
        falhas.append(msg_falha)


print(f"[*] Verificando {CAMINHO}")
if not os.path.exists(CAMINHO):
    print(f"[FALHA] arquivo não existe. O treino espera exatamente "
          f"'data/pino_v3_dataset_complete.pt'.")
    sys.exit(1)

print(f"    tamanho: {os.path.getsize(CAMINHO)/1e9:.2f} GB")
d = torch.load(CAMINHO, weights_only=False, map_location='cpu')

# ── estrutura ──────────────────────────────────────────────────────────────
checa(set(d.keys()) >= {"V", "params", "phi_scft", "dphi_dV"},
      f"chaves faltando: {set(d.keys())}",
      f"chaves presentes: {sorted(d.keys())}")

V, P, PHI = d["V"], d["params"], d["phi_scft"]
n = V.shape[0]
print(f"    amostras: {n} | grade: {V.shape[1]}x{V.shape[2]}")

# ── params: 6 colunas é o formato novo ─────────────────────────────────────
checa(P.shape[1] == 6,
      f"params tem {P.shape[1]} colunas, esperado 6 [b, kappa, u, c1, c4, A]. "
      f"Este é o dataset ANTIGO — c1/c4/amplitude não existem nele.",
      f"params com 6 colunas [b, kappa, u, c1, c4, A]")

# ── u > 0: abaixo de zero a solução colapsa num pixel ──────────────────────
if P.shape[1] >= 3:
    u_min = P[:, 2].min().item()
    checa(u_min > 0.0,
          f"u_min = {u_min:.3f} <= 0. Para u < 0 o funcional de Edwards/Flory em 2D "
          f"não tem energia mínima e a solução colapsa num único pixel — "
          f"{(P[:, 2] <= 0).sum().item()} amostras afetadas.",
          f"u ∈ [{u_min:.3f}, {P[:, 2].max():.3f}], todo positivo")

# ── tipo de polímero presente ──────────────────────────────────────────────
if P.shape[1] >= 5:
    n_c1 = (P[:, 3] > 0).sum().item()
    n_c4 = (P[:, 4] > 0).sum().item()
    checa(n_c1 > 0 and n_c4 > 0,
          f"c1 não-nulos: {n_c1}, c4 não-nulos: {n_c4}. Sem variação nesses canais o "
          f"seletor de tipo de polímero do webapp não vai ter efeito.",
          f"tipo: {n_c1} Diblock (c1), {n_c4} Alternado (c4), "
          f"{((P[:, 3] == 0) & (P[:, 4] == 0)).sum().item()} neutro")

# ── amplitude ──────────────────────────────────────────────────────────────
if P.shape[1] >= 6:
    checa(P[:, 5].std().item() > 0.01,
          f"amplitude A é constante ({P[:, 5].mean():.3f}) — o canal 2 não vai "
          f"carregar informação.",
          f"A ∈ [{P[:, 5].min():.2f}, {P[:, 5].max():.2f}]")

# ── massa conservada ───────────────────────────────────────────────────────
massas = PHI.sum(dim=(1, 2))
checa(abs(massas.mean().item() - 100.0) < 1.0 and massas.std().item() < 1.0,
      f"massa média {massas.mean():.2f} ± {massas.std():.2f}, esperado 100.00 ± 0.00",
      f"massa {massas.mean():.2f} ± {massas.std():.2f}")

# ── colapso: fração da massa no pixel do pico ──────────────────────────────
frac = PHI.amax(dim=(1, 2)) / (massas + 1e-12)
n_spike = (frac > 0.5).sum().item()
# Poucas amostras colapsadas são toleráveis: o pino_train_physics.py as descarta ao
# carregar. Muitas indicam faixa de parâmetros errada (u perto de zero, A alta demais)
# e aí o dataset precisa ser regerado.
checa(n_spike / n < 0.01,
      f"{n_spike}/{n} ({n_spike/n*100:.1f}%) amostras colapsadas num único pixel. "
      f"Acima de 1% o problema é a faixa de parâmetros — regere com U_MIN maior "
      f"ou A_MAX menor.",
      f"colapso sob controle: {n_spike}/{n} amostras "
      f"({n_spike/n*100:.2f}%, descartadas no treino); "
      f"fração no pico mediana {frac.median()*100:.2f}%")

# ── Dirichlet nas paredes ──────────────────────────────────────────────────
paredes = V > 9.0
if paredes.any():
    phi_parede = PHI[paredes].abs().max().item()
    checa(phi_parede < 1e-6,
          f"φ nas paredes chega a {phi_parede:.2e}, deveria ser 0",
          f"φ = 0 nas paredes ({paredes.float().mean()*100:.1f}% das células)")
    frac_com_parede = paredes.any(dim=1).any(dim=1).float().mean().item()
    if frac_com_parede < 0.3:
        avisos.append(f"só {frac_com_parede*100:.0f}% das amostras têm obstáculo; "
                      f"o backend sempre apresenta uma esfera")
else:
    avisos.append("nenhuma amostra tem parede/obstáculo — o backend sempre tem uma esfera")

# ── V normalizado ──────────────────────────────────────────────────────────
fora = V[~paredes] if paredes.any() else V
checa(fora.min() >= -1.001 and fora.max() <= 1.001,
      f"V fora das paredes ∈ [{fora.min():.2f}, {fora.max():.2f}], esperado [-1, 1] "
      f"(a amplitude física vai em params[:, 5], não no campo)",
      f"V normalizado em [{fora.min():.2f}, {fora.max():.2f}]")

# ── resultado ──────────────────────────────────────────────────────────────
print()
for a in avisos:
    print(f"  [aviso] {a}")
if falhas:
    print(f"\n[FALHA] {len(falhas)} problema(s). NÃO inicie o treino.")
    sys.exit(1)
print("\n[+] Dataset aprovado. Pode treinar.")

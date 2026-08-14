# Guia Definitivo PINO (Physics-Informed Neural Operator) - V3

Este documento resume a arquitetura computacional do projeto e serve como um guia direto de como migrar e executar o treinamento na infraestrutura do Kaggle.

## 1. Fluxo de Funcionamento do Projeto (Visão Geral)

O projeto aproxima soluções termodinâmicas complexas de polímeros (Teoria de Campos Autoconsistentes - SCFT) usando uma rede neural operadora no domínio das frequências espaciais. O pipeline opera nas seguintes fases estritas:

### A. Geração de Dados Físicos (Ground Truth)
1. **`grf_generator.py`**: Utiliza amostragem de hipercubo latino (LHS) para sortear parâmetros físicos (`b`, `kappa`, `u`). Gera a topologia de fundo (obstáculos e campo base) via Gaussian Random Fields em domínio de Fourier, criando 5.000 amostras iniciais não-viciadas.
2. **`scft_solver.py`**: Atua como o "Oráculo Termodinâmico". Ele relaxa os campos gerados assumindo a distribuição canônica (Ginzburg-Landau/Edwards) e gera a **Densidade Polimérica Real ($\phi_{SCFT}$)**. Crucialmente, ele usa o `autograd` do PyTorch para calcular o Jacobiano local estrito ($\frac{\partial \phi}{\partial V}$), essencial para a otimização de derivadas (Sobolev).
3. Produz o arquivo pesado central: `data/pino_v3_dataset_complete.pt`.

### B. A Arquitetura da Rede (Fourier Neural Operator)
* Definida em **`pino_architecture.py`**.
* A rede não processa o espaço $(x, y)$ apenas com convoluções padrão. Ela transforma o campo de entrada para o domínio das frequências ($\mathcal{F}$), corta os modos de alta frequência ruidosos (`modes=16`), multiplica por matrizes de pesos no espaço de Fourier e faz a transformada inversa ($\mathcal{F}^{-1}$). Isso permite resolver equações diferenciais globais em tempo constante.

### C. Treinamento Unificado (O Coração da V3)
* Realizado por **`pino_train_physics.py`**.
* Otimiza a rede resolvendo três frentes de penalização simultâneas (*PINO Loss*):
  1. **L2 Data Loss**: Ancoragem absoluta (a rede tenta copiar o SCFT bit a bit).
  2. **Sobolev Loss**: Ancoragem de suscetibilidade (a rede aprende como o SCFT "reage" a pequenas perturbações do campo. Sem isso, a termodinâmica falha em generalizar).
  3. **PDE Collocation Loss**: Força bruta na Equação de Edwards. Extrai gradientes via derivadas espectrais diretas e pune a rede se ela violar a mecânica quântica/estatística.

### D. Inferência Web e Filtro Newton-Raphson
* Localizado em **`backend/main.py`**.
* Quando o modelo infere uma densidade rápida, ela inevitavelmente possui pequenos ruídos numéricos. O backend aplica um passo analítico (Single-Step Newton-Raphson) baseado no resíduo da PDE real e ajusta a conservação de massa (ajuste do Potencial Químico).

---

## 2. Guia de Treinamento no Kaggle

Para treinar uma rede de 5.000 amostras com 16 Modos de Fourier e cálculos analíticos na Loss, você usará a arquitetura **P100** ou **T4 x2** (DataParallel) gratuita do Kaggle.

### Passo 1: Organizar e Fazer o Upload dos Dados (Kaggle Dataset)
O arquivo de dados é pesado (vários GBs), portanto **NÃO** deve ser subido como código. Suba como um Dataset:
1. No painel inicial do Kaggle, vá em **Datasets > New Dataset**.
2. Faça o upload exclusivo do arquivo local `data/pino_v3_dataset_complete.pt`.
3. Dê um nome, ex: `pino-v3-polymer-complete`.

### Passo 2: Criar o Notebook
1. Crie um novo Notebook no Kaggle e clique em **Add Input** (lado direito) para anexar o Dataset que você acabou de criar.
2. Na aba lateral em **Session Options**, mude o Acelerador para **GPU T4 x2**.

### Passo 3: Script Executável (Célula do Kaggle)
Crie uma célula contendo a fusão da arquitetura e do script de treino para evitar lidar com múltiplos arquivos no ambiente nativo do Kaggle. Ou apenas anexe os scripts e execute. A forma mais elegante é baixar o seu repositório inteiro e rodar via terminal.

**Na primeira célula, rode:**
```bash
!git clone https://github.com/SeuUsuario/PINO_Polymer_Webmap.git
%cd PINO_Polymer_Webmap
# Opcional se os scripts já não estiverem no GitHub:
# Substitua arquivos manualmente se não der git push
```

**Na segunda célula, prepare os dados e inicie o treino:**
```python
import os
import shutil

# O Kaggle monta os datasets anexados na pasta /kaggle/input/
DATASET_PATH_KAGGLE = "/kaggle/input/pino-v3-polymer-complete/pino_v3_dataset_complete.pt"

# Criar as pastas que o script espera
os.makedirs("data", exist_ok=True)
os.makedirs("weights", exist_ok=True)

# Link Simbólico ou Cópia Rápida para o script achar na pasta data/
if not os.path.exists("data/pino_v3_dataset_complete.pt"):
    shutil.copy(DATASET_PATH_KAGGLE, "data/pino_v3_dataset_complete.pt")
    print("[+] Dataset montado e preparado!")

# O Kaggle permite 12 horas corridas de GPU.
# Execute o treinamento nativamente:
!python pino_train_physics.py
```

### Passo 4: Salvar os Pesos (Sobrevivência do Modelo)
Tudo que é escrito em `weights/` dentro de `/kaggle/working/` (que é o diretório padrão onde o notebook roda) será compactado pelo Kaggle ao fim da sessão.
No final do notebook, adicione uma célula de salvaguarda explícita, caso precise baixar manualmente antes do kernel morrer por limite de tempo:

```python
import IPython
from IPython.display import FileLink
# Crie um link rápido para baixar os pesos atuais
FileLink('weights/pino_v3_phase5_physics.pth')
```

### Dicas Finais:
* A `loss` vai começar alta por causa da reestruturação. Deixe rodar. O aprendizado cruzado (L2 + Sobolev) reduz drasticamente as variâncias após umas 3.000 épocas.
* Se a sessão cair com 8.000 épocas, não se preocupe. O script foi codificado para retomar automaticamente de `weights/pino_v3_phase5_physics.pth`. Simplesmente reanexe os arquivos resultantes como um novo dataset, coloque na pasta correta e o script retomará de onde parou.

<div align="center">
  <a href="#-polyfields-pino---english-version">🇺🇸 Click here for the English Version</a>
</div>

# 🇧🇷 PolyFields-PINO (Real-Time Polymer Density Predictor for Asymmetric Multipole Potentials Using Physics-Informed Neural Operators)

> ⚠️ **Status: Work in Progress (WIP)**
> O treinamento final do pipeline encontra-se temporariamente pausado devido a limitações severas de acesso a hardware de processamento (clusters de GPU). O repositório e os scripts estão sendo disponibilizados para que a comunidade científica possa testar, validar a física e evoluir a arquitetura de forma colaborativa.

## 🚀 Como Usar (Instalação e Execução)

1. **Instale as Dependências:**
   O backend de inferência requer bibliotecas padrão de Machine Learning e APIs web.
   ```bash
   pip install -r requirements.txt
   ```

2. **Baixe os Pesos do Modelo:**
   O arquivo do modelo treinado está hospedado na aba de **Releases** deste repositório (lado direito da página no GitHub).
   * Baixe o arquivo `pino_v3_phase5_physics.pth` do release `v1.0-alpha`.
   * Salve-o dentro da pasta `weights/` na raiz do projeto (crie a pasta se não existir).

3. **Inicie o WebApp (Ambiente Interativo):**
   Para explorar as predições de densidade polimérica em tempo real, utilize o script de inicialização automático:
   ```bash
   bash iniciar_webapp.sh
   ```
   *(Este comando inicializará o servidor FastAPI no backend e abrirá automaticamente o visualizador `index.html` no seu navegador).*

---

## Objetivo do Projeto

O **PolyFields-PINO** é a versão mais recente da arquitetura de predição de densidades poliméricas. Nascido como um Fourier Neural Operator (FNO) clássico em versões anteriores, o projeto tem como objetivo principal alterar **o paradigma de mapeamento puramente estatístico (dados para dados) usado anteriormente, em direção a um aprendizado profundo guiado pelas leis fundamentais da mecanica estatistica**.

A base teórica e de engenharia desta versão introduz o **Physics-Informed Neural Operator (PINO)**, buscando criar um modelo capaz de generalizar cenários de confinamento e interação eletrostática sobre polieletrólitos sem depender exclusivamente de simulações exaustivas de Self-Consistent Field Theory (SCFT).

Os principais pontos teóricos que sustentam este objetivo são:
1. **Representação Contínua da Carga**: Utilização de coeficientes de Fourier para modelar sequências de carga do polímero, comprimindo um espaço combinatório muito grande para um espaço de frequência reduzido e interpretável.
2. **Diversidade Topológica Infinita**: Substituição de cenários rígidos de ancoragem por *Gaussian Random Fields (GRFs)* com parâmetros para simular as mais variadas topologias de confinamento e potenciais. O espaço paramétrico que dita a topologia desses campos e as propriedades termodinâmicas do polímero são amostrados usando **Latin Hypercube Sampling (LHS)**, garantindo uma boa cobertura estatística.
3. **Exploração da Simetria Física (Data Augmentation D4)**: Pelo fato da física de matrizes de confinamento e a resposta polimérica ($\phi$) serem espacialmente isotrópicas e invariantes a rotações e espelhamentos, aplica-se o Grupo de Simetria D4 (rotações 90º, 180º, 270º e reflexões) offline para multiplicar exponencialmente o *Ground Truth* do SCFT sem custo computacional, mitigando o *overfitting*, aumentando o espaço de fase do treino e evitando respostas paradoxais como artefatos do treino.
4. **Treinamento de Sobolev**: Injeção das derivadas do modelo na Função de Custo (*Loss*). A rede não aprende apenas a densidade $\phi(\mathbf{r})$, mas também a sua suscetibilidade térmica exata $\left( \frac{\delta \phi}{\delta V} \right)$, aumentando o valor informacional de cada inferencia.
5. **Resíduos Físicos & Aprendizado Ativo**: Utilização de dezenas de milhares de pontos de colocação sem rótulo, onde o PINO é punido matematicamente se violar as equações diferenciais de Edwards e Debye-Hückel. Se a física falhar e a incerteza subir, um ciclo de Active Learning chama o SCFT sob demanda para corrigir a falha.
6. **Precisão Ampliada no Front-end**: Como o PINO prevê o resultado em 0.05 segundos com extrema precisão, sua inferência servirá como o chute inicial para um solucionador de passo único (Single-Step Newton-Raphson), garantindo conservação de massa e eletroneutralidade no WebApp.

---

## 🛠️ Roadmap de Escalabilidade e Computação de Alto Desempenho (HPC)

Embora este framework sirva como uma *Prova de Conceito Avançada (PoC)* robusta, sua implantação atual é limitada por restrições de hardware locais e de nuvem gratuita (ex: ciclos semanais particionados de 12 horas/45 horas em GPUs do Kaggle). Esse gargalo computacional impacta diretamente a velocidade de convergência da camada de loss de Sobolev, atualmente em uma escala exploratória de ~2.45.

Para a transição desta arquitetura para uma *ferramenta de grau comercial pronta para produção* ou um *framework de pesquisa de doutorado em larga escala*, o pipeline é desenhado de forma modular para escalar através dos seguintes vetores:

### 1. Escalonamento de Dimensionalidade (2D para 3D)
* *A Física:* Transição de campos espaciais 2D para uma grade cartesiana/esférica 3D completa para capturar com precisão a entropia conformacional exata, efeitos de volume excluído e o rastreamento eletrostático assimétrico e multidirecional de macromoléculas biológicas do mundo real.
* *A Arquitetura:* A espinha dorsal FNO/PINO é nativamente construída para se adaptar a camadas convolucionais e espectrais 3D. O aumento de escala exige apenas ajustar os tensores de grade e gerar amostras de *Ground Truth* de Teoria de Campo Autocoerente (SCFT) em 3D.

### 2. Escalonamento Computacional e de Treinamento
* *Dinâmica de Treinamento Sem Limites:* Treinamento contínuo, não particionado e multi-GPU (ex: clusters NVIDIA A100/H100) para executar as ~200k épocas necessárias. Isso permitirá uma dinâmica de aprendizado suave para o otimizador Adam sem interrupções de partida a frio (cold-start), esmagando o alvo da loss de Sobolev para limites numéricos estritos ($\le 10^{-3}$).
* *Paralelização do Aprendizado Ativo:* Escalonar o loop automatizado de *fallback* do SCFT em *workers* paralelos de HPC, permitindo que o modelo consulte soluções de alta fidelidade para modos de falha de treinamento em tempo real.

### 3. Implantação Industrial de Triagem de Alto Rendimento (HTS)
Com um modelo 3D totalmente convergido, a atual arquitetura de processamento em lote (uploads .json/.csv) atuará como um *filtro preditivo ultra-rápido para sistemas de entrega de fármacos*:
* *Fase 1 (PINO):* Triagem de mais de 100.000 candidatos customizados de distribuição de carga em segundos, descartando arranjos termodinamicamente instáveis.
* *Fase 2 (Solver HPC):* Direciona apenas o top 1% dos candidatos ótimos para simulações pesadas e computacionalmente caras de Dinâmica Molecular ou Coarse-Grained, acelerando o cronograma de P&D em ordens de magnitude.

---

## Roadmap e Status de Execução (To-Do List)

Abaixo estão os passos práticos detalhados na nossa "Pipeline PINO V3", a serem executados gradativamente:

- [x] **Fase 1: Amostragem e Geração de Espaço Contínuo**
  - Implementado o script de `Gaussian Random Fields` cujos parâmetros estruturais e físicos foram sorteados rigorosamente via **Latin Hypercube Sampling (LHS)**. O dataset original foi expandido offline para 40.000 amostras via Data Augmentation (simetria D4).
- [x] **Fase 2: Arquitetura Base da Rede Neural (PINO)**
  - Implementada a arquitetura expandida multicanal (Input de 8 camadas) para absorver o GRF, Debye e as malhas numéricas absolutas ($x, z$).
- [x] **Fase 3: Estruturação da Loss de Sobolev**
  - Esqueleto inicial de retropropagação e extração do Jacobiano configurado usando `torch.autograd`.
- [x] **Fase 4: Warm-Up Ground Truth (SCFT Real)**
  - Rodar o Solver tradicional de SCFT nos 5.000 Campos GRFs gerados para extrair a Densidade Polimérica exata $\phi_{scft}$.
- [x] **Fase 5: Treinamento Base (Warm-Up Model)**
  - Treinar o PINO nas 5.000 amostras ancoradas focando no Erro Quadrático Médio e na Função de Perda de Sobolev por 1.000 épocas.
- [x] **Fase 6: Deploy no WebApp V3**
  - Desenvolver a interface final com as novas predições ativas, controle de sequências de carga via Fourier e campos paramétricos customizados.
- [x] **Fase 7/Extensão: Treinamento de Cargas de Polímero (Diblock e Alternado)**
  - Criação de proxies físicos (perturbações) para simular o comportamento de polímeros Diblock (C1) e Alternados (C4), e realizar o fine-tuning da rede (25 épocas).
- [x] **Fase 8/Extensão: Slider de Intensidade de Carga do Polímero**
  - Refatorar o Frontend e o Dataset para treinar a PINO em valores contínuos de intensidade de carga (ex: 0.0 a 2.0), em vez de apenas um seletor booleano.
- [X] **Fase 9: O Filtro Físico de Newton-Raphson (Pós-Inferência)**
  - Construir o acoplamento do modelo à correção analítica clássica em tempo real para melhor precisão.
- [/] **Fase 10: Aprendizado da Física (Collocation Points)**
  - Gerar 50.000 cenários físicos "vazios" (sem resolução no SCFT) e injetar as Equações Diferenciais na Loss para treinar a aderência à termodinâmica do sistema.
- [ ] **Fase 11: Active Learning e Amostragem Adaptativa**
  - Criar um loop que inspeciona a variância e a falha das predições, rodando o SCFT de volta para geometrias de falha sistêmica.
- [ ] **Fase 12: High-Throughput Screening (HTS) no WebApp**
  - Adicionar um botão de upload de CSV/JSON no WebApp para computar milhares de configurações num único Batch Forward Pass da PINO, retornando relatórios estatísticos da predição instantaneamente.

---
---

<div align="center">
  <a href="#-polyfields-pino-real-time-polymer-density-predictor-for-asymmetric-multipole-potentials-using-physics-informed-neural-operators">🇧🇷 Clique aqui para a Versão em Português</a>
</div>

# 🇺🇸 PolyFields-PINO - English Version

> ⚠️ **Status: Work in Progress (WIP)**
> The final training of the pipeline is temporarily paused due to strict GPU hardware limitations. The repository and scripts are being made available so the scientific community can test, validate the physics, and collaboratively evolve the architecture.

## 🚀 How to Use (Installation and Execution)

1. **Install Dependencies:**
   The inference backend requires standard Machine Learning and Web API libraries.
   ```bash
   pip install -r requirements.txt
   ```

2. **Download the Model Weights:**
   The trained `.pth` file is hosted in the **Releases** section of this repository (on the right sidebar of the GitHub page).
   * Download `pino_v3_phase5_physics.pth` from the `v1.0-alpha` release.
   * Place it inside the `weights/` folder at the root of the project (create the folder if it doesn't exist).

3. **Launch the WebApp (Interactive Sandbox):**
   To explore polymer density predictions in real-time, use the automated startup script:
   ```bash
   bash iniciar_webapp.sh
   ```
   *(This will initialize the FastAPI server on the backend and automatically open the `index.html` viewer in your browser).*

---

## Project Objective

The **PolyFields-PINO** is the latest version of the polymeric density prediction architecture. Born as a classic Fourier Neural Operator (FNO) in previous versions, the main goal of the project is to shift **the purely statistical mapping paradigm (data-to-data) used previously, towards deep learning guided by the fundamental laws of statistical mechanics**.

The theoretical and engineering foundation of this version introduces the **Physics-Informed Neural Operator (PINO)**, aiming to create a model capable of generalizing confinement and electrostatic interaction scenarios on polyelectrolytes without exclusively relying on exhaustive Self-Consistent Field Theory (SCFT) simulations.

The main theoretical points supporting this objective are:
1. **Continuous Charge Representation**: Use of Fourier coefficients to model polymer charge sequences, compressing a very large combinatorial space into a reduced and interpretable frequency space.
2. **Infinite Topological Diversity**: Replacement of rigid anchoring scenarios with *Gaussian Random Fields (GRFs)* with parameters to simulate the most varied confinement topologies and potentials. The parametric space dictating the topology of these fields and the thermodynamic properties of the polymer are sampled using **Latin Hypercube Sampling (LHS)**, ensuring good statistical coverage.
3. **Exploitation of Physical Symmetry (Data Augmentation D4)**: Because the physics of confinement matrices and the polymeric response ($\phi$) are spatially isotropic and invariant to rotations and mirroring, the D4 Symmetry Group (90º, 180º, 270º rotations and reflections) is applied offline to exponentially multiply the SCFT *Ground Truth* at no computational cost, mitigating *overfitting*, increasing the training phase space, and avoiding paradoxical responses like training artifacts.
4. **Sobolev Training**: Injection of model derivatives into the Cost Function (*Loss*). The network not only learns the density $\phi(\mathbf{r})$, but also its exact thermal susceptibility $\left( \frac{\delta \phi}{\delta V} \right)$, increasing the informational value of each inference.
5. **Physical Residuals & Active Learning**: Use of tens of thousands of unlabeled collocation points, where the PINO is mathematically punished if it violates the Edwards and Debye-Hückel differential equations. If physics fails and uncertainty rises, an Active Learning cycle calls the SCFT on demand to correct the failure.
6. **Enhanced Precision in Front-end**: Since PINO predicts the result in 0.05 seconds with extreme precision, its inference will serve as the initial guess for a single-step solver (Single-Step Newton-Raphson), ensuring mass conservation and electroneutrality in the WebApp.

---

## 🛠️ Scalability & High-Performance Computing (HPC) Roadmap

While this framework serves as a robust *Advanced Proof of Concept (PoC)*, its current deployment is bound by local and free cloud hardware constraints (e.g., partitioned 12-hour/45-hour weekly cycles on Kaggle GPUs). This compute bottleneck directly impacts the convergence velocity of the Sobolev loss layer, currently sitting at an exploratory scale of ~2.45.

To transition this architecture into a *production-ready, commercial-grade tool* or a *full-scale PhD research framework*, the pipeline is modularly designed to scale across the following vectors:

### 1. Dimensionality Scaling (2D to 3D)
* *The Physics:* Transitioning from 2D spatial fields to a full 3D Cartesian/Spherical grid to accurately capture the exact conformational entropy, excluded volume effects, and asymmetric multi-directional electrostatic screening of real-world biological macromolecules.
* *The Architecture:* The FNO/PINO backbone is natively built to adapt to 3D convolutional and spectral layers. Scaling up simply requires adjusting the grid tensors and generating 3D Self-Consistent Field Theory (SCFT) ground truth samples.

### 2. Computational & Training Scaling
* *Unbounded Training Dynamics:* Continuous, unpartitioned multi-GPU training (e.g., NVIDIA A100/H100 clusters) to execute the required ~200k epochs. This will enable smooth learning dynamics for the Adam optimizer without cold-start interruptions, crushing the Sobolev loss target to strict numerical bounds ($\le 10^{-3}$).
* *Active Learning parallelization:* Scaling the automated SCFT-fallback loop into parallel HPC workers, allowing the model to query high-fidelity solutions for training failure modes in real-time.

### 3. Industrial High-Throughput Screening (HTS) Deployment
With a fully converged 3D model, the current batch processing architecture (.json/.csv uploads) will act as an ultra-fast *predictive filter for drug delivery systems*:
* *Phase 1 (PINO):* Screens 100,000+ custom charge distribution candidates in seconds, discarding thermodynamically unstable layouts.
* *Phase 2 (HPC Solver):* Routes only the top 1% of optimal candidates to heavy, computationally expensive Coarse-Grained or Molecular Dynamics simulations, accelerating the R&D timeline by orders of magnitude.

---

## Roadmap and Execution Status (To-Do List)

Below are the practical steps detailed in our "Pipeline PINO V3", to be executed gradually:

- [x] **Phase 1: Sampling and Continuous Space Generation**
  - Implemented the `Gaussian Random Fields` script whose structural and physical parameters were rigorously drawn via **Latin Hypercube Sampling (LHS)**. The original dataset was expanded offline to 40,000 samples via Data Augmentation (D4 symmetry).
- [x] **Phase 2: Base Neural Network Architecture (PINO)**
  - Implemented the multi-channel expanded architecture (8-layer Input) to absorb the GRF, Debye, and absolute numerical meshes ($x, z$).
- [x] **Phase 3: Sobolev Loss Structuring**
  - Initial backpropagation and Jacobian extraction skeleton configured using `torch.autograd`.
- [x] **Phase 4: Warm-Up Ground Truth (Real SCFT)**
  - Run the traditional SCFT Solver on the 5,000 generated GRF Fields to extract the exact Polymer Density $\phi_{scft}$.
- [x] **Phase 5: Base Training (Warm-Up Model)**
  - Train the PINO on the 5,000 anchored samples focusing on the Mean Squared Error and the Sobolev Loss Function for 1,000 epochs.
- [x] **Phase 6: WebApp V3 Deploy**
  - Develop the final interface with the new active predictions, charge sequence control via Fourier, and customized parametric fields.
- [x] **Phase 7/Extension: Polymer Charge Training (Diblock and Alternating)**
  - Creation of physical proxies (perturbations) to simulate the behavior of Diblock (C1) and Alternating (C4) polymers, and perform network fine-tuning (25 epochs).
- [x] **Phase 8/Extension: Polymer Charge Intensity Slider**
  - Refactor the Frontend and Dataset to train PINO on continuous charge intensity values (e.g., 0.0 to 2.0), instead of just a boolean selector.
- [X] **Phase 9: Newton-Raphson Physical Filter (Post-Inference)**
  - Build the coupling of the model to the classical analytical correction in real time for better precision.
- [/] **Phase 10: Physics Learning (Collocation Points)**
  - Generate 50,000 "empty" physical scenarios (without SCFT resolution) and inject Differential Equations into the Loss to train adherence to system thermodynamics.
- [ ] **Phase 11: Active Learning and Adaptive Sampling**
  - Create a loop that inspects prediction variance and failure, running the SCFT back for geometries with systemic failure.
- [ ] **Phase 12: High-Throughput Screening (HTS) in the WebApp**
  - Add a CSV/JSON upload button in the WebApp to compute thousands of numerical configurations in a single PINO Batch Forward Pass, instantly returning prediction statistical reports.

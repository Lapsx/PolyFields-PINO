# Memória do Projeto: PINO Polymer Webmap

Este arquivo serve como memória persistente para qualquer agente de IA que for instanciado nesta pasta. Leia com atenção para entender o estado atual do sistema.

## O Que é o Projeto?
Uma aplicação full-stack de Física de Polímeros. O backend utiliza um modelo PINO (Physics-Informed Neural Operator) baseado na arquitetura FNO 2D para simular a separação de fase e acoplamento eletrostático de polímeros (neutros, diblocos, e alternados) submetidos a campos elétricos arbitrários gerados interativamente pelo usuário no navegador.

## Pipeline de Treino Atual (Julho/2026)
O modelo foi treinado em 5 Fases progressivas. Os pesos estão na pasta `weights/`:
* **Fase 1 e 2:** Treino Base Data-Driven (MSE em simulações SCFT puras).
* **Fase 3 e 4:** Data-Driven + Perturbações (Cargas Contínuas, Dibloco, etc). Arquivo atual: `pino_v3_phase4_final.pth`.
* **Fase 5 (Atual):** Physics-Informed (Collocation Points). O modelo aprende a minimizar o resíduo da Equação de Edwards / Ginzburg-Landau (Física de Fluidos Termodinâmicos) para garantir a deformação fluida e a separação de blocos nas cargas. Arquivo atual: `pino_v3_phase5_physics.pth`. O script `pino_train_physics.py` foi modificado para retomar o treino de onde parou automaticamente caso este peso já exista.

## Backend (FastAPI - main.py)
* O backend roda em `main.py`. Ele prioriza o carregamento do peso da **Fase 5**.
* O backend aplica pós-processamento analítico (Filtro Newton-Raphson) na saída da rede para garantir Conservação de Massa ESTRITA (ajuste no Potencial Químico).
* A função `compute_density` retorna uma tupla: `(density, drho_dmu, num_iter)`.
* As rotas de experimento (`/experiment`) geram diagramas de fase via varredura (ex: Transição Coil-Globule variando o parâmetro estérico `u` de Flory-Huggins).

## Frontend (Vanilla JS / index.html)
* A interface usa Canvas 2D. O usuário pode colocar cargas (dipolos).
* Os sliders controlam `u` (Repulsão do Solvente), `b` (Comprimento de Kuhn), `kappa` (Salinidade), e a Carga Interna (0 = Neutro, 1 = Dibloco, 4 = Alternado).
* A aba HTS (High-Throughput Screening) está presente no HTML mas está oculta (`display: none`) aguardando a finalização dos testes da Fase 5 para ser religada.

## Descobertas Recentes (Julho/2026)
* **O Mistério do Slider `u` (Flory-Huggins):** Descobrimos por que o polímero não mudava de forma (Coil-Globule) ao mexer no slider nas Fases 1 a 4. O dataset "Ground Truth" gerado em `scft_solver.py` era um Mock baseado apenas em `torch.exp(-V * 2.0)`. Isso significa que os parâmetros físicos (`u`, `b`, `kappa`) foram sorteados no `grf_generator.py` e entregues à FNO, mas o resultado `phi` não tinha nenhuma dependência matemática deles. Como resultado, o treino Data-Driven ensinou a rede a zerar e ignorar o canal do parâmetro `u`.
* **A Importância da Fase 5:** Como os dados iniciais eram ignorantes à física de fluidos termodinâmicos, a Fase 5 (Loss PDE de Ginzburg-Landau/Edwards) não é apenas um "refinamento", é a única etapa que efetivamente ensinará a rede a contrair o polímero num glóbulo quando $u > 0.2$ e deformá-lo sob tensão elétrica.
* **Migração para o Kaggle:** O script `pino_train_physics.py` foi atualizado para **5000 épocas** (LR = 1e-5, Scheduler = 1000) e empacotado no `kaggle_phase5.zip`. O script possui salvamento contínuo (Checkpoints a cada 200 épocas e no arquivo principal) para contornar o limite de tempo de sessão do Kaggle, permitindo que o usuário retome o treino facilmente.
* **Aceleração Multi-GPU:** Adicionamos suporte ao `nn.DataParallel` com escalonamento dinâmico de Batch Size (ex: 64 amostras para 2x GPUs T4). Contornamos um bug do PyTorch com tensores complexos em DataParallel armazenando os pesos de Fourier como reais (`torch.float`) e convertendo sob demanda no forward pass em `pino_architecture.py`.
* **Ajuste Fino no Treinamento Kaggle:** Para acelerar a queda da loss no Kaggle, o `base_batch_size` em `pino_train_physics.py` foi reduzido de 16 para 8 e o Learning Rate (LR) foi aumentado de `1e-5` para `5e-5`. O `kaggle_phase5.zip` já está atualizado.
* **Física e Visualização no Frontend:** Esclarecemos que o círculo fixo no centro do canvas de entrada representa uma **Nanopartícula Sólida (obstáculo)**, enquanto o polímero em si é a nuvem do gráfico de **Densidade φ**. Como o modelo garante conservação de massa (A densidade global tem integral = 1.0), os valores da "Densidade" atuam como um **Fator de Concentração Local** (ex: valor 1000 = a massa colapsou e ali há 1000x mais polímero do que uma distribuição uniforme). Tooltips explicativas foram adicionadas no arquivo `index.html`.
* **Cargas Internas e Adsorção:** A restrição geométrica que ejetava cargas de dentro da nanopartícula no frontend (`app_v4.js`) foi removida, permitindo alocar cargas locais no centro para simular interações diretas (como proteínas globulares). Implementamos uma análise de transição **Adsorvido/Desorvido vs Salinidade** via API. Descobrimos empiricamente que a "Distância do Centro de Massa" é inútil como métrica para isso, pois o sistema tem simetria radial e o COM sempre fica em `(0,0)`. A solução correta é plotar o **Raio de Giração ($R_g$)**, que cai para $\sim 1.0$ (raio da partícula) na adsorção e sobe na desorção.
* **A Cegueira da Normalização de Potencial:** Tentamos criar um experimento de "Ponto Isoelétrico / Adsorção no Lado Errado" varrendo a carga central de -10 a +10. Falhou porque descobrimos que o pré-processamento do backend aplica um `Min-Max Scaler` no potencial elétrico mapeando-o para `[-1.0, 1.0]` *por amostra*. Como resultado, qualquer carga isolada tem sua magnitude absoluta completamente apagada e padronizada (Q=1 e Q=100 geram a mesma imagem). Isso confirmou que varreduras contínuas da magnitude de cargas isoladas são incompatíveis com o formato atual de tensores do modelo V3.

## Próximos Passos (To-Do)
* Executar as sessões no Kaggle utilizando o sistema de "resume" a partir dos checkpoints até completar as 5000 épocas.
* Baixar o peso final `pino_v3_phase5_physics.pth` e colocar na pasta `weights/`.
* Testar no WebApp (`index.html`) se o polímero finalmente obedece o slider de Flory-Huggins (Aglomera como um círculo quando repulsivo e espalha quando livre) graças à minimização do resíduo PDE.
* Quando a Física estiver perfeita e validada no canvas, religar a funcionalidade HTS para testar lotes de polímeros via CSV/JSON.

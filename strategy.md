# Estratégia do Projeto - Classificação de Patologias em Folhas (Plant Pathology 2020)
### Registro de Decisões Arquiteturais e Planejamento Experimental

Este documento consolida as **decisões técnicas, justificativas arquiteturais e governança experimental** adotadas no desenvolvimento do projeto de Redes Neurais Artificiais (UFERSA), estruturado em 4 etapas ativas de execução e um documento dedicado a propostas futuras.

---

## 1. Estrutura de Execução do Projeto

| Etapa | Arquivo / Módulo | Objetivo Estratégico Principal |
| :---: | :--- | :--- |
| **01** | `01_eda.ipynb` | Análise exploratória visual, canais de cor, balanceamento e inspeção morfológica de filtros OpenCV |
| **02** | `02_pipeline_single.ipynb` <br> `02_pipeline_improved.ipynb` | PoC e Estudos de Ablação: Baseline Natural (sem filtros) vs. Realce Adaptativo CLAHE (Canal V do HSV) |
| **03** | `03_benchmark_models.ipynb` | Benchmarking empírico e sistemático dos 5 backbones selecionados (`DenseNet121`, `EfficientNetV2B0`, `ResNet50V2`, `ConvNeXtTiny`, `Xception`) |
| **04** | `04_xai_gradcam.ipynb` | Explicabilidade visual (XAI) com Grad-CAM / Grad-CAM++ sobre os modelos campeões |

> **Nota sobre Trabalhos Futuros:** As especificações formais de busca de hiperparâmetros (*Grid Search*) e combinação de modelos (*Ensemble*) foram centralizadas no arquivo [propostas_futuras.md](propostas_futuras.md), localizado ao final da documentação.

---

## 2. Decisões Estratégicas por Etapa

### Etapa 01: Análise Exploratória e Diagnóstico de Desbalanceamento (`01_eda.ipynb`)
- **Diagnóstico de Raridade:** A classe `multiple_diseases` representa apenas ~5% das amostras (91 de 1.821). Definiu-se que nenhuma divisão de dados pode ser aleatória pura; o uso de `train_test_split(..., stratify=y)` e `class_weight` balanceado é mandatório.
- **Triagem de Filtros OpenCV:** Filtros monocromáticos como `cv2.Canny` são restritos à inspeção morfológica na EDA. Para o pipeline de Deep Learning com pesos do ImageNet, qualquer transformação deve obrigatoriamente preservar 3 canais de cor.

### Etapa 02: Baseline Puro vs. Estudo de Ablação CLAHE
Para isolar com precisão o ganho de técnicas clássicas de visão computacional, a Etapa 2 foi dividida em dois cadernos com regime idêntico de hiperparâmetros:

1. **`02_pipeline_single.ipynb` (Baseline Natural - PoC):**
   - **Zero Pré-processamento OpenCV:** Imagens lidas em RGB e redimensionadas diretamente para $512 \times 512$, com normalização padrão do ImageNet aplicada internamente pelo modelo.
   - Serve como controle para verificar a capacidade de generalização da `DenseNet121` sem interferência de filtros morfológicos.
2. **`02_pipeline_improved.ipynb` (Ablação V1 - Realce CLAHE):**
   - **Realce no Canal V do HSV:** Aplica-se o CLAHE (*Contrast Limited Adaptive Histogram Equalization*) exclusivamente no canal de luminância ($V$) do espaço HSV, convertendo de volta para RGB tridimensional.
   - **Justificativa Estratégica:** Realça bordas e relevo de pústulas de ferrugem e lesões necróticas de sarna sem distorcer as coordenadas cromáticas ($H$ e $S$) da folha.
3. **Eficiência Computacional Compartilhada:**
   - **Precisão Mista (`mixed_float16`):** Redução do uso de VRAM pela metade e aceleração de até 3x na GPU Tesla T4 (Google Colab).
   - **Caching em RAM (`.cache()`):** O carregamento ou pré-processamento roda uma única vez na 1ª época. A partir da 2ª época, os tensores residem na RAM, garantindo 100% de saturação da GPU.

### Etapa 03: Benchmarking de Arquiteturas Pré-Treinadas (`03_benchmark_models.ipynb`)

#### 3.1 Matriz de Triagem Arquitetural (Critérios de Seleção e Descarte)
Dentre as famílias de redes neurais do Keras Applications, avaliou-se uma representante por família para justificar a triagem:

| Família | Versão | Decisão | Justificativa Arquitetural |
| :--- | :--- | :---: | :--- |
| **DenseNet** | `DenseNet121` | **Executada (PoC)** | Conexões densas concatenam mapas de todas as camadas. Baixo custo de parâmetros (~7M) e alto reuso de feições. Serve como backbone de referência. |
| **EfficientNetV2** | `EfficientNetV2B0` | **Selecionada** | Evolução da V1 com Fused-MBConv acelerado na GPU e blocos de atenção de canal (Squeeze-and-Excitation). |
| **ResNet** | `ResNet50V2` | **Selecionada** | Padrão-ouro em skip connections residuais com pré-ativação (fluxo direto de gradientes sem atenuação). |
| **ConvNeXt** | `ConvNeXtTiny` | **Selecionada** | Estado da arte convolucional (2022). Emula blocos de Vision Transformers com kernels largos $7 \times 7$ e ativação GELU. |
| **Xception** | `Xception` | **Selecionada** | Desacoplamento explícito de correlações espaciais e de canal via convoluções separáveis em profundidade. |

#### 3.2 Protocolo Experimental Unificado (Condições e Pesos)
Para garantir que a comparação entre os 5 modelos seja estritamente justa e não enviesada por hiperparâmetros diferentes, fixou-se:

- **Pesos Iniciais:** `weights='imagenet'` para todos os backbones (Fine-Tuning com base descongelada).
- **Pesos de Classe Balanceados (`class_weight`):** Calculados via Scikit-Learn sobre as 1.456 amostras de treino:
  - Classe 0 (`healthy` - 413 amostras): **`0.8814`**
  - Classe 1 (`multiple_diseases` - 73 amostras): **`4.9863`** (penalização ~5.6x maior em caso de erro)
  - Classe 2 (`rust` - 497 amostras): **`0.7324`**
  - Classe 3 (`scab` - 473 amostras): **`0.7696`**
- **Hiperparâmetros:** Resolução `512x512`, `Batch Size = 16`, otimizador `Adam(learning_rate=1e-4)`, política `mixed_float16`, topo com `GAP + Dropout(0.3) + Dense(4, softmax, float32)`.
- **Mecanismos de Parada e Ajuste:**
  - `EarlyStopping(monitor='val_loss', patience=8, restore_best_weights=True)`
  - `ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=3, min_lr=1e-6)`
  - `ModelCheckpoint(filepath='models/best_{model_name}.keras', monitor='val_roc_auc', mode='max', save_best_only=True)`

### Etapa 04: Explicabilidade Visual (XAI) com Grad-CAM (`04_xai_gradcam.ipynb`)
- A etapa de explicabilidade substitui a execução do ensemble no fluxo principal da avaliação acadêmica, pois auditar visualmente a biologia das decisões convolucionais possui prioridade científica superior à fusão cega de probabilidades.
- **Protocolo de Validação:**
  1. Extração de gradientes da classe predita em relação à última camada convolucional do modelo campeão individual.
  2. Ponderação ReLU dos mapas de ativação (*Heatmaps*).
  3. Sobreposição em alta resolução para certificar que a rede ativa sobre as lesões foliares (manchas e pústulas) e não sobre solo, dedos ou artefatos de fundo.

---

## 3. Diretrizes de Padronização e Exportação de Artefatos

Para assegurar reprodutibilidade total, o pipeline salvará estruturadamente:

1. **Modelos e Pesos Compilados (`models/`):**
   - `models/best_densenet121_baseline.keras` (Modelo campeão do baseline)
   - `models/best_densenet121_clahe.keras` (Modelo com pré-processamento CLAHE)
   - `models/best_{model_name}.keras` (Checkpoints salvos pelo benchmarking)
2. **Históricos e Relatórios Tabulares (`models/`):**
   - `models/history_{model_name}.csv` (Evolução época a época de loss, acurácia, ROC-AUC e LR)
   - `models/classification_report_{model_name}.csv` (Precisão, Recall e F1 por classe)
   - `models/models_benchmark_comparison.csv` (Tabela comparativa consolidada do benchmarking)
3. **Gráficos Científicos a 300 DPI (`graphs/`):**
   - `graphs/loss_curve_{model_name}.png` (Curva de perda treino vs. validação)
   - `graphs/accuracy_auc_curve_{model_name}.png` (Curvas conjuntas de acurácia e ROC-AUC)
   - `graphs/confusion_matrix_{model_name}.png` (Matriz de confusão absoluta e normalizada)
   - `graphs/roc_curve_multiclass_{model_name}.png` (Curvas ROC multiclasse One-vs-Rest)
   - `graphs/models_benchmark_comparison.png` (Gráfico comparativo geral de backbones)

---

## 4. Propostas Futuras e Referências Teóricas - [propostas_futuras.md](propostas_futuras.md)

As etapas não executadas empiricamente devido ao limite de tempo e cotas de GPU no Google Colab estão integralmente documentadas no arquivo independente **[propostas_futuras.md](propostas_futuras.md)**:

1. **Busca em Grade (*Grid Search*):** Modelagem teórica do espaço cartesiano de 648 combinações de hiperparâmetros e justificativa metodológica de por que a busca exaustiva foi substituída pelo benchmarking direcionado de backbones.
2. **Ensemble de Modelos:** Teorema de redução de variância, formulações matemáticas de fusão (*Soft Voting*, *Weighted Blending* por ROC-AUC, SLSQP e *Stacking*) e pipeline completo de inferência e submissão para `data/submission.csv`.
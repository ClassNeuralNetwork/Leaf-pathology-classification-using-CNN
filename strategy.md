# Estratégia do Projeto - Classificação de Patologias em Folhas (Plant Pathology 2020)
### Registro de Decisões Arquiteturais e Planejamento Experimental

Este documento consolida as **decisões técnicas, justificativas arquiteturais e governança experimental** adotadas no desenvolvimento do projeto de Redes Neurais Artificiais (UFERSA), estruturado em 3 etapas ativas de execução e um documento dedicado a propostas futuras. Cada integrante da dupla desenvolveu uma arquitetura: `DenseNet121` (Etapa 02) e `ResNet50` (Etapa 03).

---

## 1. Estrutura de Execução do Projeto

| Etapa | Arquivo / Módulo | Objetivo Estratégico Principal |
| :---: | :--- | :--- |
| **01** | `01_eda.ipynb` | Análise exploratória visual, canais de cor, balanceamento e inspeção morfológica de filtros OpenCV |
| **02** | `02_pipeline_single.ipynb` <br> `02_pipeline_improved.ipynb` | `DenseNet121` com Estudo de Ablação: Baseline Natural (sem filtros) vs. Realce Adaptativo CLAHE (Canal V do HSV) |
| **03** | `ResNet50.ipynb` | `ResNet50` com treinamento em duas fases, avaliação em conjunto de teste isolado e explicabilidade visual (XAI) com SHAP |

> **Nota sobre Trabalhos Futuros:** As especificações formais de busca de hiperparâmetros (*Grid Search*) e combinação de modelos (*Ensemble*) foram centralizadas no arquivo [propostas_futuras.md](propostas_futuras.md), localizado ao final da documentação.

### 1.1 Seleção de Arquiteturas

| Família | Versão | Etapa | Justificativa Arquitetural |
| :--- | :--- | :---: | :--- |
| **DenseNet** | `DenseNet121` | **02** | Conexões densas concatenam mapas de todas as camadas. Baixo custo de parâmetros (~7M) e alto reuso de feições. |
| **ResNet** | `ResNet50` | **03** | Padrão-ouro em skip connections residuais ($x + \mathcal{F}(x)$), que permitem treinar redes profundas sem degradação do gradiente. |

---

## 2. Decisões Estratégicas por Etapa

### Etapa 01: Análise Exploratória e Diagnóstico de Desbalanceamento (`01_eda.ipynb`)
- **Diagnóstico de Raridade:** A classe `multiple_diseases` representa apenas ~5% das amostras (91 de 1.821). Definiu-se que nenhuma divisão de dados pode ser aleatória pura; o uso de `train_test_split(..., stratify=y)` e `class_weight` balanceado é mandatório.
- **Triagem de Filtros OpenCV:** Filtros monocromáticos como `cv2.Canny` são restritos à inspeção morfológica na EDA. Para o pipeline de Deep Learning com pesos do ImageNet, qualquer transformação deve obrigatoriamente preservar 3 canais de cor.

### Etapa 02: DenseNet121 — Baseline Puro vs. Estudo de Ablação CLAHE
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

#### 2.1 Protocolo Experimental Unificado (Condições e Pesos)
Para garantir que a comparação entre o baseline e a versão CLAHE seja estritamente justa e não enviesada por hiperparâmetros diferentes, fixou-se:

- **Divisão dos Dados:** `train_test_split(..., test_size=0.20, stratify=y)`, com 1.456 amostras de treino e 365 de validação.
- **Pesos Iniciais:** `weights='imagenet'` (Fine-Tuning com base descongelada).
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

### Etapa 03: ResNet50 com Fine-Tuning em Duas Fases e SHAP (`ResNet50.ipynb`)

#### 3.1 Divisão dos Dados e Desbalanceamento
- **Treino / Validação / Teste (70/15/15):** `train_test_split` estratificado em duas etapas (1.274 / 273 / 274 amostras). A validação orienta os callbacks e o conjunto de teste é utilizado uma única vez, na avaliação final, evitando o viés otimista de selecionar e avaliar o modelo sobre o mesmo conjunto.
- **Justificativa do Split Manual:** O `validation_split` do `model.fit` não é compatível com `tf.data.Dataset` e, mesmo com arrays, separa os últimos exemplos sem embaralhamento nem estratificação.
- **Pesos de Classe Balanceados (`class_weight`):** Calculados via Scikit-Learn sobre as 1.274 amostras de treino:
  - Classe 0 (`healthy` - 361 amostras): **`0.882`**
  - Classe 1 (`multiple_diseases` - 64 amostras): **`4.977`**
  - Classe 2 (`rust` - 435 amostras): **`0.732`**
  - Classe 3 (`scab` - 414 amostras): **`0.769`**
- **Métricas Principais:** *Macro F1* e recall de `multiple_diseases`, pois a acurácia global mascara o desempenho na classe minoritária.

#### 3.2 Pipeline de Dados
- **Resolução e Normalização:** $512 \times 512$ com `preprocess_input` da ResNet50 (BGR com subtração da média de cada canal da ImageNet).
- **Data Augmentation Apenas no Treino:** `RandomFlip("horizontal_and_vertical")`, `RandomRotation(0.1)`, `RandomZoom(0.1)` e `RandomBrightness(0.1)`, aplicados no pipeline `tf.data` antes da normalização. Validação e teste recebem apenas imagens limpas, garantindo métricas determinísticas e reprodutíveis.

#### 3.3 Treinamento em Duas Fases
- **Topo de Classificação:** `GAP + Dense(256, ReLU) + Dropout(0.4) + Dense(4, softmax)`.
- **Fase 1 (Base Congelada):** Apenas o topo é treinado com `Adam(learning_rate=1e-3)`, evitando que gradientes de camadas recém-inicializadas degradem os pesos pré-treinados.
- **Fase 2 (*Fine-Tuning*):** Descongelamento do bloco `conv5` com `Adam(learning_rate=1e-5)`. As camadas de `BatchNormalization` permanecem congeladas e a base é chamada com `training=False`, preservando as estatísticas da ImageNet diante de lotes pequenos (`Batch Size = 16`).
- **Mecanismos de Parada e Ajuste (nas duas fases):**
  - `EarlyStopping(monitor='val_loss', patience=7 / 6, restore_best_weights=True)`
  - `ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=3, min_lr=1e-7)`
  - `ModelCheckpoint(monitor='val_loss', mode='min', save_best_only=True, save_weights_only=True)`
- **Justificativa do Monitoramento:** A `val_loss` é mais estável que o F1 de validação, que oscila com as poucas amostras de `multiple_diseases` (14 na validação).

#### 3.4 Explicabilidade Visual (XAI) com SHAP
- **Método:** `shap.GradientExplainer`, que estima a contribuição de cada pixel para a previsão combinando gradientes da rede com um conjunto de referência (*background*) de 20 imagens do treino.
- **Protocolo de Validação:**
  1. Seleção de um acerto de cada classe e de até dois erros do conjunto de teste, priorizando `multiple_diseases`.
  2. Geração dos mapas de contribuição para as 4 classes de cada imagem, evidenciando as regiões que levam o modelo a hesitar entre classes.
  3. Verificação de que as contribuições positivas se concentram nas lesões foliares (manchas e pústulas) e não sobre solo, dedos ou artefatos de fundo.

---

## 3. Diretrizes de Padronização e Exportação de Artefatos

Para assegurar reprodutibilidade, os pipelines salvam estruturadamente:

1. **Modelos e Pesos Compilados:** Salvos no Google Drive durante a execução e não versionados no repositório (`*.keras` e `*.h5` constam no `.gitignore`).
2. **Históricos e Relatórios Tabulares (`models/`):**
   - `models/baseline_densenet121_history.csv` e `models/training_log_clahe.csv` (Evolução época a época de loss, acurácia, ROC-AUC e LR da `DenseNet121`)
   - `models/baseline_densenet121_classification_report.csv` (Precisão, Recall e F1 por classe da `DenseNet121`)
   - `models/resnet50_finetune_classification_report_teste.csv` (Precisão, Recall e F1 por classe da `ResNet50` no conjunto de teste)
3. **Gráficos Científicos a 300 DPI (`graphs/`):**
   - `graphs/pipeline_single_curves.png` e `graphs/training_curves_densenet121_clahe.png` (Curvas de treinamento da `DenseNet121`)
   - `graphs/pipeline_single_confusion_matrix.png` e `graphs/confusion_matrix_densenet121_clahe.png` (Matrizes de confusão absoluta e normalizada)
   - `graphs/roc_curves_densenet121_clahe.png` (Curvas ROC multiclasse One-vs-Rest)
   - `graphs/clahe_enhancement_sample.png` (Amostras com e sem realce CLAHE)
   - `graphs/resnet50_finetune_augmentation.png` (Exemplos de Data Augmentation da `ResNet50`)
   - `graphs/resnet50_finetune_curvas.png` (Curvas de loss, acurácia, ROC-AUC e macro F1 das duas fases da `ResNet50`)
   - `graphs/resnet50_finetune_matriz_confusao_teste.png` (Matrizes de confusão absoluta e normalizada da `ResNet50` no teste)
   - `graphs/resnet50_finetune_shap_{n}_{resultado}_{classe}.png` (Mapas SHAP de acertos e erros do teste)

---

## 4. Propostas Futuras e Referências Teóricas - [propostas_futuras.md](propostas_futuras.md)

As etapas não executadas empiricamente devido ao limite de tempo e cotas de GPU no Google Colab estão integralmente documentadas no arquivo independente **[propostas_futuras.md](propostas_futuras.md)**:

1. **Busca em Grade (*Grid Search*):** Modelagem teórica do espaço cartesiano de 648 combinações de hiperparâmetros e justificativa metodológica para a não realização da busca exaustiva.
2. **Ensemble de Modelos:** Teorema de redução de variância, formulações matemáticas de fusão (*Soft Voting*, *Weighted Blending* por ROC-AUC, SLSQP e *Stacking*) e pipeline completo de inferência e submissão para `data/submission.csv`.

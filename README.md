# Classificação de Patologias em Folhas com Redes Neurais Convolucionais e Explicabilidade (SHAP)

Este projeto foi desenvolvido como parte da disciplina de **Redes Neurais Artificiais**, na Universidade Federal Rural do Semi-Árido (UFERSA).

---

## Abstract

Automated foliar disease diagnosis is vital for precision agriculture, early crop pathogen intervention, and targeted fungicide management. In apple orchards (*Malus domestica*), fungal infections such as cedar apple rust and apple scab cause devastating yield losses if left undetected, yet open-canopy imagery introduces challenging illumination variations, complex backgrounds, overlapping foliage, and co-occurring pathologies. This project presents an end-to-end deep learning framework developed on the *Plant Pathology 2020 – FGVC7* challenge dataset (CVPR 2020), classifying leaves into four mutually exclusive categories: *healthy*, *rust*, *scab*, and *multiple diseases*. 

To overcome the severe representation deficit of the co-infection class (~5% of samples), a balanced class-weighting formulation is embedded directly into the categorical cross-entropy loss without synthetic distortion. The data pipeline leverages high-resolution representations ($512 \times 512$), asynchronous `tf.data` prefetching, GPU-accelerated stochastic augmentation, and in-memory tensor caching paired with automatic mixed-precision acceleration (`mixed_float16`). Two transfer-learning architectures were developed, one by each team member: a fully fine-tuned `DenseNet121`, used in a morphological ablation study comparing raw natural baselines against HSV-space Contrast Limited Adaptive Histogram Equalization (CLAHE), and a `ResNet50` trained in two stages (frozen backbone followed by fine-tuning of the `conv5` block) and evaluated on an isolated stratified test split. Finally, model predictions are audited using Explainable Artificial Intelligence (XAI) via SHAP (`GradientExplainer`), assessing whether pixel-level attributions localize genuine botanical lesions (rust pustules and necrotic spots) rather than spurious background artifacts.

**Keywords:** Plant Pathology, Deep Learning, Convolutional Neural Networks (CNN), Explainable AI (XAI), SHAP, Class Imbalance, Precision Agriculture.

---

## Introdução e Contexto

A agricultura de precisão e o monitoramento automatizado de lavouras desempenham um papel vital na segurança alimentar global e na redução do desperdício de defensivos agrícolas. Na fruticultura, especificamente na cultura da maçã (*Malus domestica*), doenças foliares fúngicas como a **ferrugem** (*cedar apple rust*) e a **sarna** (*apple scab*) são responsáveis por perdas drásticas de produtividade se não identificadas em estágio inicial.

O desafio reside na complexidade visual das imagens em ambiente aberto: variações severas de iluminação natural, sombras, folhas sobrepostas e a ocorrência simultânea de múltiplas patologias na mesma folha tornam o diagnóstico visual humano subjetivo e demorado.

---

## Objetivo

Desenvolver e comparar modelos de Redes Neurais Convolucionais profundas capazes de diagnosticar o estado fitopatológico de folhas de macieira com alta acurácia e generalização, garantindo:

1. **Preservação de Rigor Anti-Vazamento:** Divisão estratificada dos dados prévia a qualquer transformação.
2. **Estudos de Ablação de Pré-processamento:** Avaliação comparativa entre baseline natural (`single`), realce adaptativo CLAHE no canal V do HSV (`improved`).
3. **Desenvolvimento de Arquiteturas Pré-Treinadas:** Uma arquitetura por integrante da dupla (`DenseNet121` e `ResNet50`), ambas com *Transfer Learning* a partir da ImageNet.
4. **Auditoria Visual (*XAI*):** Mapas de contribuição por pixel via **SHAP** nos notebooks das duas arquiteturas para certificar que a rede toma decisões fundamentadas nas lesões das folhas e não em ruídos de fundo.
5. **Documentação de Propostas Futuras:** Formalização metodológica de **Grid Search** e **Ensemble de Modelos** consolidada em [propostas_futuras.md](propostas_futuras.md).

---

## Tipo do Problema

O problema é de **Classificação Multiclasse** em 4 classes mutuamente exclusivas:
- `healthy` (Folha saudável)
- `multiple_diseases` (Folha com presença simultânea de múltiplas patologias)
- `rust` (Folha infectada com ferrugem)
- `scab` (Folha infectada com sarna)

A camada de saída das redes utiliza **4 neurônios com ativação Softmax**, otimizados pela função de perda **Categorical Crossentropy**.

---

## Dataset

Foi utilizado o conjunto de dados oficial da competição internacional **Plant Pathology 2020 – FGVC7** (*Fine-Grained Visual Categorization*), organizada no âmbito da conferência **CVPR 2020** e hospedada no Kaggle.

🔗 [Plant Pathology 2020 - FGVC7 (Kaggle)](https://www.kaggle.com/c/plant-pathology-2020-fgvc7)

### Distribuição Amostral:
* **Conjunto de Treino Anotado:** 1.821 imagens em alta resolução.
* **Conjunto de Teste para Submissão:** 1.821 imagens não rotuladas.

| Classe | Frequência Aproximada | Proporção (%) | Característica Visual |
| :--- | :---: | :---: | :--- |
| **rust** | ~622 | 34.2% | Pústulas e manchas amarelo-alaranjadas |
| **scab** | ~592 | 32.5% | Manchas necróticas marrom-escuras / aveludadas |
| **healthy** | ~516 | 28.3% | Superfície verde uniforme sem lesões |
| **multiple_diseases** | **~91** | **5.0%** | Coocorrência de ferrugem e sarna *(classe minoritária)* |

> **Desbalanceamento:** A extrema raridade da classe `multiple_diseases` (~5%) exige mecanismos ativos como divisão estratificada (`train_test_split(..., stratify=y)`) e pesos de classe (`class_weight`) para evitar que a rede se torne cega à classe minoritária.

---

## Metodologia e Pipeline de Dados

```
[Imagens Brutas em Alta Resolução (512x512)]
       ↓
[01. EDA: Análise Morfológica e Filtros OpenCV (GaussianBlur, Canny, CLAHE)]
       ↓
[Divisão Estratificada + Pesos de Classe Balanceados (class_weight)]
       ↓
       ├── [02. DenseNet121: Baseline Natural (02_densenet121_baseline.ipynb) vs. CLAHE (03_densenet121_clahe.ipynb)]
       │       Pipeline tf.data: Caching Estratégico em RAM + Data Augmentation em Batch + Mixed Precision (float16)
       │       Base Convolucional Fine-Tuned (trainable=True, lr=1e-4) + GAP + Dropout(0.3) + Dense(4, float32)
       │       Avaliação na Validação + Explicabilidade com SHAP (GradientExplainer)
       │
       └── [ResNet50: Treinamento em Duas Fases (04_resnet50_finetuning.ipynb)]
               Divisão Treino / Validação / Teste (70/15/15) + Data Augmentation no Treino
               Fase 1: Base Congelada (lr=1e-3) → Fase 2: Fine-Tuning do Bloco conv5 (lr=1e-5)
               Avaliação em Teste Isolado + Explicabilidade com SHAP (GradientExplainer)
       ↓
[Documentação Teórica de Propostas Futuras: Grid Search e Ensemble (propostas_futuras.md)]
```

### 1. Pré-processamento e Estudos de Ablação (DenseNet121)
* **Análise Morfológica e Filtros (na EDA):** `cv2.GaussianBlur` para atenuação de ruído e `cv2.Canny` para destacar bordas das pústulas.
* **Estudo de Ablação no Pipeline:**
  * **Baseline (`02_densenet121_baseline.ipynb`):** Imagens naturais sem filtros adicionais em alta resolução ($512 \times 512$).
  * **Realce Adaptativo CLAHE (`03_densenet121_clahe.ipynb`):** Equalização adaptativa local no canal de luminância ($V$) do espaço de cor HSV para destacar pústulas e lesões necróticas sutis, preservando a matriz tridimensional (RGB) para Transfer Learning.

### 2. Pipeline Assíncrono com `tf.data` e Caching Estratégico (DenseNet121)
* **Resolução Alta (512x512):** Preservação de microdetalhes das pústulas e lesões foliares via decodificação eficiente `tf.io.read_file`.
* **Caching Estratégico em RAM (`.cache()`):** Qualquer pré-processamento determinístico roda apenas na 1ª época e é mantido em memória, garantindo 100% de utilização da GPU da 2ª época em diante.
* **Precisão Mista (`mixed_float16`):** Redução do consumo de VRAM em 50% e aceleração computacional de até 3x na GPU Tesla T4 (Colab) ou RTX, com camada final de classificação em `float32` para estabilidade numérica.
* **Data Augmentation Dinâmico em Lote:** Espelhamentos horizontais/verticais, rotações ($0.15$), zoom ($0.1$) e translações ($0.05$).
* **Escalabilidade Dinâmica de Batch:** 
  $$\text{BATCH\_SIZE} = \text{PER\_REPLICA\_BATCH} \times \text{num\_replicas}$$
  Ou programaticamente: `BATCH_SIZE = PER_REPLICA_BATCH * strategy.num_replicas_in_sync` (ajustado automaticamente para GPU Tesla T4 ou TPU no Google Colab com suporte nativo a `tf.data.AUTOTUNE`).

---

## Arquiteturas Avaliadas e Modelagem

Dentre as 9 famílias de arquiteturas disponíveis no Keras Applications, selecionou-se **um conjunto de duas arquiteturas**, uma desenvolvida por cada integrante da dupla:

| Família | Versão Avaliada | Decisão | Justificativa Técnica para Descarte / Seleção |
| :--- | :--- | :---: | :--- |
| **DenseNet** | `DenseNet121` | **SELECIONADA** | Conexões densas concatenam mapas de todas as camadas. Validada na PoC com excelente convergência e baixo consumo de parâmetros. |
| **ResNet** | `ResNet50` | **SELECIONADA** | Padrão-ouro em conexões residuais aditivas ($x + \mathcal{F}(x)$), que permitem treinar redes profundas sem degradação do gradiente. Utilizada com pesos da ImageNet e *fine-tuning* do último bloco convolucional (`conv5`). |

### Cabeçalho de Classificação Enxuto (Head) — DenseNet121:
```python
x = base_model.output
x = layers.GlobalAveragePooling2D()(x)
x = layers.Dropout(0.3)(x)
outputs = layers.Dense(4, activation='softmax', dtype='float32')(x)
```

### Mecanismos de Treinamento, Pesos e Callbacks — DenseNet121:
* **Pesos de Classe Balanceados (`class_weight`):**
  - Devido ao desbalanceamento severo da classe minoritária `multiple_diseases` (~5% do dataset), calcula-se os pesos inversamente proporcionais às frequências das classes via Scikit-Learn:
    ```python
    from sklearn.utils.class_weight import compute_class_weight

    class_weights = compute_class_weight(
        class_weight='balanced',
        classes=np.unique(y_train_indices),
        y=y_train_indices
    )
    class_weight_dict = dict(enumerate(class_weights))
    ```
  - Passado diretamente em `model.fit(..., class_weight=class_weight_dict)`, penalizando proporcionalmente os erros nas classes mais raras sem necessidade de sintetizar imagens artificiais.
* **`EarlyStopping`:** Monitorando `val_loss` (patience = 8 a 10 épocas, `restore_best_weights=True`) para interromper o treinamento no ponto exato de máxima generalização, evitando sobreajuste.
* **`ReduceLROnPlateau`:** Monitorando `val_loss` (`factor=0.2`, `patience=3`, `min_lr=1e-6`) para desacelerar adaptativamente a taxa de aprendizado quando o ganho na validação entrar em platô.
* **`ModelCheckpoint`:** Monitorando `val_roc_auc` (`mode='max'`, `save_best_only=True`), salvando o modelo e pesos ótimos em formato nativo `.keras` (`models/best_{model_name}.keras`), assegurando a retenção do melhor estado do modelo alcançado ao longo de todas as épocas.

---

## DenseNet121 com Estudo de Ablação CLAHE e Explicabilidade SHAP - `02_densenet121_baseline.ipynb` / `03_densenet121_clahe.ipynb`

Treinamento da `DenseNet121` pré-treinada na ImageNet sob protocolo experimental unificado nos dois cadernos:
1. **Modelo:** `DenseNet121` com *fine-tuning* completo da base convolucional (`trainable=True`).
2. **Hiperparâmetros Fixados:** Resolução `512x512`, `Batch Size = 16`, otimizador `Adam(1e-4)`, precisão mista `mixed_float16`, `Dropout(0.3)`, ponderação `class_weight` balanceada (Scikit-Learn: `healthy`: 0.8814, `multiple_diseases`: 4.9863, `rust`: 0.7324, `scab`: 0.7696).
3. **Estudo de Ablação:** Mesmo regime de treinamento aplicado às imagens naturais (`02_densenet121_baseline.ipynb`) e às imagens com realce CLAHE (`03_densenet121_clahe.ipynb`).
4. **Explicabilidade com SHAP (`GradientExplainer`):** Mesmo protocolo da `ResNet50`, aplicado ao melhor checkpoint de cada caderno (reconstruído em `float32`) sobre um acerto de cada classe, erros da validação e as folhas também explicadas pela `ResNet50`.
5. **Artefatos Gerados:** Curvas de treinamento, matrizes de confusão, curvas ROC e mapas SHAP em `graphs/` (prefixos `densenet121_baseline_` e `densenet121_clahe_`); relatórios de classificação e históricos de treinamento em `models/`.

---

## ResNet50 com Fine-Tuning e Explicabilidade SHAP - `04_resnet50_finetuning.ipynb`

Treinamento da `ResNet50` pré-treinada na ImageNet em duas fases, com avaliação em conjunto de teste isolado e auditoria visual via SHAP:

1. **Divisão Treino / Validação / Teste (70/15/15):** `train_test_split` estratificado em duas etapas. A validação orienta os callbacks e o conjunto de teste é utilizado uma única vez, na avaliação final.
2. **Pipeline `tf.data`:** Resolução $512 \times 512$, normalização `preprocess_input` da ResNet50 e *Data Augmentation* aplicado apenas no treino: espelhamentos horizontais/verticais, rotação ($0.1$), zoom ($0.1$) e brilho ($0.1$).
3. **Pesos de Classe Balanceados (`class_weight`):** Mesmo cálculo via Scikit-Learn descrito acima, restrito ao conjunto de treino.
4. **Cabeçalho de Classificação:** `GlobalAveragePooling2D` → `Dense(256, ReLU)` → `Dropout(0.4)` → `Dense(4, Softmax)`.
5. **Treinamento em Duas Fases:**
   * **Fase 1:** Base convolucional congelada, treinando apenas o cabeçalho com `Adam(1e-3)`.
   * **Fase 2 (*Fine-Tuning*):** Descongelamento do bloco `conv5` com `Adam(1e-5)`, mantendo as camadas de `BatchNormalization` congeladas para preservar as estatísticas da ImageNet.
6. **Callbacks:** `ModelCheckpoint`, `EarlyStopping` e `ReduceLROnPlateau` monitorando `val_loss`.
7. **Métricas no Teste:** Acurácia, *macro F1*, ROC-AUC macro e por classe, e matrizes de confusão absoluta e normalizada, com rótulos traduzidos para o português.
8. **Explicabilidade com SHAP (`GradientExplainer`):** Mapas de contribuição por pixel para as 4 classes, aplicados a um acerto de cada classe e a erros do conjunto de teste, verificando se a decisão se apoia nas lesões foliares.
9. **Artefatos Gerados:** Exemplos de *Data Augmentation*, curvas de treinamento, matrizes de confusão no teste e mapas SHAP em `graphs/` (prefixo `resnet50_`); relatório de classificação do teste em `models/`.

---

## Propostas Futuras e Referências Teóricas - [propostas_futuras.md](propostas_futuras.md)

As formulações metodológicas que foram registradas como trabalhos futuros e referências de extensão estão consolidadas no arquivo [propostas_futuras.md](propostas_futuras.md):

1. **Busca em Grade (*Grid Search*):** Modelagem teórica do espaço cartesiano de 648 combinações de hiperparâmetros e justificativa metodológica para a não realização da busca exaustiva.
2. **Ensemble de Modelos:** Teorema de redução de variância, formulações de fusão (*Soft Voting*, *Weighted Blending* por ROC-AUC, SLSQP e *Stacking*) e pipeline completo de inferência para `data/submission.csv`.
3. **Extensões Competitivas:** Test-Time Augmentation (TTA), validação cruzada em 5 dobras (*5-Fold CV*) e calibração de temperatura (*Temperature Scaling*).

---

## Estrutura do Repositório

```
Leaf-pathology-classification-using-CNN/
│
├── data/                               # Diretório de dados (Kaggle Plant Pathology 2020, não versionado)
│   ├── images/                         # Imagens em alta resolução (.jpg)
│   ├── train.csv                       # Metadados e rótulos de treino (1.821 amostras)
│   ├── test.csv                        # Lista de IDs para teste (1.821 amostras)
│   └── sample_submission.csv           # Formato de submissão da competição
│
├── graphs/                             # Figuras e gráficos (<modelo>_<tipo>.png)
│   ├── densenet121_baseline_curvas.png               # DenseNet121: Curvas de treinamento do baseline natural
│   ├── densenet121_baseline_matriz_confusao.png      # DenseNet121: Matrizes de confusão do baseline natural
│   ├── densenet121_baseline_shap_{n}_{resultado}_{classe}.png # DenseNet121: Mapas SHAP do baseline natural
│   ├── densenet121_clahe_amostras_realce.png         # DenseNet121: Amostras com e sem realce CLAHE
│   ├── densenet121_clahe_curvas.png                  # DenseNet121: Curvas de treinamento com CLAHE
│   ├── densenet121_clahe_matriz_confusao.png         # DenseNet121: Matrizes de confusão com CLAHE
│   ├── densenet121_clahe_curvas_roc.png              # DenseNet121: Curvas ROC One-vs-Rest por classe com CLAHE
│   ├── densenet121_clahe_shap_{n}_{resultado}_{classe}.png # DenseNet121: Mapas SHAP com CLAHE
│   ├── resnet50_augmentation.png                     # ResNet50: Exemplos de Data Augmentation
│   ├── resnet50_curvas.png                           # ResNet50: Curvas de treinamento das duas fases
│   ├── resnet50_matriz_confusao.png                  # ResNet50: Matrizes de confusão no teste
│   └── resnet50_shap_{n}_{resultado}_{classe}.png    # ResNet50: Mapas SHAP de acertos e erros do teste
│
├── models/                             # Relatórios e históricos de treinamento
│   ├── baseline_densenet121_classification_report.csv # DenseNet121: Relatório de métricas por classe
│   ├── baseline_densenet121_history.csv # DenseNet121: Histórico época a época do baseline
│   ├── training_log_clahe.csv          # DenseNet121: Histórico época a época com CLAHE
│   └── resnet50_finetune_classification_report_teste.csv # ResNet50: Relatório de métricas por classe no teste
│
├── 01_analise_exploratoria.ipynb       # Análise Exploratória e testes OpenCV
├── 02_densenet121_baseline.ipynb       # DenseNet121 com Baseline Natural (512x512) e Explicabilidade SHAP
├── 03_densenet121_clahe.ipynb          # DenseNet121 com Realce CLAHE no canal V e Explicabilidade SHAP
├── 04_resnet50_finetuning.ipynb        # ResNet50 com Fine-Tuning em Duas Fases e Explicabilidade SHAP
├── propostas_futuras.md                # Documentação teórica consolidada de Grid Search e Ensemble
│
├── strategy.md                         # Documento de planejamento e decisões técnicas
└── README.md                           # Documentação oficial do projeto
```

---

## Tecnologias e Bibliotecas Utilizadas

- **Linguagem:** Python (v3.11+)
- **Deep Learning:** TensorFlow 2.x / Keras (com `tf.keras.applications` e `tf.data`)
- **Visão Computacional Clássica:** OpenCV (`cv2`)
- **Manipulação de Dados:** Pandas, NumPy
- **Métricas e Pré-processamento:** Scikit-Learn
- **Explicabilidade (XAI):** SHAP
- **Visualização Científica:** Matplotlib, Seaborn, Plotly
- **Ambiente de Aceleração:** Google Colaboratory (GPU Tesla T4 / A100)

---

## Como Executar o Projeto

### 1. Instalação de Dependências
No seu ambiente virtual ou terminal:
```bash
pip install tensorflow opencv-python pandas numpy scikit-learn matplotlib seaborn plotly shap
```

### 2. Execução no Google Colab
Abra os notebooks desejados no Google Colab, certifique-se de selecionar o runtime com **GPU** (*Ambiente de execução $\to$ Alterar tipo de ambiente de execução $\to$ GPU T4*) e monte o Google Drive na primeira célula:
```python
from google.colab import drive
drive.mount('/content/drive')
```

### 3. Sequência Recomendada de Execução
1. **`01_analise_exploratoria.ipynb`:** Inspeção da distribuição de classes, canais de cor e demonstração visual de filtros OpenCV.
2. **`02_densenet121_baseline.ipynb` / `03_densenet121_clahe.ipynb`:** Treinamento da `DenseNet121`, validação do estudo de ablação de pré-processamento (Baseline Natural vs. CLAHE no espaço HSV) e auditoria visual com SHAP.
3. **`04_resnet50_finetuning.ipynb`:** Treinamento da `ResNet50` em duas fases, avaliação no conjunto de teste isolado e auditoria visual com SHAP.
4. **`propostas_futuras.md`:** Consulta à modelagem teórica do Grid Search (648 combinações), formulação matemática do Ensemble e roteiro de extensão futura.

### 4. Modelos Treinados (Releases)
Os modelos treinados não são versionados no repositório (`*.keras` consta no `.gitignore`) e estão disponíveis para download nas [Releases](https://github.com/ClassNeuralNetwork/Leaf-pathology-classification-using-CNN/releases):
* **[v1.0.0 — DenseNet121](https://github.com/ClassNeuralNetwork/Leaf-pathology-classification-using-CNN/releases/tag/v1.0.0):** `baseline_densenet121_best.keras` (baseline natural) e `best_densenet121_clahe.keras` (realce CLAHE).
* **[v1.1.0 — ResNet50](https://github.com/ClassNeuralNetwork/Leaf-pathology-classification-using-CNN/releases/tag/v1.1.0):** `resnet50_finetune_completo.keras` (melhores pesos da Fase 2 de *fine-tuning*).

```python
import tensorflow as tf

model = tf.keras.models.load_model("resnet50_finetune_completo.keras")
```
As imagens de entrada devem receber o mesmo pré-processamento do treinamento: redimensionamento para $512 \times 512$ e o `preprocess_input` correspondente a cada arquitetura.

---

## Referências e Créditos

Este trabalho foi fundamentado e estruturado a partir das técnicas e boas práticas desenvolvidas em dois projetos de referência da comunidade para a competição *Plant Pathology 2020 – FGVC7*:

1. **Image pre processing and DenseNet**
   * **Plataforma / Contexto:** Kaggle (*Plant Pathology 2020 - FGVC7*)
   * **Contribuição aproveitada no projeto:** Metodologia de análise visual e filtragem com OpenCV, incluindo filtros de suavização (`cv2.GaussianBlur`), detecção de bordas morfológicas (`cv2.Canny`) e a aplicação de realce adaptativo de contraste local (**CLAHE**) no espaço de cor HSV como técnica de pré-processamento aliada à `DenseNet121`.

2. **Plant Pathology 2020 - EDA, Models**
   * **Plataforma / Contexto:** Kaggle (*Plant Pathology 2020 - FGVC7*)
   * **Contribuição aproveitada no projeto:** Metodologia de Análise Exploratória (EDA) aprofundada com distribuição estatística dos canais de cor RGB, pipeline assíncrono de alto rendimento em `tf.data.Dataset`, estratégias de *Transfer Learning* sobre múltiplos backbones modernos (`DenseNet`, `EfficientNet`, `NoisyStudent`) e a arquitetura de fusão de probabilidades em *Ensemble*.

3. **Competição Oficial e Dataset:**
   * Thapa, R., Zhang, K., Snavely, N., Belongie, S., & Khan, A. (2020). *Plant Pathology 2020 – FGVC7 Challenge Dataset*. CVPR 2020 Workshop on Fine-Grained Visual Categorization. [Kaggle Competition](https://www.kaggle.com/c/plant-pathology-2020-fgvc7).

4. **Metodologia de Explicabilidade (SHAP):**
   * Lundberg, S. M., & Lee, S.-I. (2017). *A Unified Approach to Interpreting Model Predictions*. Advances in Neural Information Processing Systems 30 (NeurIPS 2017), pp. 4765-4774.

---

## Autores

<table align="center">
  <tr> 
    <td align="center">
      <a href="https://github.com/AndersonCSM">
        <img src="https://avatars.githubusercontent.com/u/9919?v=4" width="120px;" alt="Foto do Autor"/><br>
        <sub>
          <b>Anderson</b>
         </sub>
      </a>
    </td>
    <td align="center">
      <a href="https://github.com/pablonascimento1">
        <img src="https://avatars.githubusercontent.com/u/159596872?v=4" width="120px;" alt="Foto do Autor"/><br>
        <sub>
          <b>Pablo</b>
         </sub>
      </a>
    </td>
  </tr>
</table>

---

## Licença

Este projeto está disponibilizado sob a **Licença MIT**.  
O dataset *Plant Pathology 2020 – FGVC7* é de propriedade do concurso CVPR 2020 e está sujeito aos termos de uso da competição Kaggle.

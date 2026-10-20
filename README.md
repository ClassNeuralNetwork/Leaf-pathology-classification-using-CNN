# Classificação de Patologias em Folhas com Redes Neurais Convolucionais e Explicabilidade (Grad-CAM)

Este projeto foi desenvolvido como parte da disciplina de **Redes Neurais Artificiais**, na Universidade Federal Rural do Semi-Árido (UFERSA).

---

## Abstract

Automated foliar disease diagnosis is vital for precision agriculture, early crop pathogen intervention, and targeted fungicide management. In apple orchards (*Malus domestica*), fungal infections such as cedar apple rust and apple scab cause devastating yield losses if left undetected, yet open-canopy imagery introduces challenging illumination variations, complex backgrounds, overlapping foliage, and co-occurring pathologies. This project presents an end-to-end deep learning framework developed on the *Plant Pathology 2020 – FGVC7* challenge dataset (CVPR 2020), classifying leaves into four mutually exclusive categories: *healthy*, *rust*, *scab*, and *multiple diseases*. 

To overcome the severe representation deficit of the co-infection class (~5% of samples), a balanced class-weighting formulation is embedded directly into the categorical cross-entropy loss without synthetic distortion. The data pipeline leverages high-resolution representations ($512 \times 512$), asynchronous `tf.data` prefetching, GPU-accelerated stochastic augmentation, and in-memory tensor caching paired with automatic mixed-precision acceleration (`mixed_float16`). We evaluate a morphological ablation study comparing raw natural baselines against HSV-space Contrast Limited Adaptive Histogram Equalization (CLAHE), followed by a unified benchmarking protocol across five topologically distinct convolutional architectures (`DenseNet121`, `EfficientNetV2B0`, `ResNet50V2`, `ConvNeXtTiny`, and `Xception`). Finally, model predictions are audited using Explainable Artificial Intelligence (XAI) via Grad-CAM and Grad-CAM++, visually confirming that convolutional attention maps localize genuine botanical lesions (rust pustules and necrotic spots) rather than spurious background artifacts.

**Keywords:** Plant Pathology, Deep Learning, Convolutional Neural Networks (CNN), Explainable AI (XAI), Grad-CAM, Class Imbalance, Precision Agriculture.

---

## Introdução e Contexto

A agricultura de precisão e o monitoramento automatizado de lavouras desempenham um papel vital na segurança alimentar global e na redução do desperdício de defensivos agrícolas. Na fruticultura, especificamente na cultura da maçã (*Malus domestica*), doenças foliares fúngicas como a **ferrugem** (*cedar apple rust*) e a **sarna** (*apple scab*) são responsáveis por perdas drásticas de produtividade se não identificadas em estágio inicial.

O desafio reside na complexidade visual das imagens em ambiente aberto: variações severas de iluminação natural, sombras, folhas sobrepostas e a ocorrência simultânea de múltiplas patologias na mesma folha tornam o diagnóstico visual humano subjetivo e demorado.

---

## Objetivo

Desenvolver e comparar modelos de Redes Neurais Convolucionais profundas capazes de diagnosticar o estado fitopatológico de folhas de macieira com alta acurácia e generalização, garantindo:

1. **Preservação de Rigor Anti-Vazamento:** Divisão estratificada de treino e validação prévia a qualquer transformação.
2. **Estudos de Ablação de Pré-processamento:** Avaliação comparativa entre baseline natural (`single`), realce adaptativo CLAHE no canal V do HSV (`improved`).
3. **Benchmarking Sistemático de Arquiteturas:** Comparação entre famílias arquiteturais consagradas sob hiperparâmetros fixados (`DenseNet121`, `EfficientNetV2B0`, `ResNet50V2`, `ConvNeXtTiny`, `Xception`).
4. **Auditoria Visual (*XAI*):** Mapeamento térmico via **Grad-CAM / Grad-CAM++** (`04_xai_gradcam.ipynb`) para certificar que a rede toma decisões fundamentadas nas lesões das folhas e não em ruídos de fundo.
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

> **Desbalanceamento:** A extrema raridade da classe `multiple_diseases` (~5%) exige mecanismos ativos como divisão estratificada (`StratifiedKFold`), pesos de classe (`class_weight`) para evitar que a rede se torne cega à classe minoritária.

---

## Metodologia e Pipeline de Dados

```
[Imagens Brutas em Alta Resolução (512x512)]
       ↓
[01. EDA: Análise Morfológica e Filtros OpenCV (GaussianBlur, Canny, CLAHE)]
       ↓
[02. PoC & Estudos de Ablação: Baseline Natural (02_pipeline_single.ipynb) vs. CLAHE (02_pipeline_improved.ipynb)]
       ↓
[Pipeline tf.data: Caching Estratégico em RAM + Data Augmentation em Batch + Mixed Precision (float16)]
       ↓
[Base Convolucional Fine-Tuned (trainable=True, lr=1e-4) + GAP + Dropout(0.3) + Dense(4, float32)]
       ↓
[03. Benchmarking de Arquiteturas: DenseNet121, EfficientNetV2B0, ResNet50V2, ConvNeXtTiny, Xception]
       ↓
[04. Explicabilidade Visual com Grad-CAM / Grad-CAM++ (04_xai_gradcam.ipynb)]
       ↓
[Documentação Teórica de Propostas Futuras: Grid Search e Ensemble (propostas_futuras.md)]
```

### 1. Pré-processamento e Estudos de Ablação
* **Análise Morfológica e Filtros (na EDA):** `cv2.GaussianBlur` para atenuação de ruído e `cv2.Canny` para destacar bordas das pústulas.
* **Estudo de Ablação no Pipeline:**
  * **Baseline (`02_pipeline_single.ipynb`):** Imagens naturais sem filtros adicionais em alta resolução ($512 \times 512$).
  * **Realce Adaptativo CLAHE (`02_pipeline_improved.ipynb`):** Equalização adaptativa local no canal de luminância ($V$) do espaço de cor HSV para destacar pústulas e lesões necróticas sutis, preservando a matriz tridimensional (RGB) para Transfer Learning.

### 2. Pipeline Assíncrono com `tf.data` e Caching Estratégico
* **Resolução Alta (512x512):** Preservação de microdetalhes das pústulas e lesões foliares via decodificação eficiente `tf.io.read_file`.
* **Caching Estratégico em RAM (`.cache()`):** Qualquer pré-processamento determinístico roda apenas na 1ª época e é mantido em memória, garantindo 100% de utilização da GPU da 2ª época em diante.
* **Precisão Mista (`mixed_float16`):** Redução do consumo de VRAM em 50% e aceleração computacional de até 3x na GPU Tesla T4 (Colab) ou RTX, com camada final de classificação em `float32` para estabilidade numérica.
* **Data Augmentation Dinâmico em Lote:** Espelhamentos horizontais/verticais, rotações ($0.15$), zoom ($0.1$) e translações ($0.05$).
* **Escalabilidade Dinâmica de Batch:** 
  $$\text{BATCH\_SIZE} = \text{PER\_REPLICA\_BATCH} \times \text{num\_replicas}$$
  Ou programaticamente: `BATCH_SIZE = PER_REPLICA_BATCH * strategy.num_replicas_in_sync` (ajustado automaticamente para GPU Tesla T4 ou TPU no Google Colab com suporte nativo a `tf.data.AUTOTUNE`).

---

## Arquiteturas Avaliadas e Modelagem

Dentre as 9 famílias de arquiteturas disponíveis no Keras Applications, selecionou-se **um conjunto de cinco arquiteturas** para análise e triagem técnica:

| Família | Versão Avaliada | Decisão | Justificativa Técnica para Descarte / Seleção |
| :--- | :--- | :---: | :--- |
| **DenseNet** | `DenseNet121` | **BASE** | Conexões densas concatenam mapas de todas as camadas. Validada na PoC com excelente convergência e baixo consumo de parâmetros. Modelo baseline de referência para o benchmarking. |
| **EfficientNetV2** | `EfficientNetV2B0` | **SELECIONADA** | Evolução da V1. Substitui as primeiras camadas por **Fused-MBConv**, acelerando o treino na GPU, além de incorporar atenção de canal (*Squeeze-and-Excitation*). Alta complementaridade com a DenseNet. |
| **ResNet** | `ResNet50V2` | **SELECIONADA** | Padrão-ouro em conexões residuais aditivas ($x + \mathcal{F}(x)$). A versão V2 utiliza pré-ativação (BN e ReLU antes das convoluções), garantindo fluxo direto de gradientes. Erros ortogonais aos da DenseNet. |
| **ConvNeXt** | `ConvNeXtTiny` | **SELECIONADA** | Representa o estado da arte convolucional (2022). Incorpora conceitos de *Vision Transformers* (kernels grandes 7x7, ativação GELU, LayerNorm), capturando contexto espacial amplo das folhas de macieira. |
| **Xception** | `Xception` | **SELECIONADA** | Convoluções separáveis em profundidade com desacoplamento espacial/canal. Excelente capacidade de extração com parametrização eficiente. |

### Cabeçalho de Classificação Enxuto (Head):
```python
x = base_model.output
x = layers.GlobalAveragePooling2D()(x)
x = layers.Dropout(0.3)(x)
outputs = layers.Dense(4, activation='softmax', dtype='float32')(x)
```

### Mecanismos de Treinamento, Pesos e Callbacks:
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

## Benchmarking de Arquiteturas Pré-Treinadas - `03_benchmark_models.ipynb`

Avaliação sistemática dos backbones selecionados sob protocolo experimental unificado:
1. **Modelos Avaliados:** `DenseNet121` (referência PoC), `EfficientNetV2B0`, `ResNet50V2`, `ConvNeXtTiny` e `Xception`.
2. **Hiperparâmetros Fixados:** Resolução `512x512`, `Batch Size = 16`, otimizador `Adam(1e-4)`, precisão mista `mixed_float16`, `Dropout(0.3)`, ponderação `class_weight` balanceada (Scikit-Learn: `healthy`: 0.8814, `multiple_diseases`: 4.9863, `rust`: 0.7324, `scab`: 0.7696).
3. **Métricas Consolidadas:** Geração de `models/models_benchmark_comparison.csv` e `graphs/models_benchmark_comparison.png` contendo parâmetros, tempo/época, `val_loss`, `val_accuracy`, `macro_roc_auc` e `macro_f1`.

---

## Inteligência Artificial Explicável (XAI) - `04_xai_gradcam.ipynb`

Para validar a confiabilidade do diagnóstico automatizado, implementa-se o algoritmo **Grad-CAM (Gradient-weighted Class Activation Mapping)** e **Grad-CAM++**:

1. **Cálculo de Gradientes:** Extração dos gradientes da pontuação da classe predita em relação aos mapas de ativação da última camada convolucional dos modelos campeões.
2. **Ponderação e ReLU:** Média global dos gradientes gerando os pesos de importância de cada canal, seguida de retificação ReLU para evidenciar apenas as feições com contribuição positiva.
3. **Sobreposição de Mapa de Calor (*Heatmap*):** Mapas térmicos normalizados sobrepostos às fotografias originais em alta resolução.
4. **Auditoria Qualitativa:** Confirmação visual de que o modelo foca nas lesões foliares reais (pústulas alaranjadas na ferrugem e manchas necróticas na sarna), ignorando artefatos de fundo (solo, dedos, hastes).

---

## Propostas Futuras e Referências Teóricas - [propostas_futuras.md](propostas_futuras.md)

As formulações metodológicas que foram registradas como trabalhos futuros e referências de extensão estão consolidadas no arquivo [propostas_futuras.md](propostas_futuras.md):

1. **Busca em Grade (*Grid Search*):** Modelagem teórica do espaço cartesiano de 648 combinações de hiperparâmetros e justificativa metodológica de por que a busca exaustiva foi substituída pelo benchmarking direcionado de backbones.
2. **Ensemble de Modelos:** Teorema de redução de variância, formulações de fusão (*Soft Voting*, *Weighted Blending* por ROC-AUC, SLSQP e *Stacking*) e pipeline completo de inferência para `data/submission.csv`.
3. **Extensões Competitivas:** Test-Time Augmentation (TTA), validação cruzada em 5 dobras (*5-Fold CV*) e calibração de temperatura (*Temperature Scaling*).

---

## Estrutura do Repositório

```
neural-networks/unidade-2/projeto/
│
├── data/                               # Diretório de dados (Kaggle Plant Pathology 2020)
│   ├── images/                         # Imagens em alta resolução (.jpg)
│   ├── train.csv                       # Metadados e rótulos de treino (1.821 amostras)
│   ├── test.csv                        # Lista de IDs para teste (1.821 amostras)
│   ├── sample_submission.csv           # Formato de submissão da competição
│   └── submission.csv                  # Submissão oficial formatada para o Kaggle
│
├── graphs/                             # Figuras e gráficos em alta resolução (300 DPI)
│   ├── loss_curve_{modelo}.png         # Curvas de perda treino vs validação
│   ├── accuracy_auc_curve_{modelo}.png # Curvas de acurácia e ROC-AUC
│   ├── confusion_matrix_{modelo}.png   # Matrizes absoluta e normalizada
│   ├── roc_curve_multiclass_{modelo}.png # Curvas ROC One-vs-Rest por classe
│   └── models_benchmark_comparison.png # Gráfico comparativo geral de backbones
│
├── models/                             # Checkpoints e históricos de treinamento
│   ├── best_{modelo}.keras             # Pesos ótimos salvos pelo ModelCheckpoint
│   ├── history_{modelo}.csv            # Histórico época a época de cada execução
│   ├── classification_report_{modelo}.csv # Relatório de métricas por classe
│   └── models_benchmark_comparison.csv # Tabela comparativa do benchmarking
│
├── references/                         # Cadernos de referência do Kaggle
│   ├── image-pre-processing-and-densenet.ipynb
│   └── plant-pathology-2020-eda-models.ipynb
│
├── 01_eda.ipynb                        # Etapa 1: Análise Exploratória e testes OpenCV
├── 02_pipeline_single.ipynb            # Etapa 2: PoC e Baseline Natural (512x512)
├── 02_pipeline_improved.ipynb          # Etapa 2: Estudo de Ablação (Realce CLAHE no canal V)
├── 03_benchmark_models.ipynb           # Etapa 3: Benchmarking das 5 arquiteturas selecionadas
├── 04_xai_gradcam.ipynb                # Etapa 4: Explicabilidade visual (Grad-CAM / Grad-CAM++)
├── propostas_futuras.md                # Documentação teórica consolidada de Grid Search e Ensemble
│
├── pre_training.py                     # Módulo Python com o catálogo de modelos Keras
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
- **Visualização Científica:** Matplotlib, Seaborn, Plotly
- **Ambiente de Aceleração:** Google Colaboratory (GPU Tesla T4 / A100)

---

## Como Executar o Projeto

### 1. Instalação de Dependências
No seu ambiente virtual ou terminal:
```bash
pip install tensorflow opencv-python pandas numpy scikit-learn matplotlib seaborn plotly
```

### 2. Execução no Google Colab
Abra os notebooks desejados no Google Colab, certifique-se de selecionar o runtime com **GPU** (*Ambiente de execução $\to$ Alterar tipo de ambiente de execução $\to$ GPU T4*) e monte o Google Drive na primeira célula:
```python
from google.colab import drive
drive.mount('/content/drive')
```

### 3. Sequência Recomendada de Execução
1. **`01_eda.ipynb`:** Inspeção da distribuição de classes, canais de cor e demonstração visual de filtros OpenCV.
2. **`02_pipeline_single.ipynb` / `02_pipeline_improved.ipynb`:** PoC e validação do estudo de ablação de pré-processamento (Baseline Natural vs. CLAHE no espaço HSV).
3. **`03_benchmark_models.ipynb`:** Treinamento unificado e benchmarking comparativo dos 5 backbones selecionados (`DenseNet121`, `EfficientNetV2B0`, `ResNet50V2`, `ConvNeXtTiny`, `Xception`).
4. **`04_xai_gradcam.ipynb`:** Auditoria visual das regiões de atenção com Grad-CAM / Grad-CAM++ sobre os melhores modelos campeões.
5. **`propostas_futuras.md`:** Consulta à modelagem teórica do Grid Search (648 combinações), formulação matemática do Ensemble e roteiro de extensão futura.

---

## Referências e Créditos

Este trabalho foi fundamentado e estruturado a partir das técnicas e boas práticas desenvolvidas em dois projetos de referência da comunidade para a competição *Plant Pathology 2020 – FGVC7*:

1. **[Image pre processing and DenseNet](references/image-pre-processing-and-densenet.ipynb)**
   * **Plataforma / Contexto:** Kaggle (*Plant Pathology 2020 - FGVC7*)
   * **Contribuição aproveitada no projeto:** Metodologia de análise visual e filtragem com OpenCV, incluindo filtros de suavização (`cv2.GaussianBlur`), detecção de bordas morfológicas (`cv2.Canny`) e a aplicação de realce adaptativo de contraste local (**CLAHE**) no espaço de cor HSV como técnica de pré-processamento aliada à `DenseNet121`.

2. **[Plant Pathology 2020 - EDA, Models](references/plant-pathology-2020-eda-models.ipynb)**
   * **Plataforma / Contexto:** Kaggle (*Plant Pathology 2020 - FGVC7*)
   * **Contribuição aproveitada no projeto:** Metodologia de Análise Exploratória (EDA) aprofundada com distribuição estatística dos canais de cor RGB, pipeline assíncrono de alto rendimento em `tf.data.Dataset`, estratégias de *Transfer Learning* sobre múltiplos backbones modernos (`DenseNet`, `EfficientNet`, `NoisyStudent`) e a arquitetura de fusão de probabilidades em *Ensemble*.

3. **Competição Oficial e Dataset:**
   * Thapa, R., Zhang, K., Snavely, N., Belongie, S., & Khan, A. (2020). *Plant Pathology 2020 – FGVC7 Challenge Dataset*. CVPR 2020 Workshop on Fine-Grained Visual Categorization. [Kaggle Competition](https://www.kaggle.com/c/plant-pathology-2020-fgvc7).

4. **Metodologia de Explicabilidade (Grad-CAM):**
   * Selvaraju, R. R., Cogswell, M., Das, A., Vedaldi, A., Parikh, D., & Batra, D. (2017). *Grad-CAM: Visual Explanations from Deep Networks via Gradient-Based Localization*. IEEE International Conference on Computer Vision (ICCV), pp. 618-626.

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
      <a href="#">
        <img src="https://avatars.githubusercontent.com/u/0?v=4" width="120px;" alt="Foto do Coautor"/><br>
        <sub>
          <b>Coautor</b>
         </sub>
      </a>
    </td>
  </tr>
</table>

---

## Licença

Este projeto está disponibilizado sob a **Licença MIT**.  
O dataset *Plant Pathology 2020 – FGVC7* é de propriedade do concurso CVPR 2020 e está sujeito aos termos de uso da competição Kaggle.
